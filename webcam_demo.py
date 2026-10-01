"""
webcam_demo.py  --  Find a face live on your webcam, and recognize whether it's YOU.

Two networks run on every camera frame:
  1. WHERE: our fine-tuned ResNet-18 from object_locator.py draws a box around the face.
  2. WHO:   FaceNet (see face_id.py) turns the boxed face into 512 numbers and compares
            them to the people you enrolled. The box is labeled with the name, or "Unknown".

Step 1 - enroll yourself (once, about 15 seconds):
  python webcam_demo.py --enroll Jasmine
  Look at the camera, press SPACE, then slowly turn your head a little while it takes
  40 photos. It then measures how well it can tell you apart from 500 strangers, saves
  your profile to faces_db/Jasmine/, and goes straight to the live view.

Step 2 - live recognition (any time after):
  python webcam_demo.py
  Enroll more people with --enroll OtherName; it labels each face with the best match.

Other ways to start:
  python object_locator.py --webcam   # retrain the face locator first, then go live
  python webcam_demo.py --weights box_regressor_pets.pt   # pet-head boxes (no recognition)

Controls (click the video window first so it receives your key presses):
  q or Esc   quit
  m          toggle mirror view (selfie-style) on/off

Limitations to know about:
  - ONE face at a time: the locator outputs exactly one box. With two people in view,
    it may box one of them or land between them.
  - The locator ALWAYS outputs a box, even with nobody in view. A box around a wall
    will (correctly) be labeled "Unknown".
  - Recognition is only as good as the enrollment: enroll in the lighting you'll use,
    and re-enroll if you change your look a lot (glasses, beard).
  - This is a learning project, not a security system: a photo of you on a phone would
    be recognized as you.
"""

import argparse                         # command-line options (--camera, --weights, --enroll)
import sys                              # detect Windows, to pick the fast camera backend
import time                             # measure frames per second, pace enrollment photos

import cv2                              # OpenCV: talks to the webcam + shows a window
import numpy as np
import torch

# Reuse the exact model and preprocessing used in training. If the webcam frames
# were prepared even slightly differently (other normalization, BGR vs RGB), the
# network would see inputs unlike anything it trained on, and its boxes would be worse.
from object_locator import BoxRegressor, prepare, IMG_SIZE
import face_id

# OpenCV colors are BGR (blue, green, red), not RGB.
GREEN, ORANGE, YELLOW, WHITE = (0, 200, 0), (0, 140, 255), (0, 220, 255), (255, 255, 255)


def open_camera(index):
    """Open webcam number `index` (0 = the built-in one). Returns an OpenCV capture."""
    # On Windows the DirectShow backend (CAP_DSHOW) opens the camera much faster
    # than the default one. Elsewhere, let OpenCV pick.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Is another app (Zoom, Teams, "
                           f"Camera) using it? Try --camera 1 if you have more than one.")
    return cap


def load_locator(weights, device):
    """Build the box network and load the weights saved by object_locator.py."""
    model = BoxRegressor().to(device)
    # map_location lets weights saved on the GPU load on a CPU-only machine too.
    model.load_state_dict(torch.load(weights, map_location=device))
    print(f"Loaded face locator weights from {weights}")
    return model.eval()  # important: BatchNorm must use its stored statistics, not the frame's


@torch.no_grad()     # inference only: no gradients -> faster, less memory
def predict_box(model, frame_rgb, device):
    """Run the box network on one RGB camera frame. Returns (x1, y1, x2, y2) in 0..1."""
    # Squash to 224x224, just like the training images (aspect ratio is not kept
    # there either, so we must do the same here).
    small = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    # (H, W, C) uint8 -> (1, C, H, W): the layout PyTorch expects, batch of one.
    x = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        box = model(prepare(x, device))
    return box.float()[0].cpu().numpy()


def read_frame(cap, mirror):
    """Grab one frame. Returns (frame_bgr for display, frame_rgb for the networks), or None."""
    ok, frame = cap.read()                      # BGR, e.g. 640x480
    if not ok:
        return None
    if mirror:
        # Flip BEFORE predicting, so the box is computed on the same image that is
        # displayed and lines up with it.
        frame = cv2.flip(frame, 1)
    # OpenCV delivers pixels in BGR order (a historical quirk); both networks were
    # trained on RGB, so they get a channel-swapped copy.
    return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_box(frame, box, color, label=None):
    """Draw a 0..1 box (and an optional text label above it) onto the frame, in place."""
    h, w = frame.shape[:2]
    # The box is in fractions of width/height, so scaling to the full camera
    # resolution is just a multiplication.
    x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    if label:
        # A filled strip behind the text keeps it readable on any background.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ty = max(y1, th + 12)
        cv2.rectangle(frame, (x1, ty - th - 12), (x1 + tw + 10, ty), color, -1)
        cv2.putText(frame, label, (x1 + 5, ty - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def draw_text(frame, text, row=0):
    """Write a line of white text with a dark outline at the top-left of the frame."""
    y = 28 + row * 30
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def window_closed(window):
    """True if the user closed the window with its X button."""
    return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1


# ---------------------------------------------------------------------------
# Enrollment: collect photos of one person's face
# ---------------------------------------------------------------------------
def enroll_from_webcam(name, locator, embedder, device, camera, n_photos=40, interval=0.15):
    """Show the webcam, wait for SPACE, then take n_photos face crops and save a profile.

    Small head movements between photos matter: 40 nearly identical photos teach less
    than 40 photos from slightly different angles. That variety is what lets
    recognition still work when you turn your head later.
    """
    cap = open_camera(camera)
    window = f"Enrolling {name}"
    crops, capturing, last_shot = [], False, 0.0
    try:
        while len(crops) < n_photos:
            frames = read_frame(cap, mirror=True)
            if frames is None:
                print("Camera stopped sending frames.")
                return False
            frame, rgb = frames
            box = predict_box(locator, rgb, device)

            if capturing and time.time() - last_shot >= interval:
                crop = face_id.crop_face(rgb, box)
                if crop is not None:
                    crops.append(crop)
                    last_shot = time.time()

            draw_box(frame, box, YELLOW)
            if capturing:
                draw_text(frame, f"Capturing {len(crops)}/{n_photos}")
                draw_text(frame, "Slowly turn your head a little: left, right, up, down", row=1)
            else:
                draw_text(frame, f"Enrolling {name}: center your face in the box")
                draw_text(frame, "Press SPACE to start   (Esc = cancel)", row=1)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:                       # 32 = SPACE
                capturing = True
            if key in (ord("q"), 27) or window_closed(window):
                print("Enrollment cancelled.")
                return False
    finally:
        cap.release()
        cv2.destroyAllWindows()

    face_id.save_profile(name, crops, embedder, device)
    return True


# ---------------------------------------------------------------------------
# Live view: box + name on every frame
# ---------------------------------------------------------------------------
def run_webcam(model=None, device=None, camera=0, weights="box_regressor_faces.pt",
               recognize=True, enroll=None):
    """Open the webcam and show live boxes (and names) until the user quits.

    object_locator.py calls this with its freshly trained model. Run on its own,
    it loads the weights saved by the last training run.
    recognize=False skips the WHO step (used for the pet-head model).
    enroll="Name" first enrolls that person, then starts the live view.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locator = model.eval() if model is not None else load_locator(weights, device)

    embedder, profiles = None, []
    if recognize:
        embedder = face_id.load_embedder(device)
        if enroll and not enroll_from_webcam(enroll, locator, embedder, device, camera):
            return
        profiles = face_id.load_profiles()
        names = ", ".join(p["name"] for p in profiles) or "nobody yet"
        print(f"Enrolled: {names}")

    cap = open_camera(camera)
    window = "Face recognition - live (q to quit)"
    mirror = True       # a mirrored view feels natural when you're facing the camera
    fps, last = 0.0, time.time()
    smooth_emb = None   # running average of recent face embeddings (see below)
    print("Webcam running. Click the video window, then press q or Esc to quit.")

    try:
        while True:
            frames = read_frame(cap, mirror)
            if frames is None:
                print("Camera stopped sending frames.")
                break
            frame, rgb = frames
            box = predict_box(locator, rgb, device)           # 1. WHERE is the face?

            if not recognize:
                draw_box(frame, box, GREEN)
            elif not profiles:
                draw_box(frame, box, GREEN)
                draw_text(frame, "Nobody enrolled: run  python webcam_demo.py --enroll YourName", row=1)
            else:
                crop = face_id.crop_face(rgb, box)            # 2. WHO is it?
                if crop is not None:
                    emb = face_id.embed(embedder, [crop], device)[0]
                    # Average the embedding over recent frames. A single frame can be
                    # blurry or mid-blink, which would make the label flicker between
                    # your name and "Unknown". Mostly old + a little new = smooth.
                    smooth_emb = emb if smooth_emb is None else \
                        torch.nn.functional.normalize(0.7 * smooth_emb + 0.3 * emb, dim=0)
                    name, score = face_id.identify(smooth_emb, profiles)
                    if name:
                        draw_box(frame, box, GREEN, f"{name}  {score:.2f}")
                    else:
                        draw_box(frame, box, ORANGE, f"Unknown  {score:.2f}")

            # Smoothed frames-per-second, so the number on screen doesn't flicker.
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            draw_text(frame, f"{fps:4.1f} FPS   q = quit   m = mirror")
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF             # also lets the window redraw; waits 1 ms
            if key in (ord("q"), 27):               # 27 = Esc
                break
            if key == ord("m"):
                mirror = not mirror
            if window_closed(window):
                break
    finally:
        # Always release the camera, even after an error, so other apps can use it.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam face finding + recognition.")
    parser.add_argument("--enroll", type=str, default=None, metavar="NAME",
                        help="first take photos of NAME's face and remember them")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (0 = default camera)")
    parser.add_argument("--weights", type=str, default="box_regressor_faces.pt",
                        help="trained locator weights (box_regressor_pets.pt for the pet-head model)")
    args = parser.parse_args()
    run_webcam(camera=args.camera, weights=args.weights,
               recognize="pets" not in args.weights, enroll=args.enroll)
