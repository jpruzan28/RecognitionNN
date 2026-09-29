"""
webcam_demo.py  --  Run the trained box locator live on your webcam.

Every frame from the camera goes through the same steps as a test image in
object_locator.py: shrink to 224x224, normalize, run the network, get 4 numbers
(a box in 0..1 fractions), and draw that box on the full-size frame.

Two ways to start it:
  python object_locator.py --webcam   # train first, then open the webcam
  python webcam_demo.py               # skip training; load the saved box_regressor_faces.pt
  python webcam_demo.py --weights box_regressor_pets.pt   # the pet-head model instead

Controls (click the video window first so it receives your key presses):
  q or Esc   quit
  m          toggle mirror view (selfie-style) on/off

Things to know when testing:
  - The model was trained on photos with exactly ONE face. With two people in view
    it has no concept of "two", so its box may land on one face or between them.
  - The network ALWAYS outputs exactly one box. It has no way to say "nothing here",
    because in training every image contained exactly one face. So with nobody in
    view, it still draws its best guess somewhere. Real detectors add a
    "confidence" output to handle this.
"""

import argparse                         # command-line options (--camera, --weights)
import sys                              # detect Windows, to pick the fast camera backend
import time                            # measure frames per second

import cv2                              # OpenCV: talks to the webcam + shows a window
import numpy as np
import torch

# Reuse the exact model and preprocessing used in training. If the webcam frames
# were prepared even slightly differently (other normalization, BGR vs RGB), the
# network would see inputs unlike anything it trained on, and its boxes would be worse.
from object_locator import BoxRegressor, prepare, IMG_SIZE


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


@torch.no_grad()     # inference only: no gradients -> faster, less memory
def predict_box(model, frame_bgr, device):
    """Run the network on one camera frame. Returns the box as (x1, y1, x2, y2) in 0..1."""
    # OpenCV delivers pixels in BGR order (a historical quirk); the network was
    # trained on RGB, so swap the channel order first.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Squash to 224x224, just like the training images (aspect ratio is not kept
    # there either, so we must do the same here).
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    # (H, W, C) uint8 -> (1, C, H, W): the layout PyTorch expects, batch of one.
    x = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        box = model(prepare(x, device))
    return box.float()[0].cpu().numpy()


def draw_overlay(frame, box, fps):
    """Draw the predicted box and an info line onto the frame (in place)."""
    h, w = frame.shape[:2]
    # The box is in fractions of width/height, so scaling to the full camera
    # resolution is just a multiplication.
    x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)   # OpenCV colors are BGR: this is green
    cv2.putText(frame, f"{fps:4.1f} FPS   q = quit   m = mirror", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def run_webcam(model=None, device=None, camera=0, weights="box_regressor_faces.pt"):
    """Open the webcam and show live predictions until the user quits.

    object_locator.py calls this with its freshly trained model. Run on its own,
    it builds the model and loads the weights saved by the last training run.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = BoxRegressor().to(device)
        # map_location lets weights saved on the GPU load on a CPU-only machine too.
        model.load_state_dict(torch.load(weights, map_location=device))
        print(f"Loaded trained weights from {weights}")
    model.eval()        # important: BatchNorm must use its stored statistics, not the frame's

    cap = open_camera(camera)
    window = "Box locator - live (q to quit)"
    mirror = True       # a mirrored view feels natural when you're facing the camera
    fps, last = 0.0, time.time()
    print("Webcam running. Click the video window, then press q or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()                  # grab one frame (BGR, e.g. 640x480)
            if not ok:
                print("Camera stopped sending frames.")
                break
            if mirror:
                # Flip BEFORE predicting, so the box is computed on the same
                # image that is displayed and lines up with it.
                frame = cv2.flip(frame, 1)

            box = predict_box(model, frame, device)

            # Smoothed frames-per-second: mostly the old value, a little of the new one,
            # so the number on screen doesn't flicker.
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            draw_overlay(frame, box, fps)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF             # also lets the window redraw; waits 1 ms
            if key in (ord("q"), 27):               # 27 = Esc
                break
            if key == ord("m"):
                mirror = not mirror
            # Closing the window with its X button should also quit.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Always release the camera, even after an error, so other apps can use it.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam demo of the trained box locator.")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (0 = default camera)")
    parser.add_argument("--weights", type=str, default="box_regressor_faces.pt",
                        help="trained weights file (box_regressor_pets.pt for the pet-head model)")
    args = parser.parse_args()
    run_webcam(camera=args.camera, weights=args.weights)
