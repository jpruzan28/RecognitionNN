"""
face_id.py  --  Recognize WHO a face belongs to: you, or someone unknown.

object_locator.py answers "WHERE is the face?" (a box). This file answers "WHOSE face
is it?" It does not draw anything or touch the camera; webcam_demo.py does that and
calls the functions here.

The network: FaceNet (InceptionResnetV1), already trained on VGGFace2, 3.3 million
photos of 9,000 people. It turns a face photo into 512 numbers, an "embedding".
It was trained so that:
  - two photos of the SAME person give embeddings that point in nearly the same direction,
  - photos of DIFFERENT people give embeddings that point in different directions.
So it has learned a "face space" where distance means "how different are these people".

Why this beats training a "you vs. not you" classifier from scratch:
  A classifier would need thousands of photos of you, plus thousands of other people,
  and could easily learn shortcuts ("webcam lighting = you"). FaceNet already knows
  what makes faces differ. To recognize you, we only need to know WHERE you are in its
  face space: the average embedding of ~40 webcam photos of you. This is called
  one-shot (or few-shot) learning, and there is no training step at all.

How a live face is recognized:
  1. Crop the face (using the box from object_locator's network) and resize to 160x160.
  2. FaceNet -> 512-number embedding, scaled to length 1.
  3. Cosine similarity with your average embedding: a single DOT PRODUCT
     (1 = same direction = surely you, 0 = unrelated).
  4. Above the threshold -> your name. Below -> "Unknown".

The threshold is not guessed: calibrate_threshold() measures how similar your own photos
are to your average, and how similar ~500 strangers' faces from WIDER FACE are, and puts
the cut between the two groups.

Optics note: step 3 is a dot product, one row of a matrix multiplication, exactly the
kind of operation optics does naturally. Comparing a face against N enrolled people at
once is a single matrix-vector product.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")                   # draw charts to files, not windows
import matplotlib.pyplot as plt

# FaceNet's code ships in the facenet-pytorch package. It was installed with
# `pip install --no-deps facenet-pytorch`: its declared requirements pin old versions
# (torch 2.2, numpy 1.x) that would replace the CUDA PyTorch, but the model code
# itself is plain PyTorch and runs fine on newer versions.
from facenet_pytorch import InceptionResnetV1


FACE_SIZE = 160          # FaceNet was trained on 160x160 face crops
MARGIN = 0.2             # widen each box by 20% per side: FaceNet's training crops included
                         # a bit of hair/chin/background, so ours should look the same
DB_DIR = Path("faces_db")             # enrolled people. Ignored by git: your photos stay here.
STRANGERS_CACHE = DB_DIR / "strangers.pt"  # embeddings of WIDER FACE faces, for calibration
DEFAULT_THRESHOLD = 0.55  # fallback if WIDER FACE isn't available to calibrate with


def load_embedder(device):
    """Load FaceNet with its VGGFace2 weights (downloads ~107 MB the first time)."""
    return InceptionResnetV1(pretrained="vggface2").eval().to(device)


def crop_face(image_rgb, box, margin=MARGIN):
    """Cut the face out of an image and resize it to 160x160.

    image_rgb: numpy array (H, W, 3), uint8, RGB order.
    box: (x1, y1, x2, y2) in FRACTIONS of width/height (0..1), as our locator outputs.
    Returns a (160, 160, 3) uint8 array, or None if the box is degenerate.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:                 # tiny box: nothing recognizable in it
        return None
    # Grow the box by the margin, but don't go past the image edges.
    x1, x2 = max(0, x1 - margin * bw), min(w, x2 + margin * bw)
    y1, y2 = max(0, y1 - margin * bh), min(h, y2 + margin * bh)
    crop = Image.fromarray(image_rgb).crop((int(x1), int(y1), int(x2), int(y2)))
    return np.asarray(crop.resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR))


@torch.no_grad()
def embed(model, crops, device, batch_size=64):
    """Face crops (list of 160x160x3 uint8 arrays) -> embeddings [N, 512], each of length 1."""
    out = []
    for i in range(0, len(crops), batch_size):
        x = torch.from_numpy(np.stack(crops[i:i + batch_size])).permute(0, 3, 1, 2).float().to(device)
        # FaceNet's own normalization: pixels 0..255 -> roughly -1..1.
        # (Different from ImageNet's mean/std: every network expects what it was trained on.)
        x = (x - 127.5) / 128.0
        out.append(model(x).cpu())
    # FaceNet already scales embeddings to length 1, which makes the dot product of two
    # embeddings equal to their cosine similarity. Normalize again to be safe.
    return torch.nn.functional.normalize(torch.cat(out), dim=1)


# ---------------------------------------------------------------------------
# Calibration: where to put the "you vs. not you" cut
# ---------------------------------------------------------------------------
def stranger_embeddings(model, device, n=500):
    """Embeddings of n different people's faces from WIDER FACE (cached after the first run).

    These are our "not you" examples. WIDER FACE has thousands of different people,
    so a random sample of single-face photos is very unlikely to contain you.
    """
    if STRANGERS_CACHE.exists():
        return torch.load(STRANGERS_CACHE)
    from object_locator import wider_jobs   # reuse the WIDER FACE label reader
    try:
        jobs = wider_jobs("val")
    except FileNotFoundError:
        return None                          # dataset not downloaded: caller uses the fallback
    random.Random(0).shuffle(jobs)
    print(f"Embedding {n} strangers' faces from WIDER FACE for calibration (first time only)...")
    crops = []
    for path, x1, y1, x2, y2, _ in jobs:
        if len(crops) == n:
            break
        if not path.exists():
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        crop = crop_face(img, (x1 / w, y1 / h, x2 / w, y2 / h))
        if crop is not None:
            crops.append(crop)
    emb = embed(model, crops, device)
    DB_DIR.mkdir(exist_ok=True)
    torch.save(emb, STRANGERS_CACHE)
    return emb


def calibrate_threshold(my_emb, strangers_emb, name, chart_path):
    """Pick the similarity cut between "you" and "strangers". Returns (threshold, mean_embedding).

    - Your scores: each of your photos vs. the average of your OTHER photos ("leave one
      out"). Comparing a photo to an average that includes itself would be cheating.
    - Stranger scores: each stranger vs. your full average.
    A good threshold sits in the gap between the two groups.
    """
    mean = torch.nn.functional.normalize(my_emb.mean(0), dim=0)
    n = len(my_emb)
    loo_means = torch.nn.functional.normalize((my_emb.sum(0, keepdim=True) - my_emb) / (n - 1), dim=1)
    my_scores = (my_emb * loo_means).sum(1)            # row-wise dot products

    if strangers_emb is None:
        print(f"WIDER FACE not found; using default threshold {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD, mean
    stranger_scores = strangers_emb @ mean             # one matrix-vector product scores them all

    # Use quantiles, not min/max, so one odd photo can't drag the threshold around:
    # the strangers' 99th percentile and your 5th percentile.
    stranger_hi = torch.quantile(stranger_scores, 0.99).item()
    me_lo = torch.quantile(my_scores, 0.05).item()
    if me_lo > stranger_hi:
        # Put the cut 30% of the way from the strangers to you, not in the middle.
        # Why lean toward the strangers? Your enrollment photos were all taken in one
        # session (same lighting, same day), so they agree with each other more than
        # future photos of you will. Testing on celebrity photos showed exactly this:
        # enrollment scores >= 0.86, but new photos of the same person down to 0.67.
        # Strangers, meanwhile, came from thousands of different photos, so their
        # scores are already realistic.
        threshold = stranger_hi + 0.3 * (me_lo - stranger_hi)
    else:
        # The groups overlap. Prefer rejecting you sometimes over accepting a stranger.
        threshold = stranger_hi
        print("WARNING: your photos overlap with strangers'. Try enrolling again with "
              "better lighting and your face filling more of the frame.")

    print(f"Similarity to {name}'s average:  your photos {my_scores.min():.2f}..{my_scores.max():.2f}"
          f"   strangers {stranger_scores.min():.2f}..{stranger_scores.max():.2f}")
    print(f"Threshold set to {threshold:.2f}  "
          f"(strangers above it: {(stranger_scores >= threshold).float().mean() * 100:.1f}%, "
          f"your photos below it: {(my_scores < threshold).float().mean() * 100:.1f}%)")
    save_calibration_chart(my_scores, stranger_scores, threshold, name, chart_path)
    return threshold, mean


def save_calibration_chart(my_scores, stranger_scores, threshold, name, path):
    """Histogram of both groups' similarity scores, with the threshold drawn in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(-0.3, 1, 53)
    ax.hist(stranger_scores.numpy(), bins=bins, color="#8a8984", label=f"Strangers (WIDER FACE, n={len(stranger_scores)})")
    ax.hist(my_scores.numpy(), bins=bins, color="#2a78d6", label=f"{name} (enrollment photos + their mirror images, n={len(my_scores)})")
    ax.axvline(threshold, color="#0b0b0b", linestyle="--", linewidth=1.5)
    ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.92, f"threshold {threshold:.2f}", color="#0b0b0b")
    ax.set_xlabel(f"Cosine similarity to {name}'s average face embedding (1 = identical direction)", color="#52514e")
    ax.set_ylabel("Number of face photos", color="#52514e")
    ax.set_title(f"Can FaceNet tell {name} apart from strangers?", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Enrolled people: save, load, match
# ---------------------------------------------------------------------------
def save_profile(name, crops, model, device):
    """Turn enrollment crops into a saved profile: average embedding + calibrated threshold."""
    person_dir = DB_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    for old in person_dir.glob("*.jpg"):          # re-enrolling replaces the old photos
        old.unlink()
    for i, crop in enumerate(crops):              # keep the photos so you can see what it learned
        Image.fromarray(crop).save(person_dir / f"{i:03d}.jpg")

    # Embed each photo AND its mirror image. The webcam view can be mirrored or not
    # (press m), and faces aren't perfectly symmetric, so we learn both versions.
    my_emb = embed(model, crops + [np.ascontiguousarray(c[:, ::-1]) for c in crops], device)
    strangers = stranger_embeddings(model, device)
    threshold, mean = calibrate_threshold(my_emb, strangers, name,
                                          Path("results") / "face_id" / f"{name}_similarity.png")
    torch.save({"name": name, "embedding": mean, "threshold": threshold}, person_dir / "profile.pt")
    print(f"Enrolled {name} from {len(crops)} photos -> {person_dir / 'profile.pt'}")


def load_profiles():
    """All enrolled people as a list of {"name", "embedding", "threshold"}."""
    return [torch.load(p) for p in sorted(DB_DIR.glob("*/profile.pt"))]


def identify(embedding, profiles):
    """Compare one face embedding to every enrolled person.

    Returns (name, score) for the best match if it clears that person's threshold,
    else (None, best score) meaning "Unknown".
    """
    if not profiles:
        return None, 0.0
    scores = [float(embedding @ p["embedding"]) for p in profiles]   # one dot product each
    best = int(np.argmax(scores))
    if scores[best] >= profiles[best]["threshold"]:
        return profiles[best]["name"], scores[best]
    return None, scores[best]
