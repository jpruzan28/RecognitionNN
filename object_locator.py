"""
object_locator.py  --  Find ONE object in an image and draw a box around it.

This is the step after MiniNN. MiniNN answered "WHAT digit is this?" (classification).
This script answers "WHERE is the thing?" (localization): the network outputs 4 numbers
that describe a bounding box, instead of 10 class scores.

Two tasks, same network, chosen with --dataset:
  faces (default): find the human FACE in photos from WIDER FACE, a standard face
                   detection dataset. We keep only its photos that contain exactly one
                   face, which matches the goal: find MY face on my webcam and box it.
  pets:            find the HEAD of the cat or dog in Oxford-IIIT Pet photos. This was
                   the first version, a stand-in for faces while the pipeline was built.

The network: a ResNet-18 that was already trained on ImageNet (1.2M photos, 1000 classes).
We keep its "eyes" (the convolutional layers that learned edges, fur, eyes, noses, ...)
and replace only its last layer, so it outputs a box instead of 1000 class scores.
This is called TRANSFER LEARNING: a few thousand labeled images and about a minute of
GPU time are enough, where training from scratch would need far more of both.

Pipeline:
  1. Download the dataset (first run only: ~1.8 GB faces, ~800 MB pets).
  2. Read every image + its box, shrink images to 224x224, cache them to disk.
  3. Load the pretrained ResNet-18, swap its final layer for a 4-number box head.
  4. Train (fine-tune) for a few epochs on the GPU.
  5. Measure accuracy with IoU (Intersection over Union, explained below).
  6. Save charts + pictures of every test prediction -> results/<dataset>/
  7. Optionally: run the trained model on your own photo with --image path/to/photo.jpg
  8. Optionally: test it live on your webcam with --webcam

Usage:
  python object_locator.py                        # train on faces + evaluate + save results
  python object_locator.py --dataset pets         # the pet-head version instead
  python object_locator.py --image me.jpg         # also box something in your own image
  python object_locator.py --epochs 15            # train longer
  python object_locator.py --webcam               # train, then test live on your webcam
                                                  # (webcam code lives in webcam_demo.py)
"""

import argparse                         # read command-line options like --epochs
import time                             # measure how long each stage takes
import zipfile                          # unpack the downloaded dataset
import random                           # shuffle the dataset before splitting it
import xml.etree.ElementTree as ET      # the box labels are stored in small XML files
from pathlib import Path                # file paths that work on Windows and Linux alike
from concurrent.futures import ThreadPoolExecutor  # decode many JPEGs at the same time

import numpy as np                      # PIL image -> array conversion
import torch                            # core tensor library
import torch.nn as nn                   # neural-network layers
import torchvision                      # datasets + pretrained models
from torchvision.ops import generalized_box_iou_loss  # a loss designed for boxes
from PIL import Image, ImageDraw, ImageFont  # open JPEGs; draw boxes/text for the gallery
import matplotlib
matplotlib.use("Agg")                   # draw to files, not to a window (works everywhere)
import matplotlib.pyplot as plt
import matplotlib.patches as patches    # for drawing rectangles on images


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
IMG_SIZE = 224          # ResNet was pretrained on 224x224 images, so we use the same size
# Where the dataset and our cache live. Deliberately OUTSIDE the project folder:
# the project sits in OneDrive, which would try to upload ~1.3 GB of images and cache.
DATA_DIR = Path.home() / "datasets"
# Both datasets come from Hugging Face mirrors: the original servers can be ~100x slower.
PETS_URL = ("https://huggingface.co/datasets/antokun/The-Oxford-IIIT-Pet-Dataset-With-Annotations"
            "/resolve/main/The%20Oxford-IIIT%20Pet%20Dataset%20With%20Annotations.zip")
WIDER_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/"
MIN_FACE_FRACTION = 0.05  # skip faces narrower than 5% of the photo (a few pixels after resizing)
SEED = 0             # fixed random seed -> same train/test split every run

# ImageNet statistics. The pretrained network saw images normalized with these
# exact numbers during its original training, so we must normalize the same way,
# or its learned features would receive inputs on a scale it has never seen.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# 1 + 2. Data: download, read boxes, resize, cache
# ---------------------------------------------------------------------------
def load_image(image_path, x1, y1, x2, y2, min_width_fraction=0.0):
    """Open one photo, given its box in PIXELS. Return (image tensor, box) or None to skip it."""
    if not image_path.exists():          # a handful of labels point at missing files
        return None
    img = Image.open(image_path).convert("RGB")  # some images are grayscale/RGBA -> force 3 channels
    w, h = img.size
    if (x2 - x1) < min_width_fraction * w:       # object too small to be useful
        return None

    # Store the box in NORMALIZED coordinates: fractions of image width/height (0..1).
    # Why? We squash every image to 224x224 (changing its aspect ratio). A box in
    # fractions stays correct under that squash, whereas pixel coordinates would not.
    # It also keeps the numbers the network must predict in a small, fixed range.
    box = torch.tensor([x1 / w, y1 / h, x2 / w, y2 / h]).clamp(0, 1)

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    # PIL image (H, W, C) -> tensor (C, H, W), kept as uint8 (0..255) to save memory.
    # 4000 images x 3 x 224 x 224 bytes = ~600 MB. As float32 it would be 4x bigger.
    img = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
    return img, box


def load_many(jobs):
    """Run load_image on a list of argument tuples, several at once. Returns (images, boxes)."""
    # Threads let several JPEGs decode at once (PIL releases Python's GIL while decoding).
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [r for r in pool.map(lambda job: load_image(*job), jobs) if r is not None]
    return torch.stack([r[0] for r in results]), torch.stack([r[1] for r in results])


def download_zip(url, dest_dir, name):
    """Download a zip into dest_dir, extract it there, then delete the zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / name
    print(f"Downloading {name} (first run only)...")
    torch.hub.download_url_to_file(url, str(zip_path))
    print("Extracting...")
    zipfile.ZipFile(zip_path).extractall(dest_dir)
    zip_path.unlink()   # the extracted files are all we need


def pets_jobs():
    """Oxford-IIIT Pet: ~3,700 photos, each with a box around the animal's HEAD."""
    pet_dir = DATA_DIR / "oxford-iiit-pet"
    if not (pet_dir / "annotations" / "xmls").exists():
        download_zip(PETS_URL, pet_dir, "pets.zip")

    # Each label is a small XML file, roughly:
    #   <annotation><filename>Abyssinian_1.jpg</filename>
    #     <object><bndbox><xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax></bndbox></object>
    #   </annotation>
    jobs = []
    for xml_path in sorted((pet_dir / "annotations" / "xmls").glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        bnd = root.find("object/bndbox")     # take the first (and usually only) head box
        coords = (float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        jobs.append((pet_dir / "images" / root.findtext("filename"), *coords))

    # This dataset has no official split for its head boxes, so we make one:
    # shuffle, then 80% train / 20% test.
    random.Random(SEED).shuffle(jobs)
    n_train = int(0.8 * len(jobs))
    return jobs[:n_train], jobs[n_train:]


def wider_jobs(split):
    """WIDER FACE: photos of people in every kind of scene, with a box on every face.

    The label file lists, for each photo: its path, the number of faces, then one line
    per face: "x y width height" plus 6 flags (blur, expression, lighting, invalid,
    occlusion, pose). Example:
        0--Parade/0_Parade_Parade_0_904.jpg
        1
        361 98 263 339 0 0 0 0 0 0
    """
    wider_dir = DATA_DIR / "wider_face"
    lines = (wider_dir / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt").read_text().split("\n")
    jobs, i = [], 0
    while i < len(lines) and lines[i].strip():
        name, n_faces = lines[i].strip(), int(lines[i + 1])
        rows = lines[i + 2: i + 2 + max(n_faces, 1)]   # photos with 0 faces still have 1 filler line
        i += 2 + max(n_faces, 1)
        # Keep ONLY photos with exactly one face. Our network outputs exactly one box,
        # so a photo with 3 faces would have 3 "right answers" and confuse it.
        # One face per photo also matches the goal: your face in front of your webcam.
        if n_faces != 1:
            continue
        x, y, w, h, *flags = map(int, rows[0].split())
        if flags[3] or w <= 0 or h <= 0:               # flags[3] = "invalid" label
            continue
        jobs.append((wider_dir / f"WIDER_{split}" / "images" / name,
                     x, y, x + w, y + h, MIN_FACE_FRACTION))
    return jobs


def faces_jobs():
    """WIDER FACE single-face photos. Its official train/val split is used as train/test."""
    wider_dir = DATA_DIR / "wider_face"
    if not (wider_dir / "wider_face_split").exists():
        download_zip(WIDER_URL + "wider_face_split.zip", wider_dir, "wider_face_split.zip")  # labels
    for split in ("train", "val"):
        if not (wider_dir / f"WIDER_{split}").exists():   # photos: ~1.5 GB + ~360 MB
            download_zip(WIDER_URL + f"WIDER_{split}.zip", wider_dir, f"WIDER_{split}.zip")
    return wider_jobs("train"), wider_jobs("val")


# Everything that differs between the two tasks. The model and the training are
# identical; only the photos and what the box means change.
DATASETS = {
    "faces": {"jobs": faces_jobs, "thing": "face", "title": "Face locator"},
    "pets": {"jobs": pets_jobs, "thing": "head", "title": "Pet-head locator"},
}


def load_dataset(name):
    """Return (train_x, train_y, test_x, test_y): uint8 images [N,3,224,224] + boxes [N,4]."""
    cache_file = DATA_DIR / f"{name}_{IMG_SIZE}.pt"
    if cache_file.exists():
        # Decoding thousands of JPEGs takes a while; after the first run we load the
        # already-resized tensors straight from disk in about a second.
        print(f"Loading cached dataset from {cache_file}")
        data = torch.load(cache_file)
        return data["train_x"], data["train_y"], data["test_x"], data["test_y"]

    train_jobs, test_jobs = DATASETS[name]["jobs"]()
    print(f"Decoding + resizing {len(train_jobs) + len(test_jobs)} labeled photos...")
    train_x, train_y = load_many(train_jobs)
    test_x, test_y = load_many(test_jobs)
    torch.save({"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}, cache_file)
    return train_x, train_y, test_x, test_y


# ---------------------------------------------------------------------------
# 3. Model: pretrained ResNet-18 with a box-predicting head
# ---------------------------------------------------------------------------
class BoxRegressor(nn.Module):
    """ResNet-18 backbone + a tiny head that outputs one box (x1, y1, x2, y2) in 0..1."""

    def __init__(self):
        super().__init__()
        # Download (first time only) the ImageNet-trained weights.
        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        # ResNet-18 ends with: conv layers -> global average pool -> 512 numbers ->
        # Linear(512, 1000) giving ImageNet class scores. We replace that final Linear.
        # nn.Identity() means "pass through unchanged", so the backbone now outputs
        # the 512-number feature vector: a compact description of what's in the image.
        num_features = self.backbone.fc.in_features   # 512
        self.backbone.fc = nn.Identity()

        # The new head. Same kind of layers as MiniNN (Linear -> ReLU -> Linear),
        # but ending in 4 numbers instead of 10.
        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),                  # the nonlinearity again: without it, the two
                                        # Linears would collapse into one linear map
            nn.Linear(256, 4),
        )

    def forward(self, x):
        features = self.backbone(x)             # [B, 512]
        raw = self.head(features)               # [B, 4], any real numbers

        # Turn 4 unconstrained numbers into a VALID box.
        # We predict center (cx, cy) and size (w, h), each squeezed into 0..1 by a
        # sigmoid. Then convert to corners. Doing it this way guarantees x2 >= x1
        # and y2 >= y1. If we predicted the corners directly, an untrained network
        # could output a box whose right edge sits left of its left edge.
        cx, cy, w, h = torch.sigmoid(raw).unbind(dim=1)
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        return torch.stack([x1, y1, x2, y2], dim=1)

    # NOTE for the optical-computing goal:
    # Convolution layers are LINEAR, just like nn.Linear. A convolution is exactly
    # what a lens-based "4f" optical system computes (multiply in the Fourier plane),
    # so the bulk of ResNet's work is optics-friendly. The hard parts are the same as
    # in MiniNN: ReLU (nonlinear), plus BatchNorm and max-pooling, which also are not
    # plain matrix multiplications.


# ---------------------------------------------------------------------------
# Helpers: IoU, preprocessing, augmentation
# ---------------------------------------------------------------------------
def box_iou(pred, target):
    """Intersection over Union for matching pairs of boxes. Shape [B,4] -> [B].

    IoU = (area where the boxes overlap) / (area covered by either box).
    1.0 = perfect match, 0.0 = no overlap. The usual rule of thumb is that
    IoU >= 0.5 counts as a "correct" detection, so that is our accuracy metric.
    """
    ix1 = torch.max(pred[:, 0], target[:, 0])   # overlap rectangle = the inner edges
    iy1 = torch.max(pred[:, 1], target[:, 1])
    ix2 = torch.min(pred[:, 2], target[:, 2])
    iy2 = torch.min(pred[:, 3], target[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)  # 0 if they don't overlap
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    return inter / (area_p + area_t - inter + 1e-7)  # tiny epsilon avoids divide-by-zero


def prepare(images_uint8, device):
    """uint8 [B,3,H,W] (0..255) -> normalized float on the GPU, ready for ResNet."""
    x = images_uint8.to(device, non_blocking=True).float() / 255.0   # like ToTensor() in MiniNN
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def augment(x, boxes):
    """Randomly mirror half the batch left<->right, and move the boxes to match.

    Data augmentation = showing the network slightly altered copies of the training
    images, so it learns "a head is a head" rather than memorizing exact pictures.
    A mirrored cat is still a cat, but its box moves: new x1 = 1 - old x2, and so on.
    """
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(dims=[3]), x)  # dim 3 = width
    flipped = torch.stack([1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], dim=1)
    boxes = torch.where(flip.view(-1, 1), flipped, boxes)
    return x, boxes


@torch.no_grad()    # no gradients needed when only measuring -> faster, less memory
def evaluate(model, images, boxes, device, batch_size=256):
    """Return (predicted boxes, IoU per image) for a whole set of images."""
    model.eval()    # switches BatchNorm to use its stored statistics (see train loop)
    preds = []
    for i in range(0, len(images), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            preds.append(model(prepare(images[i:i + batch_size], device)).float())
    preds = torch.cat(preds)
    return preds, box_iou(preds, boxes.to(device))


# ---------------------------------------------------------------------------
# 6 + 7. Drawing results
# ---------------------------------------------------------------------------
def draw_box(ax, box, size, color, style="-"):
    """Draw a normalized (0..1) box on a matplotlib axis showing an image of `size` = (w, h)."""
    w, h = size
    x1, y1, x2, y2 = box.tolist()
    ax.add_patch(patches.Rectangle((x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
                                   fill=False, edgecolor=color, linewidth=2.5, linestyle=style))


def save_samples(images, true_boxes, pred_boxes, ious, path, thing, n=8):
    """Plot n test images: dashed white = true box, solid green/red = prediction."""
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 6.5))
    for ax, img, t, p, iou in zip(axes.flat, images[:n], true_boxes[:n], pred_boxes[:n], ious[:n]):
        ax.imshow(img.permute(1, 2, 0).numpy())  # (C,H,W) -> (H,W,C) for matplotlib
        good = iou >= 0.5
        draw_box(ax, t, (IMG_SIZE, IMG_SIZE), "white", "--")
        draw_box(ax, p, (IMG_SIZE, IMG_SIZE), "lime" if good else "red")
        ax.set_title(f"IoU = {iou:.2f}", color="green" if good else "red")
        ax.axis("off")
    fig.suptitle(f"Dashed white = true {thing} box   |   solid = network's prediction")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved {path}")


# Colors for the stats charts: one blue for "the model", status green/red for hit/miss.
BLUE, GRAY, INK, MUTED = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"
HIT_GREEN, MISS_RED = "#0ca30c", "#d03b3b"


def style_axis(ax, title, ylabel):
    """Quiet axes so the data stands out: no top/right frame, faint horizontal grid."""
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e9e8e4", linewidth=0.8)
    ax.set_axisbelow(True)


def save_stats(history, baseline_iou, final_ious, train_seconds, path, title, n_train):
    """One figure with 4 small charts: how training went, and how good the final model is.

    Each chart has its own y-axis on purpose: loss, IoU and % are different units,
    and squeezing two of them onto one chart with two y-scales is easy to misread.
    """
    epochs = range(1, len(history["loss"]) + 1)
    ious = final_ious.numpy()
    accuracy = (ious >= 0.5).mean() * 100

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    (ax_loss, ax_iou), (ax_acc, ax_hist) = axes

    def line(ax, values, fmt):
        ax.plot(epochs, values, color=BLUE, linewidth=2, marker="o", markersize=6)
        # Label only the final value; a number on every point would be clutter.
        ax.annotate(fmt.format(values[-1]), (len(values), values[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=10)
        ax.set_xlabel(f"Epoch (one full pass over the {n_train:,} training images)", color=MUTED)
        ax.set_xticks(list(epochs))
        ax.set_xlim(0.5, len(values) + 1.2)

    # 1. Loss: the number training directly minimizes. Should fall steadily.
    line(ax_loss, history["loss"], "{:.3f}")
    style_axis(ax_loss, "Training loss (lower is better)", "L1 + GIoU loss")

    # 2. Mean IoU on test images the model never trained on, vs the "dumb guess" baseline.
    line(ax_iou, history["iou"], "{:.2f}")
    ax_iou.axhline(baseline_iou, color=GRAY, linestyle="--", linewidth=1.5)
    ax_iou.text(1, baseline_iou + 0.015, f"Baseline: always guess the average box ({baseline_iou:.2f})",
                color=MUTED, fontsize=9)
    ax_iou.set_ylim(0, 1)
    style_axis(ax_iou, "Test mean IoU (box overlap, 1 = perfect)", "Mean IoU")

    # 3. Accuracy: % of test images where the box overlaps the true one by IoU >= 0.5.
    line(ax_acc, history["acc"], "{:.1f}%")
    ax_acc.set_ylim(min(history["acc"]) - 5, 100)
    style_axis(ax_acc, "Test accuracy (% of images with IoU ≥ 0.5)", "Accuracy (%)")

    # 4. Histogram: the spread of IoU over every test image after training.
    #    The mean hides the failures; this shows how many there are and how bad.
    bins = np.linspace(0, 1, 21)
    counts, edges = np.histogram(ious, bins=bins)
    colors = [MISS_RED if left < 0.5 else HIT_GREEN for left in edges[:-1]]
    ax_hist.bar(edges[:-1], counts, width=0.05 - 0.004, align="edge", color=colors)
    ax_hist.axvline(0.5, color=INK, linewidth=1, linestyle=":")
    n_miss = int((ious < 0.5).sum())
    ax_hist.text(0.48, counts.max() * 0.95, f"✗ {n_miss} misses", ha="right", color=MISS_RED, fontsize=10)
    ax_hist.text(0.52, counts.max() * 0.95, f"✓ {len(ious) - n_miss} hits", ha="left", color=HIT_GREEN, fontsize=10)
    ax_hist.set_xlabel("IoU of each test image (0.5 = hit/miss threshold)", color=MUTED)
    ax_hist.set_xlim(0, 1)
    style_axis(ax_hist, f"Final IoU per test image ({len(ious)} images)", "Number of images")

    fig.suptitle(f"{title} (ResNet-18):  {accuracy:.1f}% accuracy,  "
                 f"mean IoU {ious.mean():.2f},  trained in {train_seconds:.0f}s",
                 fontsize=14, color=INK, x=0.02, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


def save_gallery(images, true_boxes, pred_boxes, ious, out_dir, thing, cols=8, rows=6):
    """Save EVERY test image with its boxes, as pages of cols x rows tiles, worst first.

    Uses PIL directly instead of matplotlib: drawing ~1000 images with matplotlib
    subplots would take minutes, while PIL just paints pixels and takes seconds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("gallery_*.png"):   # remove pages from a previous run
        old.unlink()

    font = ImageFont.load_default(size=14)
    label_h, header_h, pad = 22, 34, 4
    tile_w, tile_h = IMG_SIZE + pad, IMG_SIZE + label_h + pad
    per_page = cols * rows
    order = torch.argsort(ious)                 # lowest IoU first: the failures are the interesting part
    n_pages = (len(order) + per_page - 1) // per_page

    def to_px(box):                             # normalized (0..1) box -> pixel corners
        return [round(v * (IMG_SIZE - 1)) for v in box.tolist()]

    for page in range(n_pages):
        idxs = order[page * per_page:(page + 1) * per_page]
        sheet = Image.new("RGB", (cols * tile_w + pad, header_h + rows * tile_h + pad), "#1a1a19")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 9), f"Page {page + 1}/{n_pages}  -  sorted worst to best IoU  -  "
                           f"white = true {thing} box,  green = hit (IoU >= 0.5),  red = miss",
                  fill="white", font=font)

        for slot, i in enumerate(idxs.tolist()):
            rank = page * per_page + slot + 1
            x0 = pad + (slot % cols) * tile_w
            y0 = header_h + pad + (slot // cols) * tile_h
            tile = Image.fromarray(images[i].permute(1, 2, 0).numpy())
            t = ImageDraw.Draw(tile)
            hit = ious[i] >= 0.5
            color = HIT_GREEN if hit else MISS_RED
            t.rectangle(to_px(true_boxes[i]), outline="white", width=2)
            t.rectangle(to_px(pred_boxes[i]), outline=color, width=3)
            sheet.paste(tile, (x0, y0))
            draw.text((x0 + 2, y0 + IMG_SIZE + 3),
                      f"#{rank}  IoU {ious[i]:.2f}  {'HIT' if hit else 'MISS'}",
                      fill=color if not hit else "#9fe39f", font=font)
        sheet.save(out_dir / f"gallery_{page + 1:02d}.png")
    print(f"Saved {len(order)} test images on {n_pages} pages in {out_dir}/")


@torch.no_grad()
def locate_in_file(model, image_path, device, out_path="my_image_prediction.png"):
    """Run the trained model on any image file and save it with the predicted box drawn."""
    model.eval()
    original = Image.open(image_path).convert("RGB")
    small = original.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(small).copy()).permute(2, 0, 1).unsqueeze(0)
    box = model(prepare(x, device)).float()[0].cpu()
    # Because the box is normalized (0..1), we can draw it straight onto the
    # ORIGINAL full-resolution image, not just the 224x224 copy.
    fig, ax = plt.subplots(figsize=(6, 6 * original.height / original.width))
    ax.imshow(original)
    draw_box(ax, box, original.size, "lime")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Predicted box (fractions of width/height): {[round(v, 3) for v in box.tolist()]}")
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main: put it all together
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 to box one face (or pet head) per image.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="faces",
                        help="faces = WIDER FACE human faces (default), pets = Oxford-IIIT Pet heads")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image", type=str, default=None, help="optional: your own image to run the model on")
    parser.add_argument("--webcam", action="store_true", help="after training, test the model live on your webcam")
    parser.add_argument("--camera", type=int, default=0, help="webcam index for --webcam (0 = default camera)")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    DATA_DIR.mkdir(exist_ok=True)

    # ---- Data ----
    t0 = time.time()
    train_x, train_y, test_x, test_y = load_dataset(args.dataset)
    # The test images are never used for training, so the test score shows how
    # well the model handles NEW photos it has never seen.
    print(f"{len(train_x)} training / {len(test_x)} test images  ({time.time() - t0:.1f}s)")

    # The whole training set fits in GPU memory as uint8 (~450 MB), so we move it
    # there once. Every batch is then sliced directly on the GPU, with no per-batch
    # copying from the CPU. That is the main reason training is so fast.
    if device.type == "cuda":
        train_x, train_y = train_x.to(device), train_y.to(device)

    # ---- Model, loss, optimizer ----
    model = BoxRegressor().to(device)

    # Before training: how good is a "dumb" guess? Predict the average training box for
    # every image. Any real learning must beat this number.
    mean_box = train_y.mean(dim=0, keepdim=True).cpu()
    baseline_iou = box_iou(mean_box.expand(len(test_y), 4), test_y).mean()
    print(f"Baseline (always guess the average box): mean IoU = {baseline_iou:.3f}")

    # Adam, as in MiniNN, but with a smaller learning rate (3e-4 vs 0.01). The backbone
    # already holds useful knowledge; big steps would wreck it ("catastrophic forgetting").
    # AdamW = Adam + weight decay, a mild pull toward small weights that reduces overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (len(train_x) + args.batch_size - 1) // args.batch_size
    # Cosine schedule: start at lr, smoothly decrease to ~0 by the end. Big steps
    # early to learn fast, tiny steps late to settle into a good solution.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)

    # ---- 4. Training ----
    print(f"\nTraining for {args.epochs} epochs...")
    history = {"loss": [], "iou": [], "acc": []}   # one entry per epoch, for the stats charts
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()   # BatchNorm layers use the current batch's statistics while training
        perm = torch.randperm(len(train_x), device=train_x.device)  # new random order every epoch
        total_loss = 0.0
        for i in range(0, len(train_x), args.batch_size):
            idx = perm[i:i + args.batch_size]
            x = prepare(train_x[idx], device)
            y = train_y[idx].to(device)
            x, y = augment(x, y)

            # Mixed precision: run the heavy conv math in bfloat16 (16-bit) on the GPU.
            # Roughly 2x faster, and accurate enough for training. (Skipped on CPU.)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(x)
            pred = pred.float()  # compute the loss in full 32-bit precision

            # The loss combines two terms:
            #  - L1: average absolute error of the 4 coordinates. Simple, always has a gradient.
            #  - GIoU loss: 1 - (a smarter IoU). It directly rewards overlap, the thing we
            #    care about. "Generalized" IoU still gives a useful signal when the boxes
            #    don't overlap at all, where plain IoU would be stuck at 0.
            loss = nn.functional.l1_loss(pred, y) + generalized_box_iou_loss(pred, y, reduction="mean")

            optimizer.zero_grad()   # clear gradients from the previous batch
            loss.backward()         # backprop: how should each weight change?
            optimizer.step()        # apply the change
            scheduler.step()        # lower the learning rate a tiny bit
            total_loss += loss.item() * len(idx)

        _, test_iou = evaluate(model, test_x, test_y, device)
        history["loss"].append(total_loss / len(train_x))
        history["iou"].append(test_iou.mean().item())
        history["acc"].append((test_iou >= 0.5).float().mean().item() * 100)
        print(f"Epoch {epoch + 1:2d}/{args.epochs}  loss {history['loss'][-1]:.4f}  "
              f"test mean IoU {history['iou'][-1]:.3f}  "
              f"accuracy (IoU>=0.5) {history['acc'][-1]:.1f}%  "
              f"[{time.time() - t0:.0f}s]")
    train_seconds = time.time() - t0

    # ---- 5. Final evaluation ----
    preds, test_iou = evaluate(model, test_x, test_y, device)
    print(f"\nFinal test mean IoU: {test_iou.mean():.3f}   (baseline was {baseline_iou:.3f})")
    print(f"Final accuracy (IoU >= 0.5): {(test_iou >= 0.5).float().mean() * 100:.1f}%")

    weights_file = f"box_regressor_{args.dataset}.pt"
    torch.save(model.state_dict(), weights_file)   # keep the trained weights (webcam_demo.py loads them)
    print(f"Saved trained weights to {weights_file}")

    # ---- 6. Charts + pictures of the predictions (all saved in results/) ----
    info = DATASETS[args.dataset]
    results_dir = Path("results") / args.dataset     # results/faces/ or results/pets/
    results_dir.mkdir(parents=True, exist_ok=True)
    preds, test_iou = preds.cpu(), test_iou.cpu()
    save_stats(history, baseline_iou.item(), test_iou, train_seconds,
               results_dir / "training_stats.png", info["title"], len(train_x))
    save_samples(test_x, test_y, preds, test_iou, results_dir / "sample_predictions.png", info["thing"])
    save_gallery(test_x, test_y, preds, test_iou, results_dir / "gallery", info["thing"])

    # ---- 7. Optional: your own photo ----
    if args.image:
        locate_in_file(model, args.image, device)

    print(f"\nTotal run time: {time.time() - t_start:.0f}s")

    # ---- 8. Optional: live webcam test ----
    if args.webcam:
        # Imported here rather than at the top: webcam_demo itself imports this file
        # (for BoxRegressor and prepare), and it needs OpenCV, which training doesn't.
        from webcam_demo import run_webcam
        # Face recognition (who is it?) only makes sense for the face model.
        run_webcam(model=model, device=device, camera=args.camera, recognize=args.dataset == "faces")


# On Windows, code that may start extra processes must sit behind this guard.
# It is also good practice in general: importing this file won't start training.
if __name__ == "__main__":
    main()
