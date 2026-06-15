"""
Dataset Preparation Utility
============================
Organises downloaded public datasets into the layout expected by train.py:

    data/
      train/  real/  fake/
      val/    real/  fake/
      test/   real/  fake/

Supported dataset helpers
--------------------------
  1. CIFAKE           (Kaggle — AI vs real, CIFAR-10 based)
  2. FaceForensics++  (frame extraction from videos)
  3. Celeb-DF v2      (frame extraction from videos)
  4. ArtiFact         (multi-generator AI images)
  5. GenImage         (Midjourney / SD / DALL-E images)
"""

import os
import shutil
import random
import cv2

from pathlib import Path
from typing import Literal

import kagglehub


# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR    = Path("/home/joan/.caatasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3")   # where you downloaded the datasets
OUTPUT_DIR = Path("/home/joan/code/ai_im_detect_train")           # destination consumed by train.py
SPLITS     = {"train": 0.75, "val": 0.15, "test": 0.10}
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED       = 42
random.seed(SEED)


def make_split_dirs():
    for split in SPLITS:
        for label in ["real", "fake"]:
            (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)


def split_and_copy(files: list[Path], label: Literal["real", "fake"]):
    """Shuffle files and copy them into train / val / test with prefix."""
    random.shuffle(files)
    n = len(files)
    boundaries = {
        "train": (0,                      int(n * SPLITS["train"])),
        "val":   (int(n * SPLITS["train"]),
                  int(n * (SPLITS["train"] + SPLITS["val"]))),
        "test":  (int(n * (SPLITS["train"] + SPLITS["val"])), n),
    }
    for split, (start, end) in boundaries.items():
        for src in files[start:end]:
            dst = OUTPUT_DIR / split / label / src.name
            # Avoid name collisions by prefixing dataset name
            if dst.exists():
                dst = dst.with_stem(src.stem + f"_{random.randint(0, 99999):05d}")
            shutil.copy2(src, dst)
    print(f"  [{label}] {n} files → train/val/test "
          f"({int(n*SPLITS['train'])}/{int(n*SPLITS['val'])}/{n - int(n*SPLITS['train']) - int(n*SPLITS['val'])})")


# ── Dataset Helpers ───────────────────────────────────────────────────────────

# no faces!!!!!
# def prepare_cifake():
#     """
#     CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
#     Download and unzip; expected structure:
#         raw_datasets/CIFAKE/
#           train/  REAL/  FAKE/
#           test/   REAL/  FAKE/
#     """
#     print("\n[CIFAKE]")
#     base = Path("/home/joan/.cache/kagglehub/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3")   # where you downloaded the datasets

#     for folder in ["train", "test"]:
#         real_imgs = list((base / folder / "REAL").glob("*"))
#         fake_imgs = list((base / folder / "FAKE").glob("*"))
#         real_imgs = [p for p in real_imgs if p.suffix.lower() in IMG_EXTS]
#         fake_imgs = [p for p in fake_imgs if p.suffix.lower() in IMG_EXTS]
#         # Note: CIFAKE already has a train/test split — we re-split uniformly
#         split_and_copy(real_imgs, "real")
#         split_and_copy(fake_imgs, "fake")


# def prepare_faceforensics(max_frames_per_video: int = 20):
#     """
#     FaceForensics++: https://github.com/ondyari/FaceForensics
#     After downloading, extract frames with the official script or ffmpeg.
#     Expected structure after frame extraction:
#         raw_datasets/FaceForensics/
#           original_sequences/youtube/raw/videos/   *.mp4   (real)
#           manipulated_sequences/Deepfakes/raw/videos/      (fake)
#           manipulated_sequences/Face2Face/raw/videos/      (fake)
#           manipulated_sequences/FaceSwap/raw/videos/       (fake)
#           manipulated_sequences/NeuralTextures/raw/videos/ (fake)
#     """
#     print("\n[FaceForensics++]")
#     try:
#         import cv2
#     except ImportError:
#         print("  ✗ opencv-python not installed. Run: pip install opencv-python")
#         return

#     base = RAW_DIR / "FaceForensics"

#     def extract_frames(video_path: Path, out_dir: Path, n: int = max_frames_per_video):
#         out_dir.mkdir(parents=True, exist_ok=True)
#         cap = cv2.VideoCapture(str(video_path))
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if total <= 0:
#             cap.release(); return []
#         indices = sorted(random.sample(range(total), min(n, total)))
#         saved = []
#         for i, idx in enumerate(indices):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#             if not ret: continue
#             p = out_dir / f"{video_path.stem}_f{idx:05d}.jpg"
#             cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
#             saved.append(p)
#         cap.release()
#         return saved

#     real_dir  = base / "original_sequences/youtube/raw/videos"
#     fake_dirs = [
#         base / "manipulated_sequences/Deepfakes/raw/videos",
#         base / "manipulated_sequences/Face2Face/raw/videos",
#         base / "manipulated_sequences/FaceSwap/raw/videos",
#         base / "manipulated_sequences/NeuralTextures/raw/videos",
#     ]

#     real_frames, fake_frames = [], []
#     tmp_real = RAW_DIR / "ff_frames" / "real"
#     tmp_fake = RAW_DIR / "ff_frames" / "fake"

#     for vp in sorted(real_dir.glob("*.mp4"))[:200]:
#         real_frames.extend(extract_frames(vp, tmp_real))

#     for fdir in fake_dirs:
#         for vp in sorted(fdir.glob("*.mp4"))[:50]:
#             fake_frames.extend(extract_frames(vp, tmp_fake))

#     split_and_copy(real_frames, "real")
#     split_and_copy(fake_frames, "fake")


def prepare_artifact(max_per_class: int = 50_000):
    """
    ArtiFact — after manual cleanup:
      - non-face folders deleted
      - real image folders renamed to real_* prefix
      - fake image folders have no consistent prefix (but are not real_*)
      - internal folder structure is inconsistent across folders,
        so we rglob for all images regardless of nesting
    """
    print("\n[ArtiFact]")
    base_path = "/home/joan/.cache/kagglehub/datasets/awsaf49/artifact-dataset/versions/1"

    base = Path(base_path)

    real_imgs: list[Path] = []
    fake_imgs: list[Path] = []

    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue

        is_real = folder.name.lower().startswith("real_")

        imgs = [
            p for p in folder.rglob("*")
            if p.suffix.lower() in IMG_EXTS and p.is_file()
        ]

        if not imgs:
            print(f"  [skip]  {folder.name} — no images found")
            continue

        label = "real" if is_real else "fake"
        print(f"  [{label}]  {folder.name:<35} {len(imgs):>8,} images")

        if is_real:
            real_imgs.extend(imgs)
        else:
            fake_imgs.extend(imgs)

    print(f"\n  Before cap — real: {len(real_imgs):,}  fake: {len(fake_imgs):,}")

    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)
    real_imgs = real_imgs[:max_per_class]
    fake_imgs = fake_imgs[:max_per_class]

    print(f"  After cap  — real: {len(real_imgs):,}  fake: {len(fake_imgs):,}")

    split_and_copy(real_imgs, "real")
    split_and_copy(fake_imgs, "fake")


# def prepare_genimage(max_per_generator: int = 5000):
#     """
#     GenImage: https://github.com/GenImage-Dataset/GenImage
#     Expected structure:
#         raw_datasets/GenImage/
#           imagenet_ai_<generator>/  train/  ai/   *.jpeg
#                                             nature/ *.jpeg
#     """
#     print("\n[GenImage]")
#     base = RAW_DIR / "GenImage"
#     real_imgs, fake_imgs = [], []
#     for gen_dir in sorted(base.glob("imagenet_ai_*")):
#         for split_dir in gen_dir.glob("*"):
#             ai = list((split_dir / "ai").glob("*") if (split_dir / "ai").exists() else [])
#             na = list((split_dir / "nature").glob("*") if (split_dir / "nature").exists() else [])
#             ai = [p for p in ai if p.suffix.lower() in IMG_EXTS]
#             na = [p for p in na if p.suffix.lower() in IMG_EXTS]
#             fake_imgs.extend(random.sample(ai, min(max_per_generator, len(ai))))
#             real_imgs.extend(random.sample(na, min(max_per_generator, len(na))))
#     split_and_copy(real_imgs, "real")
#     split_and_copy(fake_imgs, "fake")


def print_summary():
    print("\n── Dataset Summary ──────────────────────────────────────")
    total = 0
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            folder = OUTPUT_DIR / split / label
            n = len(list(folder.glob("*")))
            total += n
            print(f"  {split:5s}/{label:4s}: {n:7,} images")
    print(f"  {'TOTAL':10s}: {total:7,} images")


if __name__ == "__main__":

    # download artifact
    #path = kagglehub.dataset_download("awsaf49/artifact-dataset")
    path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")


    print("Path to dataset files:", path)

    # make_split_dirs()
    # # Uncomment the datasets you have downloaded:
    # prepare_cifake()
    # # prepare_faceforensics()

    # artifaceP = "/home/joan/.cache/kagglehub/datasets/awsaf49/artifact-dataset/versions/1"
    # audit_artifact(artifaceP)

    # prepare_artifact()
    # # # prepare_genimage()
    # print_summary()

