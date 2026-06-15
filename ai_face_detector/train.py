"""
Lightweight AI-Generated / Deepfake Image Detector
====================================================
Architecture : EfficientNet-B0 (fine-tuned)
Target       : Mobile on-device inference (TFLite / CoreML)
Task         : Binary classification — Real vs Fake
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
# pip install torch torchvision timm torchaudio
# pip install coremltools onnx onnxruntime
# (TFLite export requires tensorflow separately)

import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

import timm  # pip install timm  — provides EfficientNet-B0 & MobileNetV3


# ── 1. Config ─────────────────────────────────────────────────────────────────

class Config:
    # Paths  ── update these to your dataset root
    DATA_DIR   = Path("/home/joan/code/ai_im_detect/data/xhlulu/real-vs-fake")          # expects data/train, data/val, data/test
    #              each sub-folder:   real/  and  fake/
    CKPT_DIR   = Path("checkpoints")
    EXPORT_DIR = Path("exported_models")

    # Model
    MODEL_NAME  = "efficientnet_b0"    # or "mobilenetv3_small_100"
    NUM_CLASSES = 2                    # real / fake
    PRETRAINED  = True

    # Training
    IMG_SIZE    = 200
    BATCH_SIZE  = 128
    EPOCHS      = 60
    LR          = 1e-4
    WEIGHT_DECAY= 1e-5
    GRAD_CLIP   = 1.0
    UNFREEZE_EPOCH = 8

    # Augmentation strength (0–1)
    AUG_LEVEL   = 0.5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else (
             "mps"  if torch.backends.mps.is_available() else "cpu")

    SEED = 42

    UNFREEZE_SCHEDULE = {
        # epoch: unfreeze from this block onward
        5:  6,   # epoch 5:  unfreeze only block 6 (deepest, most task-specific)
        10: 4,   # epoch 10: also unfreeze blocks 4-5
        15: 2,   # epoch 15: also unfreeze blocks 2-3
        20: 0,   # epoch 20: unfreeze everything including stem
    }

cfg = Config()
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
cfg.EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── 2. Data Transforms ────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Training  : aggressive augmentation to prevent over-fitting on GAN artefacts.
    Val / Test: deterministic centre-crop only.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE + 32, cfg.IMG_SIZE + 32)),
            transforms.RandomCrop(cfg.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            # artifact already compressed
            # JPEG compression simulation — important for deepfake robustness
            # transforms.RandomApply([
            #     transforms.Lambda(lambda img: _jpeg_compress(img, quality=random.randint(50, 95)))
            # ], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # Random erasing mimics social-media cropping / overlays
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def _jpeg_compress(img: Image.Image, quality: int = 75) -> Image.Image:
    """Round-trip through JPEG to simulate compression artefacts."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ── 3. Dataset ────────────────────────────────────────────────────────────────

class RealFakeDataset(Dataset):
    """
    Expects directory layout:
        root/
          real/   *.jpg  *.png  …
          fake/   *.jpg  *.png  …

    Labels:  real → 0,  fake → 1
    """
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: Path, split: str = "train"):
        self.transform = get_transforms(split)
        self.samples: list[tuple[Path, int]] = []

        for label, folder in enumerate(["real", "fake"]):
            folder_path = root / folder
            if not folder_path.exists():
                raise FileNotFoundError(f"Expected folder: {folder_path}")
            for p in folder_path.rglob("*"):
                if p.suffix.lower() in self.EXTS:
                    self.samples.append((p, label))

        random.shuffle(self.samples)
        print(f"[{split}] loaded {len(self.samples)} images "
              f"(real={sum(1 for _,l in self.samples if l==0)}, "
              f"fake={sum(1 for _,l in self.samples if l==1)})")

    def __len__(self):
        return len(self.samples)

    def sample_patches_face_aware(img: Image.Image, n_patches: int = 8,
                                   face_ratio: float = 0.7) -> list:
        """
        face_ratio: proportion of patches drawn from face regions (vs random)
        """
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = detector.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5)

        transform = get_transforms("test")
        patches = []

        n_face   = int(n_patches * face_ratio) if len(faces) > 0 else 0
        n_random = n_patches - n_face

        # ── Face-region patches ───────────────────────────────────────────
        if n_face > 0:
            W, H = img.size
            for _ in range(n_face):
                # Pick a random detected face
                x, y, w, h = faces[np.random.randint(len(faces))]

                # Expand the box to include boundary artifacts
                # (faceswap blending seams sit just outside the face box)
                pad    = int(max(w, h) * 0.3)
                x1     = max(0, x - pad)
                y1     = max(0, y - pad)
                x2     = min(W, x + w + pad)
                y2     = min(H, y + h + pad)

                region = img.crop((x1, y1, x2, y2))
                region = region.resize((224, 224), Image.BILINEAR)
                patches.append(transform(region))

        # ── Random background patches ─────────────────────────────────────
        W, H = img.size
        for _ in range(n_random):
            if W > 224 and H > 224:
                x = np.random.randint(0, W - 224)
                y = np.random.randint(0, H - 224)
                patch = img.crop((x, y, x + 224, y + 224))
            else:
                patch = img.resize((224, 224))
            patches.append(transform(patch))

        return patches

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

        # Sample N patches per image per epoch
        # smart sampling (detect face then sample around the face) 
        # use when input image are more than just face
        # patches = sample_patches_face_aware(img)
        # labels  = [label] * self.patches_per_image
        # return torch.stack(patches), torch.tensor(labels)


def build_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = RealFakeDataset(cfg.DATA_DIR / "train", "train")
    val_ds   = RealFakeDataset(cfg.DATA_DIR / "val",   "val")
    test_ds  = RealFakeDataset(cfg.DATA_DIR / "test",  "test")

    kw = dict(num_workers=4, pin_memory=(cfg.DEVICE == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ── 4. Model ──────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """
    EfficientNet-B0 with a custom binary-classification head.
    The backbone is frozen for the first few epochs (warm-up), then
    gradually unfrozen for fine-tuning — prevents catastrophic forgetting.
    """
    model = timm.create_model(cfg.MODEL_NAME, pretrained=cfg.PRETRAINED, num_classes=0)
    in_features = model.num_features   # 1280 for EfficientNet-B0

    # Custom head: Dropout → FC → (optional) extra FC → logits
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, cfg.NUM_CLASSES),
    )

    # Freeze backbone initially
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    return model.to(cfg.DEVICE)


def unfreeze_backbone(model: nn.Module, unfreeze_from_block: int = 2):
    """
    Unfreeze all blocks from unfreeze_from_block onward.
    With unfreeze_from_block=2, blocks 2-6 + head all train.
    Early layers (stem, block 0-1) stay frozen — they learn universal
    edge/texture features that don't need task-specific adjustment.
    """
    for name, param in model.named_parameters():
        # Always train the classifier
        if "classifier" in name:
            param.requires_grad = True
            continue
        # Unfreeze from specified block onward
        for b in range(unfreeze_from_block, 7):
            if f"blocks.{b}" in name:   # timm uses "blocks.N" not "blocks_N"
                param.requires_grad = True
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ── 5. Training Loop ──────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_val  = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.counter  = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, loader, optimizer, criterion, scaler) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type=cfg.DEVICE, enabled=(cfg.DEVICE == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def train():
    train_loader, val_loader, _ = build_loaders()
    model     = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1
    )
    scaler     = torch.GradScaler(enabled=(cfg.DEVICE == "cuda"))
    early_stop = EarlyStopping(patience=8)

    # ── Resume logic ──────────────────────────────────────────────────
    start_epoch  = 1
    best_val_acc = 0.0
    resume_ckpt  = cfg.CKPT_DIR / "resume.pt"

    if resume_ckpt.exists():
        print(f"Resuming from {resume_ckpt}")
        state = torch.load(resume_ckpt, map_location=cfg.DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        early_stop   = state["early_stop"]
        start_epoch  = state["epoch"] + 1
        best_val_acc = state["best_val_acc"]
        print(f"  resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    else:
        print("No resume checkpoint found — starting fresh")
    # ─────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        # Progressive unfreeze
        if epoch in cfg.UNFREEZE_SCHEDULE:
            block = cfg.UNFREEZE_SCHEDULE[epoch]
            print(f"\n[Epoch {epoch}] Unfreezing from block {block} onward")
            unfreeze_backbone(model, unfreeze_from_block=block)
            # Rebuild optimizer so new params are included
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.LR * 0.05,
                weight_decay=cfg.WEIGHT_DECAY
            )
            # Rebuild scheduler around remaining epochs
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1
            )
            early_stop.counter = 0   # ← reset patience after each unfreeze
            print(f"  Early stopping counter reset")

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        # ── Save resume checkpoint every epoch ────────────────────────
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "scaler":       scaler.state_dict(),
            "early_stop":   early_stop,
            "best_val_acc": best_val_acc,
        }, resume_ckpt)
        # ─────────────────────────────────────────────────────────────

        ckpt_tag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), cfg.CKPT_DIR / "best_model.pt")
            ckpt_tag = "  ✓ saved"

        print(f"Epoch {epoch:03d}/{cfg.EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{ckpt_tag}")

        if early_stop(vl_loss):
            print("Early stopping triggered.")
            break

    # Delete resume checkpoint on clean finish so next run starts fresh
    if resume_ckpt.exists():
        resume_ckpt.unlink()
        print("Resume checkpoint deleted — next run will start fresh")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return model


# ── 6. Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module):
    """Full evaluation on the test set with precision / recall / F1."""
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score)

    _, _, test_loader = build_loaders()
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(cfg.DEVICE)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    print("\n" + "─"*60)
    print("Classification Report (0=real, 1=fake):")
    print(classification_report(all_labels, all_preds,
                                 target_names=["real", "fake"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC: {auc:.4f}")


# ── 7. Mobile Export ──────────────────────────────────────────────────────────

def load_best_model() -> nn.Module:
    model = build_model()
    model.load_state_dict(torch.load(cfg.CKPT_DIR / "best_model.pt",
                                     map_location=cfg.DEVICE))  # was "cpu"
    return model.eval().to(cfg.DEVICE)


# 7a — ONNX (universal intermediate format)
def export_onnx(model: nn.Module):
    model_cpu = model.cpu()          # move to CPU
    model_cpu.eval()
    out   = cfg.EXPORT_DIR / "detector.onnx"
    dummy = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)   # dummy is already CPU
    torch.onnx.export(
        model_cpu, dummy, str(out),
        export_params=True,
        opset_version=18,
        input_names=["image"],
        output_names=["logits"],
        # dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"ONNX exported → {out}")
    return out


def export_coreml(model: nn.Module):
    try:
        import coremltools as ct

        model_cpu = model.cpu().eval()
        dummy     = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
        traced    = torch.jit.trace(model_cpu, dummy)

        ml_model = ct.convert(
            traced,
            inputs=[ct.ImageType(
                name="image",
                shape=dummy.shape,
                scale=1 / (255.0 * 0.226),
                bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            )],
            outputs=[ct.TensorType(name="logits")],
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT16,
        )

        # ── Check for unconverted ops ─────────────────────────────────
        spec = ml_model.get_spec()
        print("\n[CoreML] Checking for CPU-fallback ops...")
        # Any op not running on Neural Engine shows up in the mil program
        mil_program = ml_model._mil_program
        if mil_program:
            for func in mil_program.functions.values():
                for op in func.operations:
                    if op.op_type not in {"const", "return"}:
                        pass  # all ops — uncomment below to see them all
                        # print(f"  {op.op_type}")
        
        ml_model.short_description = "AI-Generated / Deepfake Image Detector"
        ml_model.input_description["image"]   = "200×200 RGB face image"
        ml_model.output_description["logits"] = "Logits [real, fake]"

        out = cfg.EXPORT_DIR / "RealFakeDetector.mlpackage"
        ml_model.save(str(out))
        print(f"CoreML exported → {out}")

    except ImportError:
        print("CoreML export skipped — install coremltools.")


# 7b — TFLite (Android / cross-platform)
def export_tflite(onnx_path: Path):
    """
    Requires:  pip install onnx-tf tensorflow
    Convert:   ONNX → TF SavedModel → TFLite (INT8 quantised)
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        tf_model_path = cfg.EXPORT_DIR / "tf_model"
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(tf_model_path))

        # INT8 post-training quantisation for smallest model size
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Supply a representative dataset for full INT8 calibration
        # converter.representative_dataset = _representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_model = converter.convert()

        out = cfg.EXPORT_DIR / "detector_int8.tflite"
        out.write_bytes(tflite_model)
        print(f"TFLite (INT8) exported → {out}  "
              f"({out.stat().st_size / 1e6:.1f} MB)")
    except ImportError:
        print("TFLite export skipped — install onnx-tf and tensorflow.")



# ── 8. Inference Helper ───────────────────────────────────────────────────────

class RealFakeDetector:
    LABELS = ["real", "fake"]

    def __init__(self, checkpoint: str | Path = None):
        self.model     = load_best_model() if checkpoint is None else self._load(checkpoint)
        self.device    = next(self.model.parameters()).device  # infer device from model
        self.transform = get_transforms("test")

    def _load(self, path):
        m = build_model()
        m.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
        return m.eval().to(cfg.DEVICE)

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        img    = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # ← move to model's device
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
        return {
            "label":      self.LABELS[idx],
            "confidence": round(probs[idx].item(), 4),
            "scores":     {l: round(p.item(), 4)
                           for l, p in zip(self.LABELS, probs)},
        }


# ── 9. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Generated Face Detector")
    parser.add_argument("--mode", choices=["train", "eval", "export", "infer"],
                        default="train")
    parser.add_argument("--image", type=str, help="Path to image (infer mode)")
    args = parser.parse_args()

    if args.mode == "train":
        trained_model = train()
        print("\nRunning test-set evaluation…")
        best = load_best_model()
        evaluate(best)

    elif args.mode == "eval":
        best = load_best_model()
        evaluate(best)

    elif args.mode == "export":
        best     = load_best_model()
        onnx_out = export_onnx(best)
        export_tflite(onnx_out)
        export_coreml(best)

    elif args.mode == "infer":
        if not args.image:
            print("Provide --image path")
        else:
            detector = RealFakeDetector()
            result   = detector.predict(args.image)
            print(f"\nResult: {result['label'].upper()}  "
                  f"(confidence {result['confidence']:.1%})")
            print(f"  real={result['scores']['real']:.4f}  "
                  f"fake={result['scores']['fake']:.4f}")

