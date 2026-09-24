"""
============================================================================
MODULE 3 — GOOGLE COLAB TRAINING NOTEBOOK
============================================================================
Copy-paste this entire file into a single Google Colab cell, OR paste
each section marked with "# === CELL X ===" into separate cells.

Runtime: GPU → T4 (free tier works, Pro is faster)
Time: ~2-3.5 hours total
============================================================================
"""

# === CELL 1: SETUP & KAGGLE AUTH ===
# Run this cell first. It will ask you to upload kaggle.json.

import os
from google.colab import files

# Install dependencies
!pip install -q torch torchvision kaggle Pillow pandas tqdm scikit-learn numpy

# Kaggle API setup
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Check if kaggle.json already exists
if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
    print("="*50)
    print("  Upload your kaggle.json file now")
    print("  (Download from https://www.kaggle.com/settings)")
    print("="*50)
    uploaded = files.upload()  # This opens an upload dialog
    for fn in uploaded:
        with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
            f.write(uploaded[fn])

!chmod 600 ~/.kaggle/kaggle.json

# Verify Kaggle works
!kaggle datasets list --max-size 1 > /dev/null 2>&1 && echo "✓ Kaggle API works" || echo "✗ Kaggle API failed — re-upload kaggle.json"

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ No GPU — go to Runtime → Change runtime type → GPU")


# === CELL 2: DOWNLOAD ALL 5 DATASETS ===
# ~20-45 minutes depending on network

WORK_DIR = "/content/tamper_training"
RAW_DIR = f"{WORK_DIR}/raw"
os.makedirs(RAW_DIR, exist_ok=True)
os.chdir(WORK_DIR)

print("="*50)
print("  Downloading 5 datasets")
print("="*50)

# Upgrade kaggle first to prevent outdated version warnings
!pip install -q --upgrade kaggle

# 1. DocXPand-25k (~4GB -> 17GB unzipped)
print("\n[1/5] DocXPand-25k...")
if not os.path.exists(f"{RAW_DIR}/docxpand"):
    !kaggle datasets download -d satishlokkoju/docxpand-25k --unzip -p {RAW_DIR}/docxpand || echo "  ⚠ Failed, continuing without"
    print("  ✓ Done (or skipped)")
else:
    print("  ✓ Already exists")

# 2. CASIA v2 (~6.8GB unzipped)
print("\n[2/5] CASIA v2...")
if not os.path.exists(f"{RAW_DIR}/casia"):
    # Try alternative slugs as the original sometimes gives 403 Forbidden
    !kaggle datasets download -d awsaf49/casia-20-image-tampering-detection-dataset --unzip -p {RAW_DIR}/casia || kaggle datasets download -d sophatvathana/casia-dataset --unzip -p {RAW_DIR}/casia || echo "  ⚠ Failed, continuing without"
    print("  ✓ Done (or skipped)")
else:
    print("  ✓ Already exists")

# 3. FantasyID (~3-5GB)
print("\n[3/5] FantasyID...")
if not os.path.exists(f"{RAW_DIR}/fantasyid"):
    !kaggle datasets download -d wricha/fantasyid --unzip -p {RAW_DIR}/fantasyid || echo "  ⚠ Failed, continuing without"
    print("  ✓ Done (or skipped)")
else:
    print("  ✓ Already exists")

# 4. DocTamper Dataset (Text Tampering)
print("\n[4/5] DocTamper (Text Tampering)...")
if not os.path.exists(f"{RAW_DIR}/doctamper"):
    !kaggle datasets download -d dinmkeljiame/doctamper --unzip -p {RAW_DIR}/doctamper || echo "  ⚠ Failed, continuing without"
    print("  ✓ Done (or skipped)")
else:
    print("  ✓ Already exists")

# 5. SIDTD (GitHub ~68MB)
print("\n[5/5] SIDTD...")
if not os.path.exists(f"{RAW_DIR}/sidtd/SIDTD_Dataset"):
    !git clone --depth 1 https://github.com/Oriolrt/SIDTD_Dataset.git {RAW_DIR}/sidtd/SIDTD_Dataset || echo "  ⚠ Failed, continuing without"
    print("  ✓ Done (or skipped)")
else:
    print("  ✓ Already exists")

# Show what we got
print("\n" + "="*50)
!du -sh {RAW_DIR}/*
print("="*50)


# === CELL 3 (CORRECTED): NORMALIZE DATASETS — SYMLINK-ONLY, LMDB-AWARE ===
# ~5-15 minutes (doctamper label-scan is the slow part)
#
# What changed vs your previous run:
#   - docxpand: only "documents/" is used (fields/ dropped — those are per-field crops)
#   - casia:    Au / Sp folder detection made more defensive (this is CASIA1, not v2)
#   - fantasyid: uses the REAL structure (train|test)/(bonafide|attack), not keyword guessing
#   - doctamper: NOT extracted to files (would need ~20GB you don't have). Instead we keep
#                it as LMDB and just index it — a custom Dataset class reads directly from
#                the .mdb files at train time, zero-copy, zero extra disk.
#   - sidtd:    skipped with a warning — the git clone only pulled code, not the dataset.
#
# This wipes and rebuilds normalized/ from scratch (previous run's symlinks were incomplete
# for fantasyid/casia and totally empty for doctamper).

import os, csv, json, shutil
from pathlib import Path
from collections import Counter

WORK_DIR = "/content/tamper_training"
RAW_DIR = Path(f"{WORK_DIR}/raw")
NORM_DIR = Path(f"{WORK_DIR}/normalized")
IMG_DIR = NORM_DIR / "images"
LABELS_CSV = NORM_DIR / "labels.csv"
DOCTAMPER_INDEX_CSV = NORM_DIR / "doctamper_index.csv"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --- wipe previous (incorrect) normalization; symlinks only, so this frees ~0 real data ---
if NORM_DIR.exists():
    shutil.rmtree(NORM_DIR)
IMG_DIR.mkdir(parents=True, exist_ok=True)

def find_images(directory, recursive=True):
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = "**/*" if recursive else "*"
    images = []
    for ext in IMG_EXTS:
        images.extend(directory.glob(f"{pattern}{ext}"))
        images.extend(directory.glob(f"{pattern}{ext.upper()}"))
    return sorted(set(images))

def symlink_image(src: Path, dst_name: str) -> str:
    """Symlink only — never copies bytes. Returns the filename written into labels.csv."""
    ext = src.suffix.lower()
    if ext not in IMG_EXTS:
        ext = ".jpg"
    dst = IMG_DIR / f"{dst_name}{ext}"
    if not dst.exists():
        os.symlink(str(src.resolve()), str(dst))
    return dst.name

all_rows = []  # (filename_or_ref, photo_replacement, text_manipulation, compression_anomaly, source, storage)
idx = 0

# ---------------------------------------------------------------------------
# 1) DocXPand-25k — all authentic, only the "documents" tree (skip "fields")
# ---------------------------------------------------------------------------
print("Processing DocXPand-25k (all authentic)...")
dx_documents = RAW_DIR / "docxpand" / "DocXPand-25k" / "documents"
if dx_documents.exists():
    dx_count = 0
    for img in find_images(dx_documents):
        fname = symlink_image(img, f"docxpand_{idx}")
        all_rows.append((fname, 0, 0, 0, "docxpand", "symlink"))
        idx += 1; dx_count += 1
    print(f"  DocXPand: {dx_count} images (documents/ only)")
else:
    print(f"  ⚠ Not found at expected path: {dx_documents}")

# ---------------------------------------------------------------------------
# 2) CASIA1 — Au (authentic) vs Sp (spliced/tampered); defensive matching
# ---------------------------------------------------------------------------
print("Processing CASIA...")
casia_dir = RAW_DIR / "casia"
if casia_dir.exists():
    auth_dirs, tamp_dirs = [], []
    for d in casia_dir.rglob("*"):
        if d.is_dir():
            nl = d.name.lower()
            if nl in ("au", "authentic", "real", "original"):
                auth_dirs.append(d)
            elif nl in ("sp", "tp", "tampered", "fake", "forged", "spliced", "copymove"):
                tamp_dirs.append(d)
    if not auth_dirs and not tamp_dirs:
        print("  ⚠ No Au/Sp-style folders found — check structure manually:")
        for d in sorted({p.parent for p in find_images(casia_dir)}):
            print("   ", d)
    else:
        a_count = t_count = 0
        for adir in auth_dirs:
            for img in find_images(adir, recursive=False):
                fname = symlink_image(img, f"casia_auth_{idx}")
                all_rows.append((fname, 0, 0, 0, "casia", "symlink")); idx += 1; a_count += 1
        for tdir in tamp_dirs:
            for img in find_images(tdir, recursive=False):
                fname = symlink_image(img, f"casia_tamp_{idx}")
                all_rows.append((fname, 1, 0, 0, "casia", "symlink")); idx += 1; t_count += 1
        print(f"  CASIA: {a_count} authentic, {t_count} tampered  (auth dirs: {[d.name for d in auth_dirs]}, tamp dirs: {[d.name for d in tamp_dirs]})")
else:
    print("  ⚠ CASIA not found")

# ---------------------------------------------------------------------------
# 3) FantasyID — REAL structure: (train|test)/(bonafide|attack)/**
#    "attack" = face-swap presentation attacks -> photo_replacement=1
# ---------------------------------------------------------------------------
print("Processing FantasyID...")
fid_root = RAW_DIR / "fantasyid" / "FantasyID"
if fid_root.exists():
    bf_count = at_count = 0
    for split in ("train", "test"):
        bonafide_dir = fid_root / split / "bonafide"
        attack_dir = fid_root / split / "attack"
        for img in find_images(bonafide_dir):
            fname = symlink_image(img, f"fid_bf_{idx}")
            all_rows.append((fname, 0, 0, 0, "fantasyid", "symlink")); idx += 1; bf_count += 1
        for img in find_images(attack_dir):
            fname = symlink_image(img, f"fid_at_{idx}")
            # face-swap presentation attack: photo tampering=1, text n/a, compression n/a (unknown)
            all_rows.append((fname, 1, 0, -1, "fantasyid", "symlink")); idx += 1; at_count += 1
    print(f"  FantasyID: {bf_count} bonafide, {at_count} attack")
else:
    print(f"  ⚠ Not found at expected path: {fid_root}")

# ---------------------------------------------------------------------------
# 4) DocTamper — pure LMDB. We do NOT extract. We index it: for each of the
#    4 LMDB envs, we scan the mask ('label-%09d') for each sample and record
#    whether it's tampered (mask.max()>0), plus which env+index to read at
#    train time. A DocTamperLMDB Dataset class (written to disk below) reads
#    straight from the .mdb files — zero additional disk usage.
# ---------------------------------------------------------------------------
print("Processing DocTamper (indexing LMDB, not extracting)...")
try:
    import lmdb
    import numpy as np
    import cv2

    doctamper_dir = RAW_DIR / "doctamper"
    dt_subsets = ["DocTamperV1-FCD", "DocTamperV1-SCD",
                  "DocTamperV1-TrainingSet", "DocTamperV1-TestingSet"]

    dt_rows = []
    for subset in dt_subsets:
        subset_path = doctamper_dir / subset
        if not subset_path.exists():
            print(f"  ⚠ {subset} not found, skipping")
            continue
        env = lmdb.open(str(subset_path), readonly=True, lock=False,
                         readahead=False, meminit=False, max_readers=32)
        with env.begin(write=False) as txn:
            n = txn.get(b"num-samples")
            if n is None:
                print(f"  ⚠ {subset}: no 'num-samples' key found, skipping")
                env.close()
                continue
            n = int(n)
            pos = 0
            for i in range(n):
                lbl_key = ("label-%09d" % i).encode("utf-8")
                lblbuf = txn.get(lbl_key)
                tampered = 0
                if lblbuf is not None:
                    mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
                    if mask is not None and mask.max() > 0:
                        tampered = 1
                        pos += 1
                # reference row: no filename/symlink, just env subset + index
                dt_rows.append((subset, i, tampered))
        env.close()
        print(f"  {subset}: {n} samples, {pos} tampered / {n - pos} clean")

    with open(DOCTAMPER_INDEX_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lmdb_subset", "lmdb_index", "text_manipulation"])
        w.writerows(dt_rows)

    # also fold a reference into the main label table so total counts make sense,
    # using storage="lmdb" + a "ref" column pointing back into doctamper_index.csv
    for subset, i, tampered in dt_rows:
        ref = f"{subset}:{i}"
        all_rows.append((ref, -1, tampered, -1, "doctamper", "lmdb"))

    print(f"  ✓ DocTamper indexed: {len(dt_rows)} samples referenced (0 bytes copied/extracted)")
except ImportError:
    print("  ⚠ lmdb/opencv not installed — run: pip install lmdb opencv-python-headless")
except Exception as e:
    print(f"  ⚠ DocTamper indexing failed: {e}")

# ---------------------------------------------------------------------------
# 5) SIDTD — skipped. Repo clone only has code, not the actual dataset.
# ---------------------------------------------------------------------------
print("Processing SIDTD...")
print("  ⚠ SKIPPED: /raw/sidtd/SIDTD_Dataset is the GitHub *code* repo, not the dataset.")
print("    Only a handful of test-fixture images exist in it (SIDTD/utils/Test_Samples).")
print("    Get the real data from the official Dataverse release:")
print("    https://doi.org/10.34810/data1815")
print("    Re-run this cell after downloading it into raw/sidtd/ — I'll wire it in then.")

# --- write labels.csv ---
with open(LABELS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ref", "photo_replacement", "text_manipulation", "compression_anomaly", "source", "storage"])
    writer.writerows(all_rows)

sources = Counter(r[4] for r in all_rows)
print(f"\n{'='*50}")
print(f"  TOTAL: {len(all_rows)} samples indexed (symlinks + lmdb refs)")
for src, cnt in sorted(sources.items()):
    print(f"    {src}: {cnt}")
for hi, hn in [(1, "photo_replacement"), (2, "text_manipulation"), (3, "compression_anomaly")]:
    pos = sum(1 for r in all_rows if r[hi] == 1)
    neg = sum(1 for r in all_rows if r[hi] == 0)
    msk = sum(1 for r in all_rows if r[hi] == -1)
    print(f"  {hn}: {pos} pos / {neg} neg / {msk} masked(unknown)")
print(f"{'='*50}")
print("\nDisk check:")
!du -sh {NORM_DIR}
!df -h /content


# === CELL 4: PRECOMPUTE ELA CACHE ===
import os, io, re, csv, random, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, lmdb
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm.notebook import tqdm

# ---------------- config ----------------
ELA_DIR  = NORM_DIR / "ela_cache_v2"                # dynamic path instead of hardcoded /content
ELA_CSV  = NORM_DIR / "labels_ela.csv"              # cell 5 must read THIS, not labels.csv
IMG_DIR  = NORM_DIR / "images"                      # where cell 3 put the symlinks
ELA_SIZE, ELA_Q, ELA_SCALE = 224, 90, 20            # keep identical at inference
LMDB_CAP = 15000                                    # max per DocTamper subset (None = all)
DEDUPE, SEED = True, 42
ELA_DIR.mkdir(parents=True, exist_ok=True)
assert IMG_DIR.exists(), f"IMG_DIR not found: {IMG_DIR}"

# ---------------- worker code ----------------
_ENVS = {}
def _env(path):
    if path not in _ENVS:
        _ENVS[path] = lmdb.open(path, readonly=True, lock=False, readahead=False,
                                meminit=False, max_readers=64)
    return _ENVS[path]

def load_image(ref, storage):
    if storage == "symlink":
        return Image.open(IMG_DIR / ref).convert("RGB")
    if storage == "lmdb":
        subset, i = ref.split(":")
        with _env(str(Path(RAW_DIR) / "doctamper" / subset)).begin(write=False) as txn:
            buf = txn.get(("image-%09d" % int(i)).encode())
        if buf is None:
            raise KeyError(f"no LMDB key for {ref}")
        return Image.open(io.BytesIO(buf)).convert("RGB")
    raise ValueError(f"unknown storage {storage}")

def make_ela(orig):
    b = io.BytesIO()
    orig.save(b, "JPEG", quality=ELA_Q); b.seek(0)
    comp = Image.open(b).convert("RGB")
    diff = np.abs(np.asarray(orig, np.int16) - np.asarray(comp, np.int16))
    ela = np.clip(diff * ELA_SCALE, 0, 255).astype(np.uint8)
    return Image.fromarray(ela).resize((ELA_SIZE, ELA_SIZE), Image.BILINEAR)

def process_ela(args):
    ref, storage, dst = args
    if os.path.exists(dst):
        return dst, True, ""
    tmp = dst + ".tmp"
    try:
        make_ela(load_image(ref, storage)).save(tmp, format="JPEG", quality=95)
        os.replace(tmp, dst)                          # atomic: no half-written files
        return dst, True, ""
    except Exception as e:
        if os.path.exists(tmp): os.remove(tmp)
        return dst, False, f"{type(e).__name__}: {e}"  # NO placeholder image written

def ela_name(ref):                                    # unique, keeps extension
    return re.sub(r"[^A-Za-z0-9_.-]", "_", ref) + ".jpg"

# ---------------- build work list ----------------
with open(LABELS_CSV) as f:
    reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
random.Random(SEED).shuffle(rows)
print("labels.csv rows:", len(rows))

if DEDUPE:                                            # catches the CASIA multi-copy problem
    seen, kept, dup = set(), [], 0
    for r in tqdm(rows, desc="dedupe"):
        if r["storage"] == "symlink":
            try:
                p = IMG_DIR / r["ref"]
                with open(p, "rb") as fh: head = fh.read(65536)
                fp = (os.path.getsize(p), hashlib.md5(head).hexdigest())
            except Exception:
                kept.append(r); continue              # ELA step will report it
            if fp in seen: dup += 1; continue
            seen.add(fp)
        kept.append(r)
    rows = kept
    print(f"removed {dup} duplicate files")

if LMDB_CAP:
    cnt, defaultdict(int), []
    cnt = defaultdict(int)
    kept = []
    for r in rows:
        if r["storage"] == "lmdb":
            k = r["ref"].split(":")[0]
            if cnt[k] >= LMDB_CAP: continue
            cnt[k] += 1
        kept.append(r)
    rows = kept

dsts = [ELA_DIR / ela_name(r["ref"]) for r in rows]
assert len(set(dsts)) == len(dsts), "cache filename collision!"
todo = [(r["ref"], r["storage"], str(d)) for r, d in zip(rows, dsts) if not d.exists()]
print(f"images: {len(rows)} | cached: {len(rows)-len(todo)} | to process: {len(todo)}")

# ---------------- preflight (20 random items, main process) ----------------
bad = [(a[0], e) for a in todo[:20] for _, ok, e in [process_ela(a)] if not ok]
for e in _ENVS.values(): e.close()                    # never fork with open LMDB envs
_ENVS.clear()
if bad:
    raise RuntimeError(f"Preflight failed, fix before continuing: {bad[:3]}")
todo = [a for a in todo if not os.path.exists(a[2])]

# ---------------- parallel run ----------------
errors = Counter()
workers = min(os.cpu_count() or 4, 8)
try:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for dst, ok, err in tqdm(ex.map(process_ela, todo, chunksize=32),
                                 total=len(todo), desc="ELA"):
            if not ok: errors[err[:120]] += 1
except Exception as e:
    print("Pool interrupted:", repr(e), "-> re-run this cell, it resumes where it left off.")

# ---------------- write the CSV cell 5 will actually use ----------------
good = [(r, d) for r, d in zip(rows, dsts) if d.exists()]
with open(ELA_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields + ["ela_path"]); w.writeheader()
    for r, d in good: w.writerow({**r, "ela_path": str(d)})

print(f"\n✓ usable: {len(good)} | failed/missing: {len(rows)-len(good)}")
for msg, n in errors.most_common(5): print(f"  {n:>6} × {msg}")
print("by storage:", Counter(r["storage"] for r, _ in good))
print(f"cache size: {sum(d.stat().st_size for _, d in good)/1e9:.2f} GB")


# === CELL 5: MODEL + DATASET + LOSS + EVAL ===
import csv, time, json, random
from pathlib import Path
from collections import Counter, defaultdict
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

HEADS = ["photo_replacement", "text_manipulation", "compression_anomaly"]

# --- Model (unchanged) ---
class MultiHeadResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        bb = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(bb.children())[:-1])
        def head(): return nn.Sequential(nn.Dropout(0.3), nn.Linear(512,128), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(128,2))
        self.head_photo, self.head_text, self.head_comp = head(), head(), head()
    def forward(self, x):
        f = self.features(x).flatten(1)
        return {"photo_replacement": self.head_photo(f),
                "text_manipulation": self.head_text(f),
                "compression_anomaly": self.head_comp(f)}

# --- Samples + per-source split ---
def infer_source(row):
    for k in ("source", "dataset"):
        if row.get(k): return row[k]
    ref = row["ref"]
    if row["storage"] == "lmdb": return "doctamper_" + ref.split(":")[0]
    parts = Path(ref).parts
    return parts[0] if len(parts) > 1 else Path(ref).name.split("_")[0]

def load_samples(csv_path):
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not Path(row["ela_path"]).exists(): continue
            out.append({"p": row["ela_path"], "src": infer_source(row),
                        "y": [int(row[h]) for h in HEADS]})
    return out

def split_by_source(samples, val_frac=0.1, seed=42):
    rng, by = random.Random(seed), defaultdict(list)
    for s in samples: by[s["src"]].append(s)
    tr, va = [], []
    for items in by.values():
        rng.shuffle(items); n = max(1, int(len(items) * val_frac))
        va += items[:n]; tr += items[n:]
    return tr, va

def label_report(samples):
    tab = defaultdict(Counter)
    for s in samples:
        for i, h in enumerate(HEADS): tab[(s["src"], h)][s["y"][i]] += 1
    for (src, h), c in sorted(tab.items()):
        print(f"{src:26s} {h:22s} {dict(sorted(c.items()))}")

# --- Dataset ---
NORM = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
TRAIN_TF = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORM])  # no jitter/rotation: they alter the ELA signal
VAL_TF   = transforms.Compose([transforms.ToTensor(), NORM])

class TamperDS(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        try:
            with Image.open(s["p"]) as im: t = self.transform(im.convert("RGB"))
            y = s["y"]
        except Exception:
            t, y = torch.zeros(3, 224, 224), [-1, -1, -1]      # masked out, never trained on
        return t, dict(zip(HEADS, y))

def collate(batch):
    ts = torch.stack([b[0] for b in batch])
    ls = {k: torch.tensor([b[1][k] for b in batch], dtype=torch.long) for k in HEADS}
    return ts, ls

# --- Loss (masked + per-head class weights) ---
def head_class_weights(samples):
    w = {}
    for i, h in enumerate(HEADS):
        c = Counter(s["y"][i] for s in samples if s["y"][i] != -1)
        n0, n1 = c.get(0, 0), c.get(1, 0); tot = n0 + n1
        w[h] = torch.tensor([tot/(2*n0), tot/(2*n1)]) if n0 and n1 else None
    return w

def masked_loss(logits, labels, weights=None):
    losses = []
    for k in logits:
        m = labels[k] != -1
        if m.any():
            w = weights[k].to(logits[k].device) if weights and weights.get(k) is not None else None
            losses.append(F.cross_entropy(logits[k][m].float(), labels[k][m], weight=w))
    if not losses: return sum(v.sum() for v in logits.values()) * 0.0
    return torch.stack(losses).mean()

# --- Per-source, per-head eval (val loader must have shuffle=False) ---
@torch.no_grad()
def evaluate(model, loader, samples, device):
    model.eval(); i0 = 0; acc = defaultdict(lambda: [0, 0])
    for x, y in loader:
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x.to(device))
        for h in HEADS:
            pred = out[h].argmax(1).cpu()
            for j in range(x.size(0)):
                t = y[h][j].item()
                if t == -1: continue
                a = acc[(samples[i0+j]["src"], h)]; a[0] += int(pred[j].item() == t); a[1] += 1
        i0 += x.size(0)
    for (src, h), (c, n) in sorted(acc.items()):
        print(f"  {src:26s} {h:22s} acc {c/n:.3f}  (n={n})")
    return acc

# --- Build + report ---
samples = load_samples(ELA_CSV)
train_s, val_s = split_by_source(samples)
print(f"train {len(train_s)} | val {len(val_s)}")
label_report(samples)          # read this before training
print("✓ Model, Dataset, Loss, Eval defined")


# === CELL 6: TRAIN (photo head only, based on label_report) ===
import shutil, random
import torch.optim as optim

EPOCHS, BATCH, DOCX_CAP = 10, 64, 6000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path(WORK_DIR) / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_CKPT = None   # e.g. Path("/content/drive/MyDrive/tamper_ckpt"); set it or a disconnect loses the run
if DRIVE_CKPT: DRIVE_CKPT.mkdir(parents=True, exist_ok=True)

# Keep only samples with a photo label; mask text + compression heads (-1 = ignored by the loss)
def photo_only(ss): return [{**s, "y": [s["y"][0], -1, -1]} for s in ss if s["y"][0] != -1]
train_p, val_p = photo_only(train_s), photo_only(val_s)
rng = random.Random(0)
dx   = [s for s in train_p if s["src"] == "docxpand"]
rest = [s for s in train_p if s["src"] != "docxpand"]
rng.shuffle(dx); train_p = rest + dx[:DOCX_CAP]
print("train:", dict(Counter((s["src"], s["y"][0]) for s in train_p)))
print("val:  ", dict(Counter((s["src"], s["y"][0]) for s in val_p)))

train_dl = DataLoader(TamperDS(train_p, TRAIN_TF), batch_size=BATCH, shuffle=True, num_workers=4,
                      collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=True)
val_dl   = DataLoader(TamperDS(val_p, VAL_TF), batch_size=BATCH, shuffle=False, num_workers=4,
                      collate_fn=collate, pin_memory=True, persistent_workers=True)
weights = head_class_weights(train_p)
print(f"Training on {DEVICE} | train {len(train_p)} | val {len(val_p)} | photo weights {weights['photo_replacement']}")

@torch.no_grad()
def evaluate2(model, loader, samples, device, weights):
    model.eval(); i0 = 0; vloss = 0.0; nb = 0
    st = defaultdict(lambda: [[0, 0], [0, 0]])          # (src, head) -> [class][correct, total]
    for x, y in loader:
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x.to(device))
        vloss += masked_loss(out, {k: v.to(device) for k, v in y.items()}, weights).item(); nb += 1
        for h in HEADS:
            pred = out[h].argmax(1).cpu()
            for j in range(x.size(0)):
                t = y[h][j].item()
                if t == -1: continue
                g = st[(samples[i0 + j]["src"], h)][t]
                g[0] += int(pred[j].item() == t); g[1] += 1
        i0 += x.size(0)
    scores = []
    for (src, h), (c0, c1) in sorted(st.items()):
        acc = (c0[0] + c1[0]) / (c0[1] + c1[1])
        if c0[1] and c1[1]:
            bal = (c0[0] / c0[1] + c1[0] / c1[1]) / 2; scores.append(bal)
            print(f"  {src:12s} {h:20s} acc {acc:.3f} | bal {bal:.3f} | authentic-recall {c0[0]/c0[1]:.3f} tamper-recall {c1[0]/c1[1]:.3f} (n={c0[1]+c1[1]})")
        else:
            print(f"  {src:12s} {h:20s} acc {acc:.3f} | single-class, not scored (n={c0[1]+c1[1]})")
    return vloss / max(nb, 1), (sum(scores) / len(scores) if scores else 0.0)

model = MultiHeadResNet18(pretrained=True).to(DEVICE)
backbone = list(model.features.parameters())
heads = [p for n, p in model.named_parameters() if not n.startswith("features.")]
optimizer = optim.Adam([{"params": backbone, "lr": 1e-4}, {"params": heads, "lr": 1e-3}], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

best_score, log = -1.0, []
for epoch in range(EPOCHS):
    t0 = time.time(); model.train(); tloss = 0.0; corr = tot = 0
    for x, y in train_dl:
        x = x.to(DEVICE, non_blocking=True); y = {k: v.to(DEVICE) for k, v in y.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            out = model(x); loss = masked_loss(out, y, weights)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        tloss += loss.item()
        m = y["photo_replacement"] != -1
        corr += (out["photo_replacement"][m].argmax(1) == y["photo_replacement"][m]).sum().item(); tot += m.sum().item()
    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS} train done, evaluating...")
    vl, score = evaluate2(model, val_dl, val_p, DEVICE, weights)
    dt = time.time() - t0
    print(f"Epoch {epoch+1} | train loss {tloss/len(train_dl):.4f} | train acc {corr/max(tot,1):.3f} | "
          f"val loss {vl:.4f} | mixed-source bal acc {score:.3f} | {dt:.0f}s")

    torch.save(model.state_dict(), str(CKPT_DIR / "last.pth"))
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), str(CKPT_DIR / "tamper_multihead_resnet18.pth"))
        print(f"  ★ best saved (bal acc {score:.3f})")
    if DRIVE_CKPT:
        for f in CKPT_DIR.glob("*.pth"): shutil.copy(f, DRIVE_CKPT / f.name)
    log.append({"epoch": epoch + 1, "val_loss": round(vl, 4), "mixed_bal_acc": round(score, 4), "time": round(dt, 1)})

json.dump(log, open(CKPT_DIR / "training_log.json", "w"), indent=2)
print(f"\nDone. Best mixed-source balanced acc: {best_score:.3f}")


# === CELL 7: SAVE + DOWNLOAD CHECKPOINT ===
import os, json, shutil
from pathlib import Path
from google.colab import files

ckpt_dir = Path(WORK_DIR) / "checkpoints"
ckpt = ckpt_dir / "tamper_multihead_resnet18.pth"
assert ckpt.exists(), "Checkpoint not found, did training finish at least one epoch?"

# Inference must match training exactly
meta = {"ela_quality": 90, "ela_scale": 20, "ela_size": 224, "resize": "bilinear",
        "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225],
        "use_head": "photo_replacement", "output": "softmax(logits)[1] = tamper probability"}
json.dump(meta, open(ckpt_dir / "inference_meta.json", "w"), indent=2)

# Backup to Drive (mount first: from google.colab import drive; drive.mount('/content/drive'))
drive = Path("/content/drive/MyDrive/tamper_ckpt")
if drive.parent.exists():
    drive.mkdir(exist_ok=True)
    for f in ckpt_dir.glob("*"):
        if f.is_file(): shutil.copy(f, drive / f.name)
    print("Copied to Drive:", drive)

print(f"Checkpoint size: {ckpt.stat().st_size/1e6:.1f} MB")
for name in ["tamper_multihead_resnet18.pth", "inference_meta.json", "training_log.json"]:
    p = ckpt_dir / name
    if p.exists(): files.download(str(p))
