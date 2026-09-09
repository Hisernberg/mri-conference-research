# CELL 0
# Run once per Kaggle session. Kaggle GPU images already ship torch, sklearn and Pillow,
# so this only installs what is missing and is safe to re-run.
import importlib, subprocess, sys

def ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        importlib.import_module(import_name)
        return False
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
        return True

for pkg, name in [("Pillow", "PIL"), ("scikit-learn", "sklearn"), ("scipy", "scipy"),
                  ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("seaborn", "seaborn"),
                  ("openpyxl", "openpyxl")]:
    if ensure(pkg, name):
        print("installed:", pkg)

print("Environment ready.")

# CELL 1
import os, re, glob, json, math, random, time, platform, contextlib, warnings, hashlib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import PIL
from PIL import Image
from scipy.ndimage import rotate as nd_rotate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from scipy.stats import ttest_rel, wilcoxon

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # Without this, every dataloader worker shares the parent NumPy seed and the
    # random augmentations repeat across workers.
    ws = torch.initial_seed() % (2 ** 32)
    np.random.seed(ws)
    random.seed(ws)

seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.cuda.amp.* is deprecated in recent PyTorch. These wrappers work on old and new versions.
def amp_autocast(enabled):
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)

def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

def safe_load_state_dict(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

ENV_INFO = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "pillow": PIL.__version__,
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "seed": SEED,
}
for k, v in ENV_INFO.items():
    print(f"{k:>10}: {v}")

# CELL 2
class CONFIG:
    # ---- paths ----
    DATA_ROOT = os.environ.get(
        "MSCANET_DATA_ROOT",
        "/kaggle/input/datasets/navoneel/brain-mri-images-for-brain-tumor-detection",
    )
    OUTPUT_DIR = "/kaggle/working/mscanet_experiments" if os.path.isdir("/kaggle/working") \
        else os.path.abspath("./mscanet_experiments")

    # If the dataset is a flat folder of images plus a label file:
    LABEL_FILE_CANDIDATES = ["labels.csv", "labels.xlsx", "metadata.csv"]
    SUBJECT_ID_COLUMN_CANDIDATES = ["subject_id", "Subject", "Subject ID", "ID", "Image", "image", "filename"]
    LABEL_COLUMN_CANDIDATES = ["label", "Label", "Group", "Class", "class", "diagnosis"]
    # Folder or column names that are not diagnostic classes and must not become labels.
    # 'brain_tumor_dataset' is this dataset's own wrapper folder; it never sits directly
    # above an image so it never gets treated as a class, but it's excluded defensively.
    NON_CLASS_DIR_NAMES = {"images", "data", "brain_tumor_dataset", "raw", "train", "test", "val"}

    # ---- image handling ----
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")
    IN_CHANNELS = 1                  # grayscale; most source images are near-monochrome MRI scans anyway

    # ---- preprocessing ----
    TARGET_SHAPE = (128, 128)        # (H, W)
    CLIP_STD = 5.0
    FLIP_AXIS = 1                    # left-right axis of a (H, W) array; set None to disable flipping
    MAX_ROTATION_DEG = 10

    # ---- cross-validation ----
    N_FOLDS = 5
    INNER_VAL_FRACTION = 0.15        # held out of each training fold for early stopping

    # ---- training ----
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    EPOCHS = 60
    EARLY_STOP_PATIENCE = 10
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    GRAD_CLIP = 1.0
    BASE_CHANNELS = 16
    DROPOUT = 0.4
    USE_AMP = True
    CACHE_VOLUMES = False            # only enable with NUM_WORKERS = 0 and enough RAM

    RUN_BASELINES = True
    RUN_ABLATION = True

    # ---- smoke test ----
    # True runs the whole pipeline on a small synthetic dataset in a few minutes.
    # Use it to verify the code path, then set back to False for the real experiment.
    SMOKE_TEST = False

    @classmethod
    def as_dict(cls):
        return {k: v for k, v in vars(cls).items()
                if not k.startswith("_") and not callable(v) and not isinstance(v, classmethod)}

CONFIG.FIGURES_DIR = os.path.join(CONFIG.OUTPUT_DIR, "figures")
CONFIG.CKPT_DIR = os.path.join(CONFIG.OUTPUT_DIR, "checkpoints")
CONFIG.RESULTS_DIR = os.path.join(CONFIG.OUTPUT_DIR, "results")
for d in [CONFIG.OUTPUT_DIR, CONFIG.FIGURES_DIR, CONFIG.CKPT_DIR, CONFIG.RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("Output dir:", CONFIG.OUTPUT_DIR)
print("DATA_ROOT :", CONFIG.DATA_ROOT)

# CELL 3
def make_smoke_dataset(root, classes=("yes", "no"), images_per_class=16,
                       shape=(64, 64), seed=SEED):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    for ci, c in enumerate(classes):
        d = os.path.join(root, c)
        os.makedirs(d, exist_ok=True)
        for s in range(images_per_class):
            centre = (shape[0] // 2 + (5 if ci == 0 else -5), shape[1] // 2)
            img = rng.normal(120.0, 20.0, shape)
            blob = np.exp(-(((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) / 60.0))
            img += (80.0 * blob) if ci == 0 else 0.0     # only class 0 ("yes") gets a bright lesion
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(os.path.join(d, f"{c}_{s:03d}.png"))
    return root


if CONFIG.SMOKE_TEST:
    smoke_root = os.path.join(CONFIG.OUTPUT_DIR, "_smoke_data")
    make_smoke_dataset(smoke_root)
    CONFIG.DATA_ROOT = smoke_root
    CONFIG.TARGET_SHAPE = (64, 64)
    CONFIG.BASE_CHANNELS = 4
    CONFIG.N_FOLDS = 3
    CONFIG.EPOCHS = 2
    CONFIG.EARLY_STOP_PATIENCE = 2
    CONFIG.BATCH_SIZE = 4
    CONFIG.NUM_WORKERS = 0
    CONFIG.USE_AMP = False
    print("SMOKE TEST mode. Synthetic data at:", smoke_root)
else:
    print("Smoke test disabled. Using the real dataset.")

with open(os.path.join(CONFIG.RESULTS_DIR, "config.json"), "w") as fh:
    json.dump({"config": {k: str(v) for k, v in CONFIG.as_dict().items()}, "env": ENV_INFO}, fh, indent=2)

# CELL 4
def find_image_files(root):
    # os.walk with a lower-cased suffix test, so .JPG/.Jpeg are matched too and a file
    # is never counted twice by overlapping glob patterns.
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(CONFIG.IMAGE_EXTS):
                out.append(os.path.join(dirpath, fn))
    return sorted(set(out))


def file_hash(path, block_size=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dedup_by_content(files):
    # This exact dataset ships every image twice: once under DATA_ROOT/<class>/ and again
    # under DATA_ROOT/brain_tumor_dataset/<class>/. Grouping naively would double-count
    # every scan and silently double the reported class sizes. A content hash collapses
    # byte-identical copies wherever they live in the tree, keeping the first path seen.
    seen = {}
    kept = []
    for f in files:
        key = file_hash(f)
        if key not in seen:
            seen[key] = f
            kept.append(f)
    n_dupes = len(files) - len(kept)
    if n_dupes:
        print(f"Removed {n_dupes} duplicate file(s) (identical content found at more than one path).")
    return kept


def normalise_label(x):
    return str(x).strip().replace("_", " ").strip()


def extract_instance_id(filename):
    # This dataset has no patient/session metadata, so each image is its own unit.
    # Using the (deduplicated) file stem as the id means grouped CV degrades gracefully
    # into ordinary stratified CV (every group has exactly one member) without needing
    # a separate code path downstream.
    stem = os.path.basename(filename)
    for ext in CONFIG.IMAGE_EXTS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem


def _subdirs(path):
    return [d for d in sorted(os.listdir(path))
            if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")]


def discover_dataset(cfg):
    root = cfg.DATA_ROOT
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

    all_files = find_image_files(root)
    print(f"Found {len(all_files)} image files under {root}")
    if len(all_files) == 0:
        raise FileNotFoundError(f"No {cfg.IMAGE_EXTS} files anywhere under {root}")

    all_files = dedup_by_content(all_files)

    # --- Strategy 1: group by immediate parent folder name (handles class subfolders
    # at any depth, e.g. DATA_ROOT/yes/*.jpg or DATA_ROOT/wrapper/yes/*.jpg) ---
    parent_groups = defaultdict(list)
    for f in all_files:
        parent_dir = normalise_label(os.path.basename(os.path.dirname(f)))
        parent_groups[parent_dir].append(f)

    plausible = {
        k: v for k, v in parent_groups.items()
        if k.lower() not in cfg.NON_CLASS_DIR_NAMES
    }

    if 2 <= len(plausible) <= 10:
        print("Detected CLASS-SUBFOLDER layout:", {k: len(v) for k, v in plausible.items()})
        records = []
        for label, files in plausible.items():
            for f in files:
                records.append({"filepath": f, "label": label, "subject_id": extract_instance_id(f)})
        return pd.DataFrame(records), "class_subfolders"

    # --- Strategy 2: flat layout plus a label file ---
    label_file = None
    for cand in cfg.LABEL_FILE_CANDIDATES:
        hits = glob.glob(os.path.join(root, "**", cand), recursive=True)
        if hits:
            label_file = hits[0]
            break

    if label_file is None:
        tabular = sorted(
            glob.glob(os.path.join(root, "**", "*.csv"), recursive=True) +
            glob.glob(os.path.join(root, "**", "*.xlsx"), recursive=True)
        )
        if tabular:
            label_file = tabular[0]

    if label_file is None:
        print("\nDirectory tree under DATA_ROOT (for debugging):")
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if depth > 3:
                dirnames[:] = []
                continue
            indent = "  " * depth
            print(f"{indent}{os.path.basename(dirpath) or dirpath}/ ({len(filenames)} files)")
        raise ValueError(
            "Could not auto-detect the layout. Expected class subfolders "
            "(DATA_ROOT/yes/*.jpg) or a flat folder plus a label CSV/XLSX. "
            f"Top-level entries under {root}: {sorted(os.listdir(root))[:20]}"
        )

    print("Detected FLAT layout with label file:", label_file)
    lbl_df = pd.read_csv(label_file) if label_file.lower().endswith(".csv") else pd.read_excel(label_file)
    sid_col = next((c for c in cfg.SUBJECT_ID_COLUMN_CANDIDATES if c in lbl_df.columns), None)
    lab_col = next((c for c in cfg.LABEL_COLUMN_CANDIDATES if c in lbl_df.columns), None)

    if sid_col is None or lab_col is None:
        raise ValueError(
            f"Could not find id/label columns in {label_file}. "
            f"Columns present: {list(lbl_df.columns)}. Add the right names to "
            "CONFIG.SUBJECT_ID_COLUMN_CANDIDATES / CONFIG.LABEL_COLUMN_CANDIDATES."
        )

    lbl_df = lbl_df[[sid_col, lab_col]].rename(columns={sid_col: "subject_id", lab_col: "label"})
    lbl_df["subject_id"] = lbl_df["subject_id"].astype(str).str.strip()
    lbl_df["label"] = lbl_df["label"].map(normalise_label)
    lbl_df = lbl_df.dropna().drop_duplicates()

    file_df = pd.DataFrame([{"filepath": f, "subject_id": extract_instance_id(f)} for f in all_files])
    df = file_df.merge(lbl_df, on="subject_id", how="inner")

    if len(df) == 0:
        raise ValueError(
            "Merged 0 rows: ids in the label file do not match the filenames. "
            f"Example file id: {file_df['subject_id'].iloc[0]!r}; "
            f"example label id: {lbl_df['subject_id'].iloc[0]!r}. Adjust extract_instance_id."
        )

    print(f"Matched {len(df)}/{len(all_files)} files to labels.")
    return df, "flat_with_labels"


df, layout = discover_dataset(CONFIG)
df["label"] = df["label"].map(normalise_label)
df = df.drop_duplicates(subset="filepath").reset_index(drop=True)

print("\nLayout:", layout)
print("\nImages per class:")
print(df["label"].value_counts().to_string())
print("\nUnique subjects:", df["subject_id"].nunique(), "| total images:", len(df))
scans_per_subject = df.groupby("subject_id").size()
print("Images per subject: min={}, median={}, max={}".format(
    scans_per_subject.min(), int(scans_per_subject.median()), scans_per_subject.max()))
if scans_per_subject.max() > 1:
    print("Repeated images present, so subject-level splitting is required. Handled below.")
else:
    print("No patient/session metadata exists for this dataset, so every image is treated as its "
          "own subject. Grouped CV below therefore behaves like ordinary stratified CV.")
df.head()

# CELL 5
# 1. Ensure 'df' exists and has sufficient classes
if "df" not in globals():
    raise NameError("DataFrame 'df' is not defined. Re-run the dataset discovery cell above first.")

if df["label"].nunique() < 2:
    raise ValueError(f"Only one class found ({df['label'].unique()}). Check the dataset layout.")

# 2. Encode labels
classes = sorted(df["label"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
df["y"] = df["label"].map(class_to_idx).astype(int)
NUM_CLASSES = len(classes)

print("Classes:", classes)

# 3. Check for multi-label subject conflicts (converters or extraction artifacts)
multi = df.groupby("subject_id")["label"].nunique()
multi = multi[multi > 1]

if len(multi) > 0:
    print(f"\nWARNING: {len(multi)} subject(s) appear under more than one label:")
    for sid in list(multi.index)[:10]:
        print("  ", sid, sorted(df.loc[df.subject_id == sid, "label"].unique()))
    print("  The majority label per subject should be used for stratification. Decide explicitly whether "
          "these are genuine converters or an ID-parsing artifact before publishing.")

# 4. Check subject counts and class distribution imbalance
subj_counts = df.groupby("label")["subject_id"].nunique()
print("\nSubjects per class:")
print(subj_counts.to_string())

for c, n in subj_counts.items():
    if n < 15:
        print(f"WARNING: class '{c}' has only {n} unique subjects. Report macro-F1 and AUC alongside "
              "accuracy and expect wide fold-to-fold variance.")

imbalance = subj_counts.max() / max(subj_counts.min(), 1)
if imbalance > 3:
    print(f"\nWARNING: class imbalance ratio is {imbalance:.1f}:1. Class-weighted loss is enabled, but "
          "accuracy alone will be misleading.")

# CELL 6
def load_and_preprocess(path, target_shape=None):
    target_shape = tuple(target_shape or CONFIG.TARGET_SHAPE)   # (H, W)
    img = Image.open(path).convert("L")                          # grayscale
    if img.size[::-1] != target_shape:                            # PIL size is (W, H)
        img = img.resize((target_shape[1], target_shape[0]), resample=Image.BILINEAR)
    vol = np.asarray(img, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise over foreground pixels only, so a large dark background does not
    # dominate the statistics.
    thr = np.percentile(vol, 10)
    mask = vol > thr
    if mask.sum() < 100:
        mask = np.ones_like(vol, dtype=bool)
    mu = float(vol[mask].mean())
    sd = float(vol[mask].std()) + 1e-8
    vol = (vol - mu) / sd
    vol = np.clip(vol, -CONFIG.CLIP_STD, CONFIG.CLIP_STD)
    return np.ascontiguousarray(vol, dtype=np.float32)


def random_flip(vol):
    # Only the left-right axis. MRI images have no meaningful top-bottom symmetry.
    if CONFIG.FLIP_AXIS is not None and random.random() < 0.5:
        vol = np.flip(vol, axis=CONFIG.FLIP_AXIS)
    return vol


def random_rotate(vol, max_deg=None):
    max_deg = CONFIG.MAX_ROTATION_DEG if max_deg is None else max_deg
    angle = random.uniform(-max_deg, max_deg)
    return nd_rotate(vol, angle, axes=(0, 1), reshape=False, order=1, mode="nearest")


def random_intensity(vol, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    return vol * random.uniform(*scale_range) + random.uniform(*shift_range)


def random_gaussian_noise(vol, std=0.03):
    if random.random() < 0.3:
        vol = vol + np.random.normal(0, std, vol.shape).astype(np.float32)
    return vol


def train_augment(vol):
    vol = random_flip(vol)
    if random.random() < 0.5:
        vol = random_rotate(vol)
    vol = random_intensity(vol)
    vol = random_gaussian_noise(vol)
    # np.flip returns a negative-strided view and torch.from_numpy rejects those,
    # so force a contiguous copy before leaving this function.
    return np.ascontiguousarray(vol, dtype=np.float32)


class MRIDataset(Dataset):
    def __init__(self, dataframe, train=False, cache=False):
        self.df = dataframe.reset_index(drop=True)
        self.train = train
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        if self.cache and path in self._cache:
            vol = self._cache[path].copy()
        else:
            vol = load_and_preprocess(path)
            if self.cache:
                self._cache[path] = vol.copy()
        if self.train:
            vol = train_augment(vol)
        vol = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)   # (1, H, W)
        return vol, torch.tensor(int(row["y"]), dtype=torch.long)


# Quick check on one real image, so a preprocessing problem surfaces here and not
# thirty minutes into training.
_probe = load_and_preprocess(df["filepath"].iloc[0])
print("Probe image:", _probe.shape, "dtype", _probe.dtype,
      f"mean={_probe.mean():.3f} std={_probe.std():.3f} "
      f"min={_probe.min():.2f} max={_probe.max():.2f}")
assert _probe.shape == tuple(CONFIG.TARGET_SHAPE)
_aug = train_augment(_probe.copy())
assert _aug.flags["C_CONTIGUOUS"], "augmented image must be contiguous for torch.from_numpy"
print("Preprocessing and augmentation OK.")

# CELL 7
# One label per subject (majority vote) for stratification.
subject_labels = (df.groupby("subject_id")["label"]
                    .agg(lambda s: s.value_counts().idxmax())
                    .reset_index())
min_subjects_per_class = subject_labels["label"].value_counts().min()

n_folds = int(min(CONFIG.N_FOLDS, min_subjects_per_class))
if n_folds < CONFIG.N_FOLDS:
    print(f"WARNING: smallest class has only {min_subjects_per_class} subjects, "
          f"so N_FOLDS is reduced from {CONFIG.N_FOLDS} to {n_folds}.")
if n_folds < 2:
    raise ValueError("Fewer than 2 subjects in the smallest class. Cross-validation is not possible.")
CONFIG.N_FOLDS = n_folds


def inner_split(trainval_df, frac, seed):
    # Hold out a subject-level slice of the training fold for early stopping.
    n_inner = max(2, int(round(1.0 / max(frac, 1e-6))))
    per_class_subjects = trainval_df.groupby("y")["subject_id"].nunique().min()
    n_inner = int(min(n_inner, per_class_subjects))
    if n_inner < 2:
        print("  NOTE: too few subjects for an inner validation split; "
              "early stopping will use the test fold for this fold only.")
        return trainval_df.reset_index(drop=True), None
    sp = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(iter(sp.split(trainval_df, trainval_df["y"],
                                          groups=trainval_df["subject_id"])))
    return (trainval_df.iloc[fit_idx].reset_index(drop=True),
            trainval_df.iloc[val_idx].reset_index(drop=True))


sgkf = StratifiedGroupKFold(n_splits=CONFIG.N_FOLDS, shuffle=True, random_state=SEED)
folds = []
for fold_i, (tr_idx, te_idx) in enumerate(sgkf.split(df, df["y"], groups=df["subject_id"])):
    trainval_df = df.iloc[tr_idx].reset_index(drop=True)
    test_df = df.iloc[te_idx].reset_index(drop=True)
    fit_df, val_df = inner_split(trainval_df, CONFIG.INNER_VAL_FRACTION, SEED + fold_i)
    if val_df is None:
        val_df = test_df   # degenerate fallback, flagged above

    s_fit, s_val, s_test = (set(fit_df.subject_id), set(val_df.subject_id), set(test_df.subject_id))
    assert s_fit.isdisjoint(s_test), "subject leakage between fit and test"
    if val_df is not test_df:
        assert s_fit.isdisjoint(s_val), "subject leakage between fit and inner val"
        assert s_val.isdisjoint(s_test), "subject leakage between inner val and test"

    folds.append({"fit": fit_df, "val": val_df, "test": test_df})
    print(f"Fold {fold_i}: fit={len(fit_df):4d} scans/{len(s_fit):3d} subj | "
          f"inner-val={len(val_df):4d}/{len(s_val):3d} | test={len(test_df):4d}/{len(s_test):3d} | "
          f"test class counts={dict(test_df['label'].value_counts())}")

missing = [i for i, f in enumerate(folds) if f["test"]["y"].nunique() < NUM_CLASSES]
if missing:
    print(f"WARNING: folds {missing} do not contain every class in the test split. "
          "AUC for those folds will be reported as NaN and skipped in the mean.")

# CELL 8
def norm_layer(channels):
    # GroupNorm rather than BatchNorm2d: fold sizes here are small, and BatchNorm
    # statistics are unreliable at small batch sizes.
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def act_layer():
    # Out-of-place on purpose. Inplace ReLU inside a residual block breaks
    # register_full_backward_hook, which Grad-CAM relies on.
    return nn.ReLU(inplace=False)


class ChannelAttention2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            act_layer(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM2D(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention2D(channels, reduction)
        self.sa = SpatialAttention2D(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class MultiScaleResBlock2D(nn.Module):
    # Parallel dilated 3x3 convolutions (rates 1, 2, 3) fused by a 1x1 convolution,
    # with a residual connection and optional CBAM2D attention. Padding equals the
    # dilation rate so all three branches return identical spatial sizes.
    def __init__(self, in_ch, out_ch, stride=1, use_attention=True):
        super().__init__()
        if out_ch < 3:
            raise ValueError("out_ch must be at least 3 for the three-branch block")
        branch_ch = out_ch // 3
        rem = out_ch - branch_ch * 3
        self.b1 = nn.Conv2d(in_ch, branch_ch, 3, stride=stride, padding=1, dilation=1, bias=False)
        self.b2 = nn.Conv2d(in_ch, branch_ch, 3, stride=stride, padding=2, dilation=2, bias=False)
        self.b3 = nn.Conv2d(in_ch, branch_ch + rem, 3, stride=stride, padding=3, dilation=3, bias=False)
        self.bn1 = norm_layer(out_ch)
        self.act = act_layer()
        self.fuse = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn2 = norm_layer(out_ch)

        self.use_attention = use_attention
        self.attn = CBAM2D(out_ch) if use_attention else None

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                norm_layer(out_ch),
            )

    def forward(self, x):
        identity = x
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.act(self.bn1(out))
        out = self.bn2(self.fuse(out))
        if self.attn is not None:
            out = self.attn(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(out + identity)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class MSCANet2D(nn.Module):
    # Proposed 2D multi-scale channel-spatial attention network.
    def __init__(self, num_classes, in_channels=None, base=None, use_attention=True, dropout=None):
        super().__init__()
        in_channels = CONFIG.IN_CHANNELS if in_channels is None else in_channels
        base = CONFIG.BASE_CHANNELS if base is None else base
        dropout = CONFIG.DROPOUT if dropout is None else dropout
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, stride=1, padding=1, bias=False),
            norm_layer(base), act_layer(), nn.MaxPool2d(2),
        )
        chs = [base, base * 2, base * 4, base * 8]
        self.stage1 = MultiScaleResBlock2D(chs[0], chs[1], stride=2, use_attention=use_attention)
        self.stage2 = MultiScaleResBlock2D(chs[1], chs[2], stride=2, use_attention=use_attention)
        self.stage3 = MultiScaleResBlock2D(chs[2], chs[3], stride=2, use_attention=use_attention)
        self.stage4 = MultiScaleResBlock2D(chs[3], chs[3], stride=2, use_attention=use_attention)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(chs[3], 128), act_layer(),
            nn.Dropout(dropout / 2), nn.Linear(128, num_classes),
        )
        self.target_layer_name = "stage4"     # Grad-CAM hook point
        init_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        return self.classifier(self.gap(x))


class Plain2DCNN(nn.Module):
    # Baseline 1: no attention, no multi-scale branches, no residual connections.
    def __init__(self, num_classes, in_channels=None, base=None, dropout=None):
        super().__init__()
        in_channels = CONFIG.IN_CHANNELS if in_channels is None else in_channels
        base = CONFIG.BASE_CHANNELS if base is None else base
        dropout = CONFIG.DROPOUT if dropout is None else dropout

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 norm_layer(cout), act_layer(), nn.MaxPool2d(2))

        self.features = nn.Sequential(
            block(in_channels, base), block(base, base * 2), block(base * 2, base * 4),
            block(base * 4, base * 8), block(base * 8, base * 8),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(base * 8, 128), act_layer(), nn.Linear(128, num_classes),
        )
        self.target_layer_name = "features"
        init_weights(self)

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = norm_layer(out_ch)
        self.act = act_layer()
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                          norm_layer(out_ch))

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(out + identity)


class ResNet2D18(nn.Module):
    # Baseline 2: 2D ResNet-18 with GroupNorm, trained from scratch under the same protocol.
    def __init__(self, num_classes, in_channels=None, base=None):
        super().__init__()
        in_channels = CONFIG.IN_CHANNELS if in_channels is None else in_channels
        base = CONFIG.BASE_CHANNELS if base is None else base
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 7, stride=2, padding=3, bias=False),
            norm_layer(base), act_layer(), nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock2D(base, base), BasicBlock2D(base, base))
        self.layer2 = nn.Sequential(BasicBlock2D(base, base * 2, stride=2),
                                    BasicBlock2D(base * 2, base * 2))
        self.layer3 = nn.Sequential(BasicBlock2D(base * 2, base * 4, stride=2),
                                    BasicBlock2D(base * 4, base * 4))
        self.layer4 = nn.Sequential(BasicBlock2D(base * 4, base * 8, stride=2),
                                    BasicBlock2D(base * 8, base * 8))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(base * 8, num_classes))
        self.target_layer_name = "layer4"
        init_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.classifier(self.gap(x))


def build_model(name, num_classes):
    if name == "MSCANet2D":
        return MSCANet2D(num_classes, use_attention=True)
    if name == "MSCANet2D_NoAttention":
        return MSCANet2D(num_classes, use_attention=False)
    if name == "Plain2DCNN":
        return Plain2DCNN(num_classes)
    if name == "ResNet2D18":
        return ResNet2D18(num_classes)
    raise ValueError(f"Unknown model name: {name}")


# Shape check on a dummy batch. A reduced probe size is used because every stage
# divides the input by 2 and the check is scale invariant, so this costs a second
# rather than a minute of CPU time.
_probe_shape = tuple(max(32, min(64, s)) for s in CONFIG.TARGET_SHAPE)
_dummy = torch.zeros(2, CONFIG.IN_CHANNELS, *_probe_shape)
print("Probe input:", tuple(_dummy.shape))
for _name in ["MSCANet2D", "MSCANet2D_NoAttention", "Plain2DCNN", "ResNet2D18"]:
    _m = build_model(_name, NUM_CLASSES).eval()
    with torch.no_grad():
        _out = _m(_dummy)
    assert _out.shape == (2, NUM_CLASSES), f"{_name} returned {_out.shape}"
    print(f"{_name:24s} output {tuple(_out.shape)}  params={sum(p.numel() for p in _m.parameters()):,}")
    del _m
del _dummy
print("All architectures verified.")

# CELL 9
class EarlyStopping:
    def __init__(self, patience=10, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.should_stop = False

    def step(self, metric):
        if metric is None or (isinstance(metric, float) and math.isnan(metric)):
            self.count += 1
            self.should_stop = self.count >= self.patience
            return False
        improved = (self.best is None or
                    (self.mode == "max" and metric > self.best + self.min_delta) or
                    (self.mode == "min" and metric < self.best - self.min_delta))
        if improved:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
        return improved


def compute_class_weights(labels, num_classes):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(frame, train, seed_offset=0):
    ds = MRIDataset(frame, train=train, cache=CONFIG.CACHE_VOLUMES)
    if len(ds) == 0:
        raise ValueError("Empty dataset passed to make_loader")
    g = torch.Generator()
    g.manual_seed(SEED + seed_offset)
    return DataLoader(
        ds,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=train,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        # drop_last would empty the loader when a fold is smaller than one batch.
        drop_last=(train and len(ds) > CONFIG.BATCH_SIZE),
        worker_init_fn=seed_worker,
        generator=g,
    )


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    use_amp = bool(CONFIG.USE_AMP and DEVICE.type == "cuda")
    total_loss, n_seen = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.set_grad_enabled(train):
        for vols, labels in loader:
            vols = vols.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with amp_autocast(use_amp):
                logits = model(vols)
                loss = criterion(logits, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)          # unscale before clipping, or the clip is wrong
                    nn.utils.clip_grad_norm_(model.parameters(), CONFIG.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CONFIG.GRAD_CLIP)
                    optimizer.step()

            bs = labels.size(0)
            n_seen += bs
            total_loss += float(loss.detach()) * bs
            probs = F.softmax(logits.detach().float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if n_seen == 0:
        raise RuntimeError("Dataloader yielded no batches. Check the split sizes and BATCH_SIZE.")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                       average="macro", zero_division=0)
    return {
        "loss": total_loss / n_seen,          # divide by samples seen, not dataset size
        "acc": accuracy_score(all_labels, all_preds),
        "precision": prec, "recall": rec, "f1": f1,
        "labels": all_labels, "preds": all_preds, "probs": all_probs,
    }


def multiclass_auc(labels, probs, num_classes):
    try:
        if num_classes == 2:
            if len(np.unique(labels)) < 2:
                return float("nan")
            return roc_auc_score(labels, probs[:, 1])
        present = np.unique(labels)
        if len(present) < num_classes:
            return float("nan")
        y_bin = label_binarize(labels, classes=list(range(num_classes)))
        return roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except ValueError:
        return float("nan")


def train_model(model_name, fold, num_classes, fold_idx, tag="CV"):
    seed_everything(SEED + fold_idx)          # identical starting point per fold for every model
    model = build_model(model_name, num_classes).to(DEVICE)

    fit_loader = make_loader(fold["fit"], train=True, seed_offset=fold_idx)
    val_loader = make_loader(fold["val"], train=False)
    test_loader = make_loader(fold["test"], train=False)

    class_weights = compute_class_weights(fold["fit"]["y"].values, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CONFIG.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    scaler = make_grad_scaler(bool(CONFIG.USE_AMP and DEVICE.type == "cuda"))

    stopper = EarlyStopping(patience=CONFIG.EARLY_STOP_PATIENCE, mode="max")
    history = defaultdict(list)
    best_state, best_f1, best_epoch = None, -1.0, -1

    for epoch in range(CONFIG.EPOCHS):
        t0 = time.time()
        tr = run_epoch(model, fit_loader, criterion, optimizer, scaler)
        va = run_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr["loss"]); history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["acc"]); history["val_acc"].append(va["acc"])
        history["train_f1"].append(tr["f1"]); history["val_f1"].append(va["f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if stopper.step(va["f1"]):
            best_f1, best_epoch = va["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{tag} fold{fold_idx} {model_name}] epoch {epoch + 1}/{CONFIG.EPOCHS} "
              f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
              f"val_acc={va['acc']:.4f} val_f1={va['f1']:.4f} ({time.time() - t0:.1f}s)")

        if stopper.should_stop:
            print(f"  early stop at epoch {epoch + 1} (best inner-val F1={best_f1:.4f} "
                  f"at epoch {best_epoch + 1})")
            break

    if best_state is None:                     # never improved, keep the final weights
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = CONFIG.EPOCHS - 1
    model.load_state_dict(best_state)

    # The test fold is evaluated once, here, with the selected weights.
    test_metrics = run_epoch(model, test_loader, criterion)
    test_metrics["auc"] = multiclass_auc(test_metrics["labels"], test_metrics["probs"], num_classes)
    test_metrics["fold"] = fold_idx
    test_metrics["best_epoch"] = best_epoch + 1
    test_metrics["best_inner_val_f1"] = best_f1

    ckpt_path = os.path.join(CONFIG.CKPT_DIR, f"{model_name}_fold{fold_idx}.pt")
    torch.save(best_state, ckpt_path)

    print(f"  -> TEST fold {fold_idx} {model_name}: acc={test_metrics['acc']:.4f} "
          f"f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history, test_metrics, ckpt_path


print("Training utilities defined.")

# CELL 10
models_to_run = ["MSCANet2D"]
if CONFIG.RUN_ABLATION:
    models_to_run.append("MSCANet2D_NoAttention")
if CONFIG.RUN_BASELINES:
    models_to_run += ["Plain2DCNN", "ResNet2D18"]

all_results = defaultdict(list)     # model -> per-fold test metric dicts
all_histories = defaultdict(list)
all_ckpts = defaultdict(list)

run_t0 = time.time()
for fold_idx, fold in enumerate(folds):
    for model_name in models_to_run:
        try:
            history, test_metrics, ckpt = train_model(model_name, fold, NUM_CLASSES, fold_idx)
        except RuntimeError as err:
            # torch.cuda.OutOfMemoryError only exists on newer versions, and it subclasses
            # RuntimeError, so match on the message instead of the class.
            if "out of memory" not in str(err).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA out of memory while training {model_name}. Lower CONFIG.BATCH_SIZE, "
                "lower CONFIG.BASE_CHANNELS, or reduce CONFIG.TARGET_SHAPE to (96, 96)."
            ) from err
        all_results[model_name].append(test_metrics)
        all_histories[model_name].append(history)
        all_ckpts[model_name].append(ckpt)

print(f"\nCross-validation complete in {(time.time() - run_t0) / 60:.1f} min.")

per_fold_rows = [{"model": m, "fold": r["fold"], "acc": r["acc"], "f1": r["f1"],
                  "precision": r["precision"], "recall": r["recall"], "auc": r["auc"],
                  "best_epoch": r["best_epoch"]}
                 for m in models_to_run for r in all_results[m]]
per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(os.path.join(CONFIG.RESULTS_DIR, "per_fold_metrics.csv"), index=False)
print(per_fold_df.round(4).to_string(index=False))

# CELL 11
def summarize_model(model_name, results):
    accs = [r["acc"] for r in results]
    f1s = [r["f1"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]
    aucs = [r["auc"] for r in results]
    n = len(accs)
    return {
        "model": model_name,
        "acc_mean": np.mean(accs), "acc_std": np.std(accs, ddof=1) if n > 1 else 0.0,
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s, ddof=1) if n > 1 else 0.0,
        "precision_mean": np.mean(precs), "precision_std": np.std(precs, ddof=1) if n > 1 else 0.0,
        "recall_mean": np.mean(recs), "recall_std": np.std(recs, ddof=1) if n > 1 else 0.0,
        "auc_mean": np.nanmean(aucs), "auc_std": np.nanstd(aucs, ddof=1) if n > 1 else 0.0,
        "fold_accs": accs, "fold_f1s": f1s, "fold_aucs": aucs,
    }


summary_df = pd.DataFrame([summarize_model(m, all_results[m]) for m in models_to_run])

proposed = "MSCANet2D"
best_fold_idx = int(np.argmax(summary_df.loc[summary_df.model == proposed, "fold_accs"].values[0]))

report_cols = ["model", "acc_mean", "acc_std", "f1_mean", "f1_std", "precision_mean", "precision_std",
               "recall_mean", "recall_std", "auc_mean", "auc_std"]
paper_table = summary_df[report_cols].copy()
for c in report_cols[1:]:
    paper_table[c] = paper_table[c].astype(float).round(4)

# Publication-ready mean +/- std strings, sample size uses ddof=1 across folds.
pretty = pd.DataFrame({"Model": paper_table["model"]})
for metric in ["acc", "f1", "precision", "recall", "auc"]:
    pretty[metric.upper()] = [f"{m:.4f} ± {s:.4f}" for m, s in
                              zip(paper_table[f"{metric}_mean"], paper_table[f"{metric}_std"])]

paper_table.to_csv(os.path.join(CONFIG.RESULTS_DIR, "model_comparison_summary.csv"), index=False)
pretty.to_csv(os.path.join(CONFIG.RESULTS_DIR, "model_comparison_pretty.csv"), index=False)

print(f"Held-out test metrics over {CONFIG.N_FOLDS} folds (mean ± std, ddof=1)\n")
print(pretty.to_string(index=False))
print("\nBest fold for the proposed model:", best_fold_idx)

# CELL 12
def holm_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    valid = ~np.isnan(pvals)
    idx = np.where(valid)[0]
    order = idx[np.argsort(pvals[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)     # enforce monotonicity
        out[i] = running
    return out


acc_proposed = summary_df.loc[summary_df.model == proposed, "fold_accs"].values[0]
sig_rows = []
for m in models_to_run:
    if m == proposed:
        continue
    acc_b = summary_df.loc[summary_df.model == m, "fold_accs"].values[0]
    diffs = np.asarray(acc_proposed) - np.asarray(acc_b)
    note = ""

    if np.allclose(diffs, 0):
        # Identical fold accuracies make both tests undefined; the original code returned a
        # bare NaN with no explanation.
        t_stat = t_p = w_stat = w_p = float("nan")
        note = "identical per-fold accuracies, tests undefined"
    else:
        t_stat, t_p = ttest_rel(acc_proposed, acc_b)
        try:
            w_stat, w_p = wilcoxon(acc_proposed, acc_b, zero_method="wilcox")
        except ValueError as e:
            w_stat, w_p = float("nan"), float("nan")
            note = f"wilcoxon undefined ({e})"

    n = len(diffs)
    sem = diffs.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    sig_rows.append({
        "comparison": f"{proposed} vs {m}",
        "mean_acc_gap": diffs.mean(),
        "gap_95ci_low": diffs.mean() - 1.96 * sem,
        "gap_95ci_high": diffs.mean() + 1.96 * sem,
        "paired_t_stat": t_stat, "paired_t_p": t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "note": note,
    })

sig_df = pd.DataFrame(sig_rows)
if len(sig_df):
    sig_df["paired_t_p_holm"] = holm_correction(sig_df["paired_t_p"].values)
    sig_df["wilcoxon_p_holm"] = holm_correction(sig_df["wilcoxon_p"].values)
sig_df.to_csv(os.path.join(CONFIG.RESULTS_DIR, "significance_tests.csv"), index=False)
print(sig_df.round(4).to_string(index=False))

if len(folds) < 5:
    print("\nNOTE: fewer than 5 folds. Treat these p-values as descriptive only.")

# CELL 13
sns.set_style("whitegrid")

def savefig(fig, name):
    path = os.path.join(CONFIG.FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


# --- training curves, best fold of the proposed model ---
hist = all_histories[proposed][best_fold_idx]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist["train_loss"], label="fit")
axes[0].plot(hist["val_loss"], label="inner val")
axes[0].set_title(f"{proposed} loss (fold {best_fold_idx})")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
axes[1].plot(hist["train_acc"], label="fit")
axes[1].plot(hist["val_acc"], label="inner val")
axes[1].set_title(f"{proposed} accuracy (fold {best_fold_idx})")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
fig.tight_layout()
savefig(fig, "training_curves.png")

# --- pooled confusion matrix over the held-out test folds ---
labels_pooled = np.concatenate([r["labels"] for r in all_results[proposed]])
preds_pooled = np.concatenate([r["preds"] for r in all_results[proposed]])
probs_pooled = np.concatenate([r["probs"] for r in all_results[proposed]])

cm = confusion_matrix(labels_pooled, preds_pooled, labels=list(range(NUM_CLASSES)))
cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=classes, yticklabels=classes, ax=axes[0])
axes[0].set_title("counts"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=classes, yticklabels=classes, ax=axes[1])
axes[1].set_title("row-normalised (recall)"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
fig.suptitle(f"{proposed}: pooled held-out confusion matrix")
fig.tight_layout()
savefig(fig, "confusion_matrix.png")

# --- ROC curves, pooled over test folds ---
fig, ax = plt.subplots(figsize=(5.5, 5))
if NUM_CLASSES == 2:
    fpr, tpr, _ = roc_curve(labels_pooled, probs_pooled[:, 1])
    ax.plot(fpr, tpr, label=f"{classes[1]} vs {classes[0]} (AUC={auc(fpr, tpr):.3f})")
else:
    y_bin = label_binarize(labels_pooled, classes=list(range(NUM_CLASSES)))
    for i, c in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_pooled[:, i])
        ax.plot(fpr, tpr, label=f"{c} vs rest (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"{proposed}: pooled held-out ROC"); ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
savefig(fig, "roc_curves.png")

# --- per-fold accuracy by model ---
plot_df = pd.DataFrame([{"model": m, "accuracy": a}
                        for m in models_to_run
                        for a in summary_df.loc[summary_df.model == m, "fold_accs"].values[0]])
fig, ax = plt.subplots(figsize=(7.5, 4))
sns.boxplot(data=plot_df, x="model", y="accuracy", ax=ax, whis=(0, 100), width=0.5)
sns.stripplot(data=plot_df, x="model", y="accuracy", color="black", alpha=0.7, ax=ax)
ax.set_title("Per-fold held-out accuracy by model")
ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
savefig(fig, "fold_accuracy_comparison.png")

# CELL 14
class GradCAM2D:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        modules = dict(model.named_modules())
        if target_layer_name not in modules:
            raise KeyError(f"Layer '{target_layer_name}' not found. "
                           f"Available: {[k for k in modules if k][:20]}")
        layer = modules[target_layer_name]
        self.handles = [layer.register_forward_hook(self._forward_hook),
                        layer.register_full_backward_hook(self._backward_hook)]

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __call__(self, img_tensor, class_idx=None):
        x = img_tensor.unsqueeze(0).to(DEVICE)
        with torch.enable_grad():
            logits = self.model(x)
            pred_idx = int(logits.argmax(1).item())     # the real prediction, kept separate
            target = pred_idx if class_idx is None else int(class_idx)
            self.model.zero_grad(set_to_none=True)
            logits[0, target].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks captured nothing. The target layer may not be on the "
                               "forward path, or an inplace activation broke the backward hook.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        rng = cam.max() - cam.min()
        cam = (cam - cam.min()) / rng if rng > 1e-8 else np.zeros_like(cam)   # avoid /0 on a flat CAM
        probs = F.softmax(logits.detach().float(), dim=1).cpu().numpy()[0]
        return cam, pred_idx, probs


def plot_gradcam_overlay(img, cam, true_label, pred_label, prob, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input", fontsize=10); axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM", fontsize=10); axes[1].axis("off")
    axes[2].imshow(img, cmap="gray")
    im = axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Overlay", fontsize=10); axes[2].axis("off")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Grad-CAM")
    fig.suptitle(f"True: {idx_to_class[true_label]} | Pred: {idx_to_class[pred_label]} "
                 f"(p={prob:.2f})")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


best_ckpt = all_ckpts[proposed][best_fold_idx]
gc_model = build_model(proposed, NUM_CLASSES).to(DEVICE)
gc_model.load_state_dict(safe_load_state_dict(best_ckpt, DEVICE))
gradcam = GradCAM2D(gc_model, gc_model.target_layer_name)

test_df_best = folds[best_fold_idx]["test"]
test_ds_best = MRIDataset(test_df_best, train=False)

EXAMPLES_PER_CLASS = 2
shown = Counter()
saved_paths = []
try:
    for i in range(len(test_ds_best)):
        img, y = test_ds_best[i]
        y = int(y.item())
        if shown[y] >= EXAMPLES_PER_CLASS:
            continue
        cam, pred_idx, probs = gradcam(img)          # class_idx=None, so pred_idx is genuine
        if pred_idx != y:
            continue                                  # correctly classified examples only
        shown[y] += 1
        path = os.path.join(CONFIG.FIGURES_DIR, f"gradcam_{idx_to_class[y]}_{i}.png")
        plot_gradcam_overlay(img.squeeze(0).numpy(), cam, y, pred_idx, float(probs[pred_idx]), path)
        saved_paths.append(path)
        if all(shown[c] >= EXAMPLES_PER_CLASS for c in range(NUM_CLASSES)):
            break
finally:
    gradcam.close()      # hooks stay attached to the module otherwise

for c in range(NUM_CLASSES):
    if shown[c] == 0:
        print(f"NOTE: no correctly classified test example for class '{idx_to_class[c]}' in fold "
              f"{best_fold_idx}, so no Grad-CAM figure was produced for it.")
print(f"Saved {len(saved_paths)} Grad-CAM figures to {CONFIG.FIGURES_DIR}")

# CELL 15
rows = []
for m in models_to_run:
    model = build_model(m, NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    row = {"model": m, "n_params": n_params, "n_params_M": round(n_params / 1e6, 3),
           "n_trainable_M": round(n_train / 1e6, 3)}
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(torch.zeros(1, CONFIG.IN_CHANNELS, *CONFIG.TARGET_SHAPE),), verbose=False)
        row["gflops"] = round(flops / 1e9, 2)
    except Exception:
        row["gflops"] = None
    rows.append(row)
    del model

eff_df = pd.DataFrame(rows)
eff_df.to_csv(os.path.join(CONFIG.RESULTS_DIR, "model_efficiency.csv"), index=False)
if eff_df["gflops"].isna().all():
    print("thop not installed, so FLOPs are omitted. pip install thop to add them.")
print(eff_df.to_string(index=False))

# CELL 16
bundle = {
    "env": ENV_INFO,
    "config": {k: str(v) for k, v in CONFIG.as_dict().items()},
    "classes": classes,
    "n_scans": int(len(df)),
    "n_subjects": int(df["subject_id"].nunique()),
    "n_folds": int(CONFIG.N_FOLDS),
    "summary": paper_table.to_dict(orient="records"),
    "per_fold": per_fold_df.to_dict(orient="records"),
    "significance": sig_df.replace({np.nan: None}).to_dict(orient="records"),
    "efficiency": eff_df.replace({np.nan: None}).to_dict(orient="records"),
    "best_fold": int(best_fold_idx),
}
def json_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

with open(os.path.join(CONFIG.RESULTS_DIR, "results_bundle.json"), "w") as fh:
    json.dump(bundle, fh, indent=2, default=json_native)

print("Run complete. Artifacts in:", CONFIG.OUTPUT_DIR, "\n")
for root, _, files in os.walk(CONFIG.OUTPUT_DIR):
    if "_smoke_data" in root:
        continue
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"{os.path.getsize(p) / 1024:9.1f} KB  {os.path.relpath(p, CONFIG.OUTPUT_DIR)}")