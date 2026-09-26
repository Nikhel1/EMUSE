"""
Smoke test for EMUSE — runs outside of Streamlit.

Verifies that all three data files can be downloaded (if missing) and loaded
correctly. Called directly by the CI smoke-test job:

    python smoke_test.py

Exit 0 = success, exit 1 = failure (with traceback printed to stderr).
"""

import os
import sys
import time

print("── EMUSE smoke test ──")
print(f"Python {sys.version}")
print()

# ---------------------------------------------------------------------------
# 1. Imports — same deps as main.py, minus streamlit
# ---------------------------------------------------------------------------
try:
    import gdown
    import numpy as np
    import open_clip
    import pandas as pd
    import torch
    print("✅ Core imports OK")
except ImportError as exc:
    print(f"❌ Import failed: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. File definitions — keep in sync with main.py
# ---------------------------------------------------------------------------
MODEL_URL   = "https://drive.google.com/uc?id=1k0MNw1hyBDejxOovKwhQCPRmJil13ut5"
MODEL_FILE  = "epoch_99.pt"

FEATURE_URL  = "https://drive.google.com/uc?id=11l-iVak_8QnycuePIvPwXbDBUP_ILP_Y"
FEATURE_FILE = "all_sbid_image_features.pt"

IDX_URL  = "https://drive.google.com/uc?id=1rI1RzKDMMKrOyeE_7BaCNthYrYgYoRf8"
IDX_FILE = "allidx_sbid_ra_dec_flux_catwise.pkl"

# ---------------------------------------------------------------------------
# 3. Download missing files
# ---------------------------------------------------------------------------
def download_if_missing(url: str, dest: str, label: str) -> None:
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  ✅ {label} already present ({size_mb:.0f} MB) — skipping download")
        return
    print(f"  ⬇  Downloading {label} → {dest}")
    t0 = time.time()
    gdown.download(url, dest, quiet=False)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  ✅ {label} downloaded ({size_mb:.0f} MB in {elapsed:.0f}s)")

print("── Step 1: checking / downloading data files ──")
try:
    download_if_missing(MODEL_URL,   MODEL_FILE,   "model checkpoint (epoch_99.pt)")
    download_if_missing(FEATURE_URL, FEATURE_FILE, "image features (all_sbid_image_features.pt)")
    download_if_missing(IDX_URL,     IDX_FILE,     "source index (allidx_sbid_ra_dec_flux_catwise.pkl)")
except Exception as exc:
    print(f"❌ Download failed: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Load model
# ---------------------------------------------------------------------------
print()
print("── Step 2: loading CLIP model ──")
try:
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    checkpoint = torch.load(MODEL_FILE, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print(f"  ✅ Model loaded in {time.time() - t0:.1f}s")
except Exception as exc:
    print(f"❌ Model load failed: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Load image features
# ---------------------------------------------------------------------------
print()
print("── Step 3: loading image features ──")
try:
    t0 = time.time()
    all_image_features = torch.load(FEATURE_FILE)
    shape = tuple(all_image_features.shape) if hasattr(all_image_features, "shape") else "?"
    print(f"  ✅ Features loaded in {time.time() - t0:.1f}s — shape: {shape}")
except Exception as exc:
    print(f"❌ Feature load failed: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Load source index
# ---------------------------------------------------------------------------
print()
print("── Step 4: loading source index ──")
try:
    t0 = time.time()
    idx_dict = pd.read_pickle(IDX_FILE)
    n = len(idx_dict) if hasattr(idx_dict, "__len__") else "?"
    print(f"  ✅ Index loaded in {time.time() - t0:.1f}s — {n} entries")
except Exception as exc:
    print(f"❌ Index load failed: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 7. Quick sanity check — run a text query through the model
# ---------------------------------------------------------------------------
print()
print("── Step 5: sanity check — text query through model ──")
try:
    text_tokens = tokenizer(["a bent-tailed radio galaxy"])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * all_image_features @ text_features.T).squeeze()
    top_idx = int(probs.argmax())
    print(f"  ✅ Query OK — top match index: {top_idx}, score: {probs[top_idx]:.2f}")
except Exception as exc:
    print(f"❌ Query sanity check failed: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print()
print("══ All checks passed — EMUSE smoke test succeeded ══")
sys.exit(0)
