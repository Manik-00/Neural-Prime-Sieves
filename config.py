# config.py
# ─────────────────────────────────────────────────────────────────
# All project-wide constants, device setup, reproducibility
# utilities, matplotlib style, and I/O helpers.
# Every other module imports from here; this file imports nothing
# from the project.
# ─────────────────────────────────────────────────────────────────

import os
import sys
import json
import random
import warnings

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Output Directories ────────────────────────────────────────────
for d in ("paper/figures", "paper/results", "paper/weights"):
    os.makedirs(d, exist_ok=True)

# ── Logger: tee stdout to file ───────────────────────────────────
class Logger:
    """Mirror every print() to both the terminal and a log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log      = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

sys.stdout = Logger("paper/results/terminal_output.txt")

# ── Matplotlib Publication Style ─────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "legend.framealpha":  0.90,
    "lines.linewidth":    1.6,
    "lines.markersize":   5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── Colour Palette (Wong 2011, colour-blind safe) ─────────────────
CB = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "purple": "#CC79A7",
    "red":    "#D55E00",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}
FAM_COLORS  = [CB["blue"], CB["orange"], CB["green"],
               CB["purple"], CB["red"], CB["sky"], CB["yellow"]]
FAM_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# ── Label Names ───────────────────────────────────────────────────
LABEL_NAMES = ["twin", "sophie_germain", "safe", "cousin", "sexy",
               "chen", "isolated"]
LABEL_SHORT = ["Twin", "Sophie G.", "Safe", "Cousin", "Sexy",
               "Chen", "Isolated"]
N_FAM = len(LABEL_NAMES)

# ── Feature Constants ─────────────────────────────────────────────
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
PRIMORIALS   = [6, 30, 210, 2310]

FEATURE_GROUPS_CAUSAL = {
    "A: Primorial residues":   slice(0,  4),
    "B: Small prime residues": slice(4,  16),
    "C: Backward gap":         slice(16, 17),
    "D: Scale features":       slice(17, 20),
    "E: Digit features":       slice(20, 23),
    "F: Extended modular":     slice(23, 25),
}
FEAT_DIM_CAUSAL    = 25
FEAT_DIM_NONCAUSAL = 29

# ── Evaluation Scales ─────────────────────────────────────────────
GEN_SPECS = [
    (500_000_000,            20_000, r"$5\times10^8$",  "5e8",  False),
    (10_000_000_000,         10_000, r"$10^{10}$",       "1e10", True),
    (1_000_000_000_000,      10_000, r"$10^{12}$",       "1e12", True),
    (100_000_000_000_000,     8_000, r"$10^{14}$",       "1e14", True),
    (10_000_000_000_000_000, 15_000, r"$10^{16}$",       "1e16", True),
]
GEN_LABEL = [s[2] for s in GEN_SPECS]
GEN_TAG   = [s[3] for s in GEN_SPECS]
GEN_OOD   = [s[4] for s in GEN_SPECS]

X_TICKS  = np.arange(len(GEN_SPECS))
X_LABELS = [r"$5{\times}10^8$", r"$10^{10}$",
            r"$10^{12}$",       r"$10^{14}$", r"$10^{16}$"]

# ── Seeds ─────────────────────────────────────────────────────────
SEEDS     = [42, 123, 777]
MAIN_SEED = 42

# ── Device ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory/1e9:.1f} GB")

# ── Reproducibility ───────────────────────────────────────────────
def set_seed(s):
    """Pin all randomness sources for full bit-identical reproducibility."""
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

g_torch = torch.Generator()
g_torch.manual_seed(MAIN_SEED)

def _worker_init_fn(worker_id):
    """Deterministic per-worker DataLoader seed."""
    worker_seed = MAIN_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(MAIN_SEED)

# ── I/O Helpers ───────────────────────────────────────────────────
def save_fig(name, fig=None, tight=True):
    """Save figure as PDF + PNG to paper/figures/."""
    f = fig or plt.gcf()
    if tight:
        f.tight_layout()
    f.savefig(f"paper/figures/{name}.pdf")
    f.savefig(f"paper/figures/{name}.png")
    print(f"  Saved → paper/figures/{name}.pdf")
    plt.close(f)

def save_json(name, data):
    """Persist a dict to paper/results/ as pretty JSON."""
    with open(f"paper/results/{name}.json", "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  Saved → paper/results/{name}.json")
