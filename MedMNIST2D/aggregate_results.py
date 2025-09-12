# save as aggregate_medmnist.py
# usage:
#   python aggregate_medmnist.py /path/to/output
#   python aggregate_medmnist.py /path/to/output --csv results.csv --debug
import sys, re, os, glob, csv
import numpy as np

# ---------- CLI ----------
DEBUG = "--debug" in sys.argv
args = [a for a in sys.argv[1:] if a != "--debug"]
csv_out = None
if "--csv" in args:
    i = args.index("--csv")
    if i+1 < len(args):
        csv_out = args[i+1]
        args = args[:i] + args[i+2:]
    else:
        raise SystemExit("Usage: --csv <path>")
ROOT = args[0] if len(args) >= 1 else "./output"

def debug(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ---------- Paper Benchmarks (ResNet-50) ----------
PAPER_2D_R50_28 = {
    "pathmnist": (0.990, 0.911),
    "chestmnist": (0.769, 0.947),
    "dermamnist": (0.913, 0.735),
    "octmnist": (0.952, 0.762),
    "pneumoniamnist": (0.948, 0.854),
    "retinamnist": (0.726, 0.528),
    "breastmnist": (0.857, 0.812),
    "bloodmnist": (0.997, 0.956),
    "tissuemnist": (0.931, 0.680),
    "organamnist": (0.997, 0.935),
    "organcmnist": (0.992, 0.905),
    "organsmnist": (0.972, 0.770),
}
PAPER_2D_R50_224 = {
    "pathmnist": (0.989, 0.892),
    "chestmnist": (0.773, 0.948),
    "dermamnist": (0.912, 0.731),
    "octmnist": (0.958, 0.776),
    "pneumoniamnist": (0.962, 0.884),
    "retinamnist": (0.716, 0.511),
    "breastmnist": (0.866, 0.842),
    "bloodmnist": (0.997, 0.950),
    "tissuemnist": (0.932, 0.680),
    "organamnist": (0.998, 0.947),
    "organcmnist": (0.993, 0.911),
    "organsmnist": (0.975, 0.785),
}
PAPER_3D_R50_3D = {
    "organmnist3d": (0.994, 0.883),
    "nodulemnist3d": (0.875, 0.847),
    "fracturemnist3d": (0.725, 0.494),
    "adrenalmnist3d": (0.828, 0.745),
    "vesselmnist3d": (0.907, 0.918),
    "synapsemnist3d": (0.851, 0.795),
}

SEEDS_REQUIRED = set(range(5))  # 0..4
PAT_AUC_ACC = re.compile(r"\[AUC\]\s*([0-9.]+).*?\[ACC\]\s*([0-9.]+)", re.IGNORECASE)
PAT_VARIANT = re.compile(r"r50_(?:28|224|3d)", re.IGNORECASE)
PAT_SEED    = re.compile(r"seed\s*[_=\-\s]?(\d+)", re.IGNORECASE)

def paper_for(ds: str, var: str):
    if var == "r50_28":  return PAPER_2D_R50_28.get(ds)
    if var == "r50_224": return PAPER_2D_R50_224.get(ds)
    if var == "r50_3d":  return PAPER_3D_R50_3D.get(ds)
    return None

def infer_dataset_from_path(dirpath: str):
    rel = os.path.relpath(dirpath, ROOT)
    parts = rel.split(os.sep)
    return parts[0].lower() if parts and parts[0] not in (".", "") else None

def infer_variant_from_path(dirpath: str, run_tag: str):
    for comp in dirpath.lower().split(os.sep):
        m = PAT_VARIANT.search(comp)
        if m: return m.group(0).lower()
    if run_tag:
        m = PAT_VARIANT.search(run_tag.lower())
        if m: return m.group(0).lower()
    return None

def infer_seed_from_path(dirpath: str, run_tag: str):
    for comp in dirpath.lower().split(os.sep):
        m = PAT_SEED.search(comp)
        if m: return int(m.group(1))
    if run_tag:
        m = PAT_SEED.search(run_tag.lower())
        if m: return int(m.group(1))
    return None

def better_of(a, b):
    if a is None: return b
    if b is None: return a
    if b[0] > a[0]: return b
    if b[0] < a[0]: return a
    return b if b[1] > a[1] else a

# buckets: (dataset, variant) -> {seed -> (auc, acc)}
buckets = {}
matched_files = 0

for dirpath, _, _ in os.walk(ROOT):
    for f in glob.glob(os.path.join(dirpath, "*_test_*.csv")):  # ← fixed pattern
        matched_files += 1
        base = os.path.basename(f)
        m = PAT_AUC_ACC.search(base)
        if not m:
            debug(f"Skip (no AUC/ACC in filename): {f}")
            continue
        auc = float(m.group(1)); acc = float(m.group(2))

        # optional @suffix in filename
        run_tag = ""
        at = base.rfind("@")
        if at != -1:
            run_tag = base[at+1:-4]  # drop ".csv"

        ds = infer_dataset_from_path(dirpath)
        if not ds:
            debug(f"Skip (dataset not inferred from path): {f}")
            continue
        var = infer_variant_from_path(dirpath, run_tag)
        if not var:
            debug(f"Skip (variant not found in path/tag): {f}")
            continue
        seed = infer_seed_from_path(dirpath, run_tag)
        if seed is None or seed not in SEEDS_REQUIRED:
            debug(f"Skip (seed missing or not in 0..4): {f}")
            continue

        key = (ds, var)
        seedmap = buckets.setdefault(key, {})
        seedmap[seed] = better_of(seedmap.get(seed), (auc, acc))

if DEBUG:
    print(f"[DEBUG] Matched *_test_* CSV files: {matched_files}")

# compute averages for completed sets
rows = []
print("=== Averages over 5 seeds (compare to paper) ===")
found_any = False
for (ds, var), seedmap in sorted(buckets.items()):
    have = set(seedmap.keys())
    if have != SEEDS_REQUIRED:
        continue
    found_any = True
    aucs = np.array([seedmap[s][0] for s in sorted(seedmap)], dtype=float)
    accs = np.array([seedmap[s][1] for s in sorted(seedmap)], dtype=float)
    auc_mean, acc_mean = aucs.mean(), accs.mean()
    auc_std  = aucs.std(ddof=1)
    acc_std  = accs.std(ddof=1)
    paper = paper_for(ds, var)

    line = f"{ds}/{var}: AUC {auc_mean:.3f}±{auc_std:.3f} | ACC {acc_mean:.3f}±{acc_std:.3f}"
    if paper:
        dA = auc_mean - paper[0]
        dC = acc_mean - paper[1]
        dA_pct = (dA / paper[0] * 100) if paper[0] != 0 else 0
        dC_pct = (dC / paper[1] * 100) if paper[1] != 0 else 0

        line += (
            f"   (paper AUC {paper[0]:.3f}, ACC {paper[1]:.3f}; "
            f"ΔAUC {dA:+.3f} ({dA_pct:+.1f}%), "
            f"ΔACC {dC:+.3f} ({dC_pct:+.1f}%))"
        )
    else:
        rows.append([ds, var, f"{auc_mean:.3f}", f"{auc_std:.3f}", f"{acc_mean:.3f}", f"{acc_std:.3f}",
                     "", "", "", ""])
    print(line)

if not found_any:
    print("(no completed 5-seed sets found)")

# also list incomplete to help you finish seeds
incomplete = [((ds, var), sorted(SEEDS_REQUIRED - set(seedmap.keys())))
              for (ds, var), seedmap in sorted(buckets.items())
              if set(seedmap.keys()) != SEEDS_REQUIRED]
if incomplete:
    print("\n=== Incomplete (missing seeds) ===")
    for (ds, var), missing in incomplete:
        print(f"{ds}/{var}: missing {missing}")

# optional CSV
if csv_out:
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset","variant","auc_mean","auc_std","acc_mean","acc_std",
                    "paper_auc","paper_acc","delta_auc","delta_acc"])
        for r in rows:
            w.writerow(r)
    print(f"\nWrote summary: {csv_out}")
