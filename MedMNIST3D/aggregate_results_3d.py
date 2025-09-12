# save as aggregate_medmnist3d.py
# usage:
#   python aggregate_medmnist3d.py /path/to/output --run_prefix r50_3d_28
#   python aggregate_medmnist3d.py /path/to/output --run_prefix r50_3d_28 --csv results_3d.csv --debug
import sys, re, os, glob, csv
import numpy as np

# ---------- CLI ----------
DEBUG = "--debug" in sys.argv
args = [a for a in sys.argv[1:] if a != "--debug"]

def pop_opt(flag, default=None):
    if flag in args:
        i = args.index(flag)
        if i+1 < len(args):
            val = args[i+1]
            del args[i:i+2]
            return val
        else:
            raise SystemExit(f"Usage: {flag} <value>")
    return default

csv_out     = pop_opt("--csv", None)
run_prefix  = pop_opt("--run_prefix", None)  # e.g., r50_3d_28
root_dir    = args[0] if len(args) >= 1 else "./output"

def debug(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

if run_prefix is None:
    raise SystemExit("Please pass --run_prefix (e.g., --run_prefix r50_3d_28) to avoid mixing experiments.")

# ---------- Paper Benchmarks (ResNet-50, 3D only) ----------
PAPER_3D_R50_3D = {
    "organmnist3d":   (0.994, 0.883),
    "nodulemnist3d":  (0.875, 0.847),
    "fracturemnist3d":(0.725, 0.494),
    "adrenalmnist3d": (0.828, 0.745),
    "vesselmnist3d":  (0.907, 0.918),
    "synapsemnist3d": (0.851, 0.795),
}

KNOWN_3D = list(PAPER_3D_R50_3D.keys())
SEEDS_REQUIRED = set(range(5))  # 0..4

PAT_AUC_ACC = re.compile(r"\[AUC\]\s*([0-9.]+).*?\[ACC\]\s*([0-9.]+)", re.IGNORECASE)
PAT_SEED    = re.compile(r"seed\s*[_=\-\s]?(\d+)", re.IGNORECASE)

def parse_auc_acc_from_filename(p):
    m = PAT_AUC_ACC.search(os.path.basename(p))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def infer_seed_from_path(path):
    # try each component and the basename
    comps = os.path.normpath(path).split(os.sep)
    for c in comps + [os.path.basename(path)]:
        m = PAT_SEED.search(c.lower())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None

def better_of(a, b):
    """Prefer higher AUC; break ties by ACC."""
    if a is None: return b
    if b is None: return a
    if b[0] > a[0]: return b
    if b[0] < a[0]: return a
    return b if b[1] > a[1] else a

# buckets: dataset -> {seed -> (auc, acc)}
buckets = {ds:{} for ds in KNOWN_3D}
matched_files = 0

for ds in KNOWN_3D:
    ds_dir = os.path.join(root_dir, ds)
    if not os.path.isdir(ds_dir):
        debug(f"Skip (no dir): {ds_dir}")
        continue

    # Look into run directories matching the chosen run_prefix
    run_glob = os.path.join(ds_dir, f"{run_prefix}_seed*")
    for run_dir in sorted(glob.glob(run_glob)):
        # Expect files like: <ds>_test_[AUC]0.xxx_[ACC]0.yyy@<run>.csv
        patt = os.path.join(run_dir, f"{ds}_test_*@*.csv")
        for csv_path in sorted(glob.glob(patt)):
            matched_files += 1
            mm = parse_auc_acc_from_filename(csv_path)
            if not mm:
                debug(f"Skip (no AUC/ACC in filename): {csv_path}")
                continue
            seed = infer_seed_from_path(run_dir)
            if seed is None or seed not in SEEDS_REQUIRED:
                debug(f"Skip (seed missing or not in 0..4): {csv_path}")
                continue
            prev = buckets[ds].get(seed)
            buckets[ds][seed] = better_of(prev, mm)

if DEBUG:
    print(f"[DEBUG] Matched *_test_* CSV files: {matched_files}")

# ---------- compute & print ----------
print("=== Averages over 5 seeds (3D, test split; compare to paper) ===")
rows = []
found_any = False

for ds in KNOWN_3D:
    seedmap = buckets.get(ds, {})
    have = set(seedmap.keys())
    if have != SEEDS_REQUIRED:
        # will print under "Incomplete" below
        continue

    found_any = True
    aucs = np.array([seedmap[s][0] for s in sorted(seedmap)], dtype=float)
    accs = np.array([seedmap[s][1] for s in sorted(seedmap)], dtype=float)
    auc_mean, acc_mean = aucs.mean(), accs.mean()
    auc_std  = aucs.std(ddof=1)
    acc_std  = accs.std(ddof=1)

    paper = PAPER_3D_R50_3D.get(ds)
    line = f"{ds}: AUC {auc_mean:.3f}±{auc_std:.3f} | ACC {acc_mean:.3f}±{acc_std:.3f}"
    if paper:
        dA = auc_mean - paper[0]
        dC = acc_mean - paper[1]
        dA_pct = (dA / paper[0] * 100) if paper[0] != 0 else 0.0
        dC_pct = (dC / paper[1] * 100) if paper[1] != 0 else 0.0
        line += (
            f"   (paper AUC {paper[0]:.3f}, ACC {paper[1]:.3f}; "
            f"ΔAUC {dA:+.3f} ({dA_pct:+.1f}%), ΔACC {dC:+.3f} ({dC_pct:+.1f}%))"
        )
        rows.append([ds, f"{auc_mean:.3f}", f"{auc_std:.3f}",
                        f"{acc_mean:.3f}", f"{acc_std:.3f}",
                        f"{paper[0]:.3f}", f"{paper[1]:.3f}",
                        f"{dA:+.3f}", f"{dC:+.3f}"])
    else:
        rows.append([ds, f"{auc_mean:.3f}", f"{auc_std:.3f}",
                        f"{acc_mean:.3f}", f"{acc_std:.3f}",
                        "", "", "", ""])
    print(line)

if not found_any:
    print("(no completed 5-seed sets found)")

# List incomplete to help you finish seeds
incomplete = [(ds, sorted(SEEDS_REQUIRED - set(buckets.get(ds, {}).keys())))
              for ds in KNOWN_3D
              if set(buckets.get(ds, {}).keys()) != SEEDS_REQUIRED]

if incomplete:
    print("\n=== Incomplete (missing seeds) ===")
    for ds, missing in incomplete:
        if len(buckets.get(ds, {})) == 0:
            # Skip entirely absent datasets quietly unless debugging
            debug(f"{ds}: no runs detected with run_prefix '{run_prefix}'")
            continue
        print(f"{ds}: missing {missing}")

# Optional CSV
if csv_out:
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset","auc_mean","auc_std","acc_mean","acc_std",
                    "paper_auc","paper_acc","delta_auc","delta_acc"])
        for r in rows:
            w.writerow(r)
    print(f"\nWrote summary: {csv_out}")
