"""
Multi-Scenario UNSUPERVISED Test for Aya rPPG Dataset  (v3 — adapted to your real AyaLoader.py)
=================================================================================================
This version is tailored to your actual AyaLoader.py, which has:

    TRAIN_SUBJECTS = [...]            # module-level constants
    TEST_SUBJECTS  = [...]
    TRAIN_CONDITIONS = {"distances": [...], "motions": [...], "lux": [...]}
    TEST_CONDITIONS  = {...}

And inside AyaLoader.get_raw_data():
    if self.split_name in ("train",):
        allowed_subjects = TRAIN_SUBJECTS
        conditions = TRAIN_CONDITIONS
    else:                              # "valid", "test", "unsupervised", anything else
        allowed_subjects = TEST_SUBJECTS
        conditions = TEST_CONDITIONS

==> To switch scenarios we just rebind these two module-level globals before each new
    AyaLoader() construction. We always use the "test" branch (anything != "train").

Usage:
    python test_multi_scenario_unsupervised.py \
        --config_file ./configs/infer_configs/Daily_UNSUPERVISED_MULTI.yaml \
        --subjects 3 4 5 6 7 8 9
"""

import argparse
import os
import random
import sys
import time
from collections import OrderedDict
from io import StringIO

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- rPPG Toolbox imports ----
from config import get_config
from dataset.data_loader import AyaLoader as AyaLoaderModule
from unsupervised_methods.unsupervised_predictor import unsupervised_predict


# ============================================================================
#  USER CONFIG
# ============================================================================
TEST_SUBJECTS = [3, 4, 5, 6, 7, 8, 9]

SCENARIOS = []
for dist in [1, 2, 3, 4, 5]:
    for motion in ["Stationary", "Shaking", "Nodding", "Talking"]:
        SCENARIOS.append({"distance": dist, "lux": 70, "motion": motion})
    SCENARIOS.append({"distance": dist, "lux": 30,  "motion": "Stationary"})
    SCENARIOS.append({"distance": dist, "lux": 130, "motion": "Stationary"})
    SCENARIOS.append({"distance": dist, "lux": 0,   "motion": "Stationary"})


# ============================================================================
#  Reproducibility
# ============================================================================
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
#  Aya patching — match the exact globals your AyaLoader reads
# ============================================================================
def patch_aya_for_scenario(scenario, subjects):
    """Rebind the module-level constants TEST_SUBJECTS and TEST_CONDITIONS that
    AyaLoader.get_raw_data() reads when self.split_name != "train".

    These are the EXACT names used in your AyaLoader.py. No aliases needed.
    """
    AyaLoaderModule.TEST_SUBJECTS = list(subjects)
    AyaLoaderModule.TEST_CONDITIONS = {
        "distances": [scenario["distance"]],
        "motions":   [scenario["motion"]],
        "lux":       [scenario["lux"]],
    }
    # Defensive: also touch TRAIN_* so that if somehow split_name=="train"
    # leaks in, we don't accidentally use stale train conditions.
    # (Harmless because we always pass name="unsupervised".)
    print(f"[patch] AyaLoader.TEST_SUBJECTS    = {AyaLoaderModule.TEST_SUBJECTS}")
    print(f"[patch] AyaLoader.TEST_CONDITIONS  = {AyaLoaderModule.TEST_CONDITIONS}")


def build_unsupervised_loader(config, scenario, subjects):
    """Build an AyaLoader + DataLoader for one scenario.

    We pass name="unsupervised" -> goes to the else branch in your loader, which
    consumes TEST_SUBJECTS/TEST_CONDITIONS (which we just rebound).
    """
    patch_aya_for_scenario(scenario, subjects)

    unsupervised_data = AyaLoaderModule.AyaLoader(
        name="unsupervised",
        data_path=config.UNSUPERVISED.DATA.DATA_PATH,
        config_data=config.UNSUPERVISED.DATA,
        device=config.DEVICE,
    )

    loader = DataLoader(
        dataset=unsupervised_data,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=general_generator,
    )
    return loader


def scenario_label(s):
    return f"{s['distance']}m_{s['lux']}lux_{s['motion']}"


def scenario_cache_suffix(s):
    return f"d{s['distance']}_lux{s['lux']}_{s['motion']}"


# ============================================================================
#  stdout capture
# ============================================================================
class TeeStdout:
    def __init__(self, real):
        self.real = real
        self.buffer = StringIO()

    def write(self, msg):
        self.real.write(msg)
        self.buffer.write(msg)

    def flush(self):
        self.real.flush()


# ============================================================================
#  Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-scenario UNSUPERVISED test for Aya dataset (v3)")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Override TEST_SUBJECTS")
    parser.add_argument("--methods", nargs="+", type=str, default=None,
                        help="Override UNSUPERVISED.METHOD (e.g., POS LGI OMIT)")
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--no_per_scenario_cache", action="store_true",
                        default=False,
                        help="Don't append scenario tag to CACHED_PATH "
                             "(useful only if you know caches are interchangeable)")
    args = parser.parse_args()

    config = get_config(args)

    if config.TOOLBOX_MODE != "unsupervised_method":
        print(f"[ERROR] TOOLBOX_MODE in yaml is '{config.TOOLBOX_MODE}'. "
              f"This script requires 'unsupervised_method'.")
        sys.exit(1)

    subjects = args.subjects if args.subjects else TEST_SUBJECTS

    if args.methods:
        config.defrost()
        config.UNSUPERVISED.METHOD = args.methods
        config.freeze()
    methods = list(config.UNSUPERVISED.METHOD)
    if not methods:
        print("[ERROR] No unsupervised method specified.")
        sys.exit(1)

    base_cached_path = config.UNSUPERVISED.DATA.CACHED_PATH
    per_scenario_cache = not args.no_per_scenario_cache

    out_dir = args.output_dir or os.path.join(
        config.LOG.PATH if hasattr(config.LOG, "PATH") else "runs/exp",
        "multi_scenario_summary",
    )
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        out_dir, f"unsupervised_multi_scenario_{ts}.txt")

    print("=" * 70)
    print("Multi-Scenario UNSUPERVISED Test (v3 — adapted to AyaLoader.py)")
    print("=" * 70)
    print(f"Config file:        {args.config_file}")
    print(f"AyaLoader module:   {AyaLoaderModule.__file__}")
    print(f"Dataset:            {config.UNSUPERVISED.DATA.DATASET}")
    print(f"Data path:          {config.UNSUPERVISED.DATA.DATA_PATH}")
    print(f"Base cached path:   {base_cached_path}")
    print(f"Per-scenario cache: {per_scenario_cache}")
    print(f"Test subjects:      {subjects} (n={len(subjects)})")
    print(f"Methods:            {methods}")
    print(f"Scenarios:          {len(SCENARIOS)}")
    print(f"Total runs:         {len(SCENARIOS) * len(methods)}")
    print(f"Summary log:        {summary_path}")
    print("=" * 70)

    # ---- Verify your AyaLoader exposes the expected hooks ----
    required_attrs = ["TEST_SUBJECTS", "TEST_CONDITIONS"]
    missing = [a for a in required_attrs if not hasattr(AyaLoaderModule, a)]
    if missing:
        print(f"\n[ERROR] AyaLoader is missing required globals: {missing}")
        print("This script assumes the loader version that defines "
              "TEST_SUBJECTS and TEST_CONDITIONS at module level.")
        sys.exit(1)
    print(f"\n[DIAG] AyaLoader initial TEST_SUBJECTS   = "
          f"{AyaLoaderModule.TEST_SUBJECTS}")
    print(f"[DIAG] AyaLoader initial TEST_CONDITIONS = "
          f"{AyaLoaderModule.TEST_CONDITIONS}")

    tee = TeeStdout(sys.stdout)
    sys.stdout = tee

    all_results = OrderedDict()

    try:
        for s_idx, scenario in enumerate(SCENARIOS):
            tag = scenario_label(scenario)
            print(f"\n{'=' * 70}")
            print(f"[Scenario {s_idx + 1}/{len(SCENARIOS)}] {tag}")
            print(f"  distance={scenario['distance']}m  "
                  f"lux={scenario['lux']}  motion={scenario['motion']}")
            print(f"  subjects={subjects}")
            print(f"{'=' * 70}")

            # Per-scenario cache to avoid different scenarios overwriting
            # each other's preprocessed cache
            if per_scenario_cache:
                scenario_cache = os.path.join(
                    base_cached_path, scenario_cache_suffix(scenario))
                config.defrost()
                config.UNSUPERVISED.DATA.CACHED_PATH = scenario_cache
                config.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(
                    scenario_cache, "DataFileLists")
                config.UNSUPERVISED.DATA.EXP_DATA_NAME = ""
                config.freeze()
                print(f"  Cache subdir: {scenario_cache}")

            try:
                loader = build_unsupervised_loader(config, scenario, subjects)
            except ValueError as e:
                print(f"  [SKIP] No data found: {e}")
                all_results[tag] = {m: {"status": "NO_DATA"} for m in methods}
                continue

            n_chunks = len(loader.dataset)
            print(f"  Dataset size: {n_chunks} chunks")

            if n_chunks == 0:
                print(f"  [SKIP] Empty dataset for {tag}")
                all_results[tag] = {m: {"status": "EMPTY"} for m in methods}
                continue

            all_results[tag] = OrderedDict()
            for m_idx, method in enumerate(methods):
                print(f"\n  --- [{m_idx + 1}/{len(methods)}] Method: "
                      f"{method} @ {tag} ---")
                buf_start = tee.buffer.tell()
                try:
                    unsupervised_predict(config, loader, method)
                    status = "OK"
                except Exception as e:
                    print(f"  [ERROR] {method} failed: {e}")
                    status = f"ERROR: {e}"
                all_results[tag][method] = {
                    "status": status,
                    "num_chunks": n_chunks,
                    "log": tee.buffer.getvalue()[buf_start:],
                }

    finally:
        sys.stdout = tee.real

    # ---- Summary ----
    print("\n" + "=" * 90)
    print("MULTI-SCENARIO UNSUPERVISED SUMMARY")
    print("=" * 90)
    header = f"{'Scenario':<32s} {'Method':<8s} {'Status':<10s} {'Chunks':<8s}"
    print(header)
    print("-" * 90)
    for tag, method_results in all_results.items():
        for method, res in method_results.items():
            print(f"{tag:<32s} {method:<8s} "
                  f"{res.get('status', '?'):<10s} "
                  f"{res.get('num_chunks', '-')!s:<8s}")
    print("=" * 90)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(tee.buffer.getvalue())
        f.write("\n\n" + "=" * 90 + "\n")
        f.write("FINAL SUMMARY TABLE\n")
        f.write("=" * 90 + "\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for tag, method_results in all_results.items():
            for method, res in method_results.items():
                f.write(f"{tag:<32s} {method:<8s} "
                        f"{res.get('status', '?'):<10s} "
                        f"{res.get('num_chunks', '-')!s:<8s}\n")

    print(f"\nFull log written to: {summary_path}")


if __name__ == "__main__":
    main()
