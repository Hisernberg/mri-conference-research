#!/usr/bin/env python3
"""Generate the baseline and ablation configs from configs/base.yaml.

Every config is *derived*, so a change to the base propagates everywhere and no
ablation can silently drift from the reference setup.

    python scripts/make_configs.py --base configs/base.yaml --out configs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmmf.config import ExperimentConfig            # noqa: E402
from qmmf.models import (                            # noqa: E402
    A14_VARIANTS, ABLATIONS, BASELINE_IDS, parameter_matched_widths,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--out", default="configs")
    ap.add_argument("--check-parameter-match", action="store_true",
                    help="Re-derive the A16 parameter-matched widths and report them.")
    args = ap.parse_args()

    base = ExperimentConfig.load(args.base)
    out = Path(args.out)
    (out / "baselines").mkdir(parents=True, exist_ok=True)
    (out / "ablations").mkdir(parents=True, exist_ok=True)

    written = []
    for bid, model in BASELINE_IDS.items():
        cfg = base.merged({"model": model, "ablation": "A0"})
        path = out / "baselines" / f"{bid.lower()}_{model}.json"
        cfg.save(path)
        written.append(path)

    for ident, ablation in ABLATIONS.items():
        if ident == "A14":
            for d in A14_VARIANTS:
                cfg = ablation.apply(base).merged({"data.context_slices": d})
                path = out / "ablations" / f"A14_d{d}.json"
                cfg.save(path)
                written.append(path)
            continue
        cfg = ablation.apply(base)
        path = out / "ablations" / f"{ident}.json"
        cfg.save(path)
        written.append(path)

    print(f"wrote {len(written)} configs under {out}")
    for p in written:
        print(f"  {p}")

    if args.check_parameter_match:
        match = parameter_matched_widths(base, "equal_mean_var")
        print("\nA16 parameter matching (plan 9.2):")
        print(json.dumps(match, indent=2))
        current = ABLATIONS["A16"].overrides["widths"]
        if list(current) != list(match["widths"]):
            print(f"WARNING: ABLATIONS['A16'] uses {current} but the search "
                  f"suggests {match['widths']}. Update models/__init__.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
