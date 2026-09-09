"""Equal weighting of conservative image groups; no patient-identity claim."""
from collections import defaultdict
import numpy as np


def group_means(per_case, case_to_group):
    """Mean cases within each group, retaining one value per observed group."""
    values = defaultdict(list)
    for cid, value in per_case.items():
        if cid not in case_to_group or not np.isfinite(value):
            raise ValueError("Every finite case score needs a known similarity group")
        values[case_to_group[cid]].append(float(value))
    if not values:
        raise ValueError("Cannot aggregate an empty cohort")
    return {group: float(np.mean(items)) for group, items in sorted(values.items())}


def group_quality_medians(case_ids, cache, case_to_group):
    """One robust descriptor matrix per training group for fitting its scaler."""
    values = defaultdict(list)
    for cid in case_ids:
        values[case_to_group[cid]].append(cache.quality_raw(cid))
    return [np.median(np.stack(values[g]), axis=0) for g in sorted(values)]
