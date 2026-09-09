#!/usr/bin/env python3
"""Use the authorized Kaggle account without placing credentials in this repo."""
import argparse
import json
import os
from pathlib import Path


def connect():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        credential = Path(os.environ.get("MRI_KAGGLE_TOKEN_FILE", str(Path.home() / ".config/mri-kaggle/api-token")))
        if credential.exists():
            os.environ["KAGGLE_API_TOKEN"] = credential.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    return api


def native(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        return native(obj.to_dict())
    if isinstance(obj, (tuple, list)):
        return [native(x) for x in obj]
    return str(obj)


def quota_summary(response):
    # Kaggle SDK 2.2.4's Duration text serializer can produce strings such as
    # "500.74.0s". Read its typed timedelta fields before formatting a budget.
    result = {"quota_refresh_time": response.quota_refresh_time.isoformat()}
    for name in ["gpu", "tpu"]:
        quota = getattr(response, f"{name}_quota")
        fields = {key: getattr(quota, key).total_seconds()
                  for key in ["time_used", "time_reserved", "total_time_allowed", "minimum_time_allowed"]}
        fields["remaining_unreserved"] = max(0., fields["total_time_allowed"] - fields["time_used"] - fields["time_reserved"])
        result[name] = {"unit": "seconds", **fields}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["push", "status", "output", "quota"])
    p.add_argument("target", nargs="?")
    p.add_argument("--out", default="kaggle_outputs")
    p.add_argument("--accelerator", default="NvidiaTeslaT4", choices=["NvidiaTeslaT4", "NvidiaTeslaP100", "TpuV6E8", "None"])
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--pattern", default=None, help="Optional output filename regular expression")
    a = p.parse_args(); api = connect()
    if a.action == "push":
        result = api.kernels_push(a.target, acc=a.accelerator, timeout=str(a.timeout))
    elif a.action == "status":
        result = api.kernels_status(a.target)
    elif a.action == "quota":
        result = quota_summary(api.quota_view())
    else:
        files, token = api.kernels_output(a.target, path=a.out, force=True, quiet=True,
                                          file_pattern=a.pattern)
        result = {"output_directory": a.out, "downloaded_files": len(files), "next_page_token": token}
        Path(a.out, "download_inventory.json").write_text(json.dumps(files, indent=2))
    print(json.dumps(native(result), indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
