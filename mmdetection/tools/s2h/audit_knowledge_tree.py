#!/usr/bin/env python
"""Report S2H cache protocol and reliability statistics.

This is a read-only audit used before aggregating an ablation table or Fig. 4.
"""
import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    files = sorted(args.root.rglob("*.pth")) if args.root.exists() else []
    if not files:
        raise SystemExit(f"no .pth files under {args.root}")
    print("path,stage,classes,reliability_mean,reliability_min,reliability_max,positive")
    for path in files:
        try:
            item = torch.load(path, map_location="cpu")
            meta = item.get("meta") or {}
            stage = str(meta.get("stage", "<missing>"))
            rel = torch.as_tensor(item["reliability"], dtype=torch.float32)
            positive = int((rel > 1e-6).sum())
            print(f"{path},{stage},{len(item.get('classes', []))},"
                  f"{float(rel.mean()):.6f},{float(rel.min()):.6f},"
                  f"{float(rel.max()):.6f},{positive}")
        except Exception as exc:  # keep the audit useful for mixed directories
            print(f"{path},ERROR,{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
