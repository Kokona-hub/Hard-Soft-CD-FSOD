#!/usr/bin/env python
"""Validate an S2H knowledge cache before a train/test run.

The check is intentionally independent of the detector.  It catches the
failure mode in which CHSD/full caches contain zero reliability for every
class, which makes ATAR produce a zero correction while the launcher still
labels the run as an S2H experiment.
"""
import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--stage", required=True,
                        choices=("ffcp", "ffcp_chsd", "full"))
    parser.add_argument("--classes", type=int, default=0)
    parser.add_argument("--representation", default=None,
                        choices=("backbone_neck", "decoder_query"),
                        help="require a representation tag when present")
    parser.add_argument("--allow-zero", action="store_true",
                        help="allow zero reliability (for an explicit smoke test)")
    args = parser.parse_args()

    if not args.path.is_file() or args.path.stat().st_size == 0:
        raise SystemExit(f"missing knowledge file: {args.path}")
    item = torch.load(args.path, map_location="cpu")
    required = ("hard", "soft", "kappa", "reliability", "classes",
                "token_ids", "meta")
    missing = [key for key in required if key not in item]
    if missing:
        raise SystemExit(f"invalid knowledge file; missing keys: {missing}")

    meta = item.get("meta") or {}
    representation = str(meta.get("representation", "backbone_neck"))
    if args.representation and representation != args.representation:
        raise SystemExit(
            f"representation mismatch: expected {args.representation}, "
            f"file has {representation}")
    file_stage = str(meta.get("stage", "")).lower()
    if file_stage == "chsd":
        file_stage = "ffcp_chsd"
    if file_stage != args.stage:
        raise SystemExit(
            f"knowledge stage mismatch: requested {args.stage}, file has "
            f"{file_stage or '<missing>'}")

    classes = list(item["classes"])
    reliability = torch.as_tensor(item["reliability"], dtype=torch.float32)
    kappa = torch.as_tensor(item["kappa"], dtype=torch.float32)
    hard = torch.as_tensor(item["hard"])
    soft = torch.as_tensor(item["soft"])
    if args.classes and len(classes) != args.classes:
        raise SystemExit(
            f"class count mismatch: expected {args.classes}, got {len(classes)}")
    if hard.ndim != 2 or soft.shape != hard.shape:
        raise SystemExit(f"invalid prototype shapes: hard={tuple(hard.shape)} "
                         f"soft={tuple(soft.shape)}")
    if reliability.numel() != len(classes) or kappa.numel() != len(classes):
        raise SystemExit("per-class reliability/kappa length does not match classes")
    if not torch.isfinite(reliability).all() or not torch.isfinite(kappa).all():
        raise SystemExit("reliability or kappa contains NaN/Inf")
    if (reliability < 0).any() or (reliability > 1).any():
        raise SystemExit("reliability must lie in [0, 1]")
    if (kappa < 0).any() or (kappa >= 1).any():
        raise SystemExit("kappa must lie in [0, 1)")

    positive = int((reliability > 1e-6).sum())
    if not args.allow_zero and args.stage != "ffcp" and positive == 0:
        raise SystemExit(
            f"{args.stage} cache has zero reliability for all {len(classes)} "
            "classes; rebuild it with the reliability fallback fix")
    print(f"[knowledge-ok] stage={args.stage} classes={len(classes)} "
          f"representation={representation} "
          f"reliability_mean={float(reliability.mean()):.4f} "
          f"min={float(reliability.min()):.4f} "
          f"max={float(reliability.max()):.4f} "
          f"positive={positive}/{len(classes)}")
    zero_classes = [name for name, value in zip(classes, reliability.tolist())
                    if value <= 1e-6]
    if zero_classes:
        print(f"[knowledge-info] zero-reliability classes: {', '.join(zero_classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
