#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pg3d.eval import calibrate_verifier_buffer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the ITPS imagined-verification buffer.")
    parser.add_argument("trace_roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.output}")
    rows = [row for root in args.trace_roots for row in _load_trace_rows(root)]
    errors = [
        float(value)
        for row in rows
        for value in row.get("optimistic_clearance_error_m", [])
    ]
    calibration = calibrate_verifier_buffer(errors)
    payload: dict[str, Any] = {
        "schema_version": "pg3d.itps_verifier_calibration.v1",
        "classification": "development_only",
        "trace_roots": [str(path.resolve()) for path in args.trace_roots],
        **calibration.to_json(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if calibration.verifier_valid else 2


def _load_trace_rows(root: Path) -> list[dict[str, Any]]:
    paths = [root] if root.is_file() else sorted(root.glob("verification/*/episode_*.jsonl"))
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
