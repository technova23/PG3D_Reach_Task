#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pg3d.eval import audit_beam_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ITPS-beam pruning from its JSON trace.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--jsonl-index", type=int, default=None)
    args = parser.parse_args(argv)
    payload = _load_payload(args.trace, jsonl_index=args.jsonl_index)
    result = audit_beam_trace(payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _load_payload(path: Path, *, jsonl_index: int | None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if jsonl_index is None:
        payload = json.loads(text)
    else:
        payload = json.loads(text.splitlines()[jsonl_index])
    if "beam" in payload:
        return dict(payload["beam"])
    if "beam_trace" in payload:
        return dict(payload["beam_trace"])
    return dict(payload)


if __name__ == "__main__":
    raise SystemExit(main())
