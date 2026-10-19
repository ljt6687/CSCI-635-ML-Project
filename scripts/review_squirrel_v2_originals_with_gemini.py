#!/usr/bin/env python3
"""Blind Gemini scene pass on all 63 original images, without detector overlays."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/v2_annotation/gemini_many_boxes"
MODEL = "gemini-3.8-flash-low"


def rows() -> list[dict]:
    source = BASE / "cohort.jsonl"
    records = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    if len(records) != 63:
        raise ValueError(f"expected 63 images, got {len(records)}")
    return records


def parse(output: str, row: dict) -> dict:
    wrapper = json.loads(output)
    if wrapper.get("status") != "SUCCESS":
        raise ValueError(f"CLI status {wrapper.get('status')}")
    text = wrapper.get("response", "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    body = json.loads(text)
    if body.get("image_id") != row["image_id"]:
        raise ValueError("image ID mismatch")
    if not isinstance(body.get("squirrel_count"), int) or body["squirrel_count"] < 0:
        raise ValueError("invalid count")
    if not isinstance(body.get("uncertain"), bool):
        raise ValueError("invalid uncertainty")
    if not body.get("evidence", "").strip():
        raise ValueError("missing visual evidence")
    return {**body, "review_key": row["review_key"], "image_path": row["image_path"],
            "gemini_model": MODEL, "conversation_id": wrapper.get("conversation_id"),
            "usage": wrapper.get("usage")}


def run_one(row: dict) -> tuple[str, str]:
    key = row["review_key"]
    target = BASE / "original_reviews" / f"{key}.json"
    if target.exists():
        return key, "cached"
    original = (ROOT / row["image_path"]).resolve()
    if not original.is_relative_to(ROOT) or not original.is_file():
        return key, "ERROR missing image"
    template = {"image_id": row["image_id"], "squirrel_count": 0,
                "uncertain": False, "evidence": "brief specific visible evidence and approximate location"}
    question = (f"Examine the ORIGINAL unmarked image @{original} . Count distinct "
                f"visibly identifiable squirrel bodies, including partly hidden or deceased specimens. "
                f"Exclude tracks, drawings, and squirrel houses. Use only visible pixels; "
                f"ignore filename, species metadata, and previous model predictions. "
                f"If an object is too small or blurry to identify, set uncertain true. "
                f"If you cannot open the actual image, set uncertain true and say so. "
                f"Reply only compact JSON of this form: {json.dumps(template)}")
    command = ["agy", "--model", MODEL, "--output-format", "json",
               "--dangerously-skip-permissions", "--print-timeout", "60s",
               f"--print={question}"]
    target.parent.mkdir(parents=True, exist_ok=True)
    (BASE / "original_raw").mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 3):
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=75)
            (BASE / "original_raw" / f"{key}_attempt{attempt}.json").write_text(
                json.dumps({"returncode": result.returncode, "stdout": result.stdout,
                            "stderr": result.stderr}, indent=2) + "\n", encoding="utf-8")
            if result.returncode:
                raise ValueError(f"CLI exit {result.returncode}")
            review = parse(result.stdout, row)
            target.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
            return key, "ok"
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            last = str(exc)
    return key, f"ERROR {last}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    candidates = rows()
    if args.limit:
        candidates = candidates[:args.limit]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, row) for row in candidates]
        for future in as_completed(futures):
            key, status = future.result()
            print(key, status, flush=True)
            if status.startswith("ERROR"):
                failures.append((key, status))
    print(json.dumps({"requested": len(candidates), "failures": failures,
                      "validated": len(list((BASE / 'original_reviews').glob('*.json')))}, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
