#!/usr/bin/env python3
"""Second Gemini visual opinion on original images marked empty by box review."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/v2_annotation/gemini_many_boxes"
MODEL = "gemini-3.8-flash-high"


def candidate_rows() -> list[dict]:
    cohort = {json.loads(line)["image_id"]: json.loads(line)
              for line in (BASE / "cohort.jsonl").open(encoding="utf-8")}
    rows = []
    for path in sorted((BASE / "reviews").glob("*.json")):
        low = json.loads(path.read_text(encoding="utf-8"))
        if low["squirrel_count"] != 0:
            continue
        row = cohort[low["image_id"]]
        rows.append({**row, "low_review_path": path.relative_to(ROOT).as_posix()})
    return rows


def run_one(row: dict) -> tuple[str, str]:
    key = row["review_key"]
    target = BASE / "empty_second_look" / f"{key}.json"
    if target.exists():
        return key, "cached"
    source = (ROOT / row["image_path"]).resolve()
    if not source.is_relative_to(ROOT) or not source.is_file():
        return key, "ERROR missing image"
    question = (f"Look carefully at the ORIGINAL image @{source} . Is any live squirrel "
                f"visibly present, including small or partly hidden animals? Start your answer "
                f"with exactly YES, NO, or UNCLEAR, then one concise sentence describing the "
                f"visible evidence. Ignore filename, metadata, and any detector predictions. "
                f"If you cannot inspect the image, answer UNCLEAR. Image id {row['image_id']}.")
    command = ["agy", "--model", MODEL, "--output-format", "json",
               "--dangerously-skip-permissions", "--print-timeout", "75s",
               f"--print={question}"]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
        (BASE / "empty_second_look_raw").mkdir(parents=True, exist_ok=True)
        (BASE / "empty_second_look_raw" / f"{key}.json").write_text(
            json.dumps({"returncode": result.returncode, "stdout": result.stdout,
                        "stderr": result.stderr}, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            raise ValueError(f"CLI exit {result.returncode}")
        wrapper = json.loads(result.stdout)
        answer = wrapper.get("response", "").strip()
        match = re.match(r"^(?:\*\*)?(YES|NO|UNCLEAR)\b", answer, re.IGNORECASE)
        if not match:
            raise ValueError(f"unparseable answer: {answer[:100]}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"image_id": row["image_id"],
                                      "review_key": key, "verdict": match.group(1).upper(),
                                      "evidence": answer, "gemini_model": MODEL,
                                      "conversation_id": wrapper.get("conversation_id"),
                                      "usage": wrapper.get("usage")}, indent=2) + "\n",
                          encoding="utf-8")
        return key, match.group(1).upper()
    except (ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        return key, f"ERROR {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = candidate_rows()
    if args.limit:
        rows = rows[:args.limit]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, row) for row in rows]
        for future in as_completed(futures):
            key, outcome = future.result()
            print(key, outcome, flush=True)
            if outcome.startswith("ERROR"):
                failures.append((key, outcome))
    print(json.dumps({"empty_candidates": len(rows), "failures": failures}, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
