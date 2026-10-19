#!/usr/bin/env python3
"""Prepare and run an independent Gemini visual audit of crowded v1 detections.

Gemini results are provisional review assistance, never verified COCO labels.
The script resumes from validated per-image results and preserves raw CLI output.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "output/v2_annotation/squirrel_v2_inference.jsonl"
OUTPUT = ROOT / "output/v2_annotation/gemini_many_boxes"
MODEL = "gemini-3.8-flash-low"
COLORS = ("#ff2020", "#15de35", "#00bfff", "#e600de", "#ffbb00", "#ffffff")

SCHEMA = {
    "type": "object",
    "required": ["image_id", "squirrel_count", "box_reviews", "missed_squirrel", "overall_notes"],
    "properties": {
        "image_id": {"type": "string"},
        "squirrel_count": {"type": "integer", "minimum": 0},
        "box_reviews": {"type": "array", "items": {"type": "object",
            "required": ["box_id", "verdict", "reason"],
            "properties": {"box_id": {"type": "string"},
                           "verdict": {"enum": ["keep", "adjust", "remove", "uncertain"]},
                           "reason": {"type": "string"}}}},
        "missed_squirrel": {"type": "boolean"},
        "overall_notes": {"type": "string"},
    },
}



def proposal_rows() -> list[dict]:
    rows = []
    with PREDICTIONS.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            material = [det for det in row["detections"] if det["score"] >= 0.20]
            if len(material) > 3:
                rows.append({
                    "image_id": row["image_id"],
                    "source": row["source"],
                    "bucket": row["bucket"],
                    "image_path": row["image_path"],
                    "width": row["width"], "height": row["height"],
                    "model_sha256": row["model_sha256"],
                    "proposals": [dict(box_id=f"B{i}", score=det["score"], xyxy=det["xyxy"])
                                  for i, det in enumerate(material, 1)],
                })
    rows.sort(key=lambda row: row["image_id"])
    if len(rows) != 63:
        raise ValueError(f"expected frozen 63-image cohort, got {len(rows)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("repeated image IDs")
    return rows


def safe_name(index: int, image_id: str) -> str:
    return f"{index:03d}_{hashlib.sha256(image_id.encode()).hexdigest()[:12]}"


def make_overlay(row: dict, path: Path) -> None:
    source = (ROOT / row["image_path"]).resolve()
    if not source.is_relative_to(ROOT) or not source.is_file():
        raise ValueError(f"image outside repository or missing: {row['image_path']}")
    with Image.open(source) as original:
        image = original.convert("RGB")
    if image.size != (row["width"], row["height"]):
        raise ValueError(f"dimensions changed for {row['image_id']}")
    draw = ImageDraw.Draw(image)
    short_side = min(image.size)
    font_size = max(12, min(25, short_side // 65))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    line_width = max(3, short_side // 330)
    for index, det in enumerate(row["proposals"]):
        x1, y1, x2, y2 = det["xyxy"]
        color = COLORS[index % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{det['box_id']} {det['score']:.2f}"
        left = max(0, min(int(x1), image.width - 125))
        top = max(0, min(int(y1) - font_size - 7, image.height - font_size - 7))
        bbox = draw.textbbox((left, top), label, font=font)
        draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill="#151515")
        draw.text((left, top), label, fill=color, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=94, subsampling=0)


def prepare(rows: list[dict]) -> None:
    (OUTPUT / "overlays").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "reviews").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "schema.json").write_text(json.dumps(SCHEMA, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "cohort.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            name = safe_name(index, row["image_id"])
            overlay = OUTPUT / "overlays" / f"{name}.jpg"
            if not overlay.exists():
                make_overlay(row, overlay)
            stream.write(json.dumps({**row, "review_key": name,
                                     "overlay_path": overlay.relative_to(ROOT).as_posix()}) + "\n")


def prompt(row: dict) -> str:
    overlay = (ROOT / row["overlay_path"]).resolve()
    ids = [det["box_id"] for det in row["proposals"]]
    template = {"image_id": row["image_id"], "squirrel_count": 0,
                "box_reviews": [{"box_id": box_id, "verdict": "keep|adjust|remove|uncertain",
                                 "reason": "visual evidence"} for box_id in ids],
                "missed_squirrel": False, "overall_notes": "brief scene description"}
    return (f"Inspect the actual annotated image @{overlay} . The colored labels "
            f"{', '.join(ids)} are proposed squirrel boxes, not ground truth. "
            f"Judge each box separately from visible pixels. Count distinct squirrel bodies, "
            f"not rectangles. Keep a box only if it substantially encloses one visible squirrel; "
            f"remove background hits, duplicates, or fragment boxes when a better box covers "
            f"the same animal; adjust a box on a real squirrel with wrong extents; mark unclear "
            f"when pixels cannot resolve it. Report any squirrel missed by all boxes. "
            f"Use a specific visual reason for every box, and mention if no squirrel is visible. "
            f"If you cannot see the image, mark all boxes uncertain and say so. "
            f"Output ONLY JSON with this structure and all box IDs exactly once: "
            f"{json.dumps(template)}")



def parse_response(output: str, row: dict) -> dict:
    wrapper = json.loads(output)
    if wrapper.get("status") != "SUCCESS":
        raise ValueError(f"CLI status {wrapper.get('status')}: {str(wrapper.get('response'))[:400]}")
    body = wrapper["response"]
    if isinstance(body, str):
        body = body.strip()
        if body.startswith("```"):
            body = body.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        body = json.loads(body)
    if body.get("image_id") != row["image_id"]:
        raise ValueError("image_id mismatch")
    ids = [item["box_id"] for item in body["box_reviews"]]
    expected = [item["box_id"] for item in row["proposals"]]
    if sorted(ids) != sorted(expected) or len(ids) != len(expected):
        raise ValueError(f"box IDs do not match: {ids} vs {expected}")
    allowed = {"keep", "adjust", "remove", "uncertain"}
    if any(item["verdict"] not in allowed or not item["reason"].strip()
           for item in body["box_reviews"]):
        raise ValueError("invalid box verdict or empty reason")
    if not isinstance(body.get("squirrel_count"), int) or body["squirrel_count"] < 0:
        raise ValueError("invalid squirrel_count")
    if not isinstance(body.get("missed_squirrel"), bool):
        raise ValueError("invalid missed_squirrel")
    if not body.get("overall_notes", "").strip():
        raise ValueError("empty overall_notes")
    return {**body, "source": row["source"], "bucket": row["bucket"],
            "review_key": row["review_key"], "image_path": row["image_path"],
            "model_sha256": row["model_sha256"], "gemini_model": MODEL,
            "cli_conversation_id": wrapper.get("conversation_id"),
            "cli_usage": wrapper.get("usage")}


def run_one(row: dict) -> tuple[str, str]:
    key = row["review_key"]
    target = OUTPUT / "reviews" / f"{key}.json"
    if target.exists():
        return key, "cached"
    args = ["agy", "--model", MODEL, "--output-format", "json",
            "--dangerously-skip-permissions", "--print-timeout", "60s",
            f"--print={prompt(row)}"]
    for attempt in range(1, 3):
        completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=90)
        raw = OUTPUT / "raw" / f"{key}_attempt{attempt}.json"
        raw.write_text(json.dumps({"returncode": completed.returncode,
                                   "stdout": completed.stdout, "stderr": completed.stderr}, indent=2) + "\n",
                       encoding="utf-8")
        try:
            if completed.returncode:
                raise ValueError(f"CLI exit {completed.returncode}: {completed.stderr[-500:]}")
            review = parse_response(completed.stdout, row)
            target.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
            return key, "ok"
        except (ValueError, KeyError, TypeError) as exc:
            last_error = str(exc)
    return key, f"ERROR {last_error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    rows = proposal_rows()
    prepare(rows)
    rows = [json.loads(line) for line in (OUTPUT / "cohort.jsonl").open()]
    if args.prepare_only:
        print(f"prepared {len(rows)} overlays at {OUTPUT}")
        return
    if args.limit:
        rows = rows[:args.limit]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, row): row for row in rows}
        for future in as_completed(futures):
            key, status = future.result()
            print(f"{key} {status}", flush=True)
            if status.startswith("ERROR"):
                failures.append((key, status))
    print(json.dumps({"requested": len(rows), "failures": failures,
                      "validated_reviews": len(list((OUTPUT / 'reviews').glob('*.json')))}, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
