"""Small real-GPU verification; never launches a full dataset experiment."""
from pathlib import Path
import argparse
import json
import shutil
import uuid

from csci_635_ml_project.baseline import (default_config, create_run, prepare_run, probe_batches,
                                        train_run, predict_run, benchmark_run, run_job, package_run)
from csci_635_ml_project.baseline_data import save_json, sha256, verify_dataset
from csci_635_ml_project.baseline_reports import dataset_report, evaluate_run

ROOT = Path(__file__).resolve().parents[1]


def smoke_data():
    source = ROOT / "data/squirrel-v2-audited-coco"
    manifest, docs = verify_dataset(source)
    output = ROOT / "output_model_comparison/smoke/data" / uuid.uuid4().hex[:8]
    for split, doc in docs.items():
        ids = set()
        for category in doc["categories"]:
            candidates = sorted({a["image_id"] for a in doc["annotations"] if a["category_id"] == category["id"]})
            ids.update(candidates[:2 if split == "train" else 1])
        subset = {**doc, "images": [im for im in doc["images"] if im["id"] in ids],
                  "annotations": [a for a in doc["annotations"] if a["image_id"] in ids]}
        folder = output / split; folder.mkdir(parents=True)
        for im in subset["images"]: shutil.copyfile(source / split / im["file_name"], folder / im["file_name"])
        save_json(folder / "_annotations.coco.json", subset)
        manifest["split_counts"][split] = len(subset["images"])
        manifest["annotation_counts"][split] = len(subset["annotations"])
        manifest["annotation_hashes"][split] = sha256(folder / "_annotations.coco.json")
    manifest["name"] = "SMOKE ONLY — source subset, not a valid baseline"
    save_json(output / ".complete.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["yolo26", "yolov5", "both"], default="both")
    parser.add_argument("--check-resume", action="store_true")
    args = parser.parse_args()
    data = smoke_data()
    for model in (["yolo26", "yolov5"] if args.model == "both" else [args.model]):
        config = default_config(model)
        config.update(epochs=2, workers=0, batch_candidates=[4], benchmark_images=4, benchmark_warmup=2, smoke=True)
        run = create_run(ROOT, config, data=data)
        print("SMOKE_RUN", model, run, flush=True)
        prepare_run(run); dataset_report(run, data); selected = probe_batches(run)
        if args.check_resume:
            context = json.loads((run / "run_manifest.json").read_text())
            try:
                run_job(run, "train", batch=selected["batch_size"], probe=False, data_yaml=context["yolo_yaml"],
                        destination=str(run / "native"), resume=None, interrupt_after=1)
            except RuntimeError as exc:
                if "Intentional smoke interruption" not in str(exc): raise
            else:
                raise AssertionError("Expected interruption after the first checkpoint")
            assert (run / "native/weights/last_resumable.pt").exists()
        train_run(run); predict_run(run)
        metrics = evaluate_run(run, data, heads=config["heads"])
        assert set(metrics) == set(config["heads"])
        benchmark_run(run)
        # Skip optional external tracking for verification runs.
        import csci_635_ml_project.baseline as baseline
        baseline.track_run = lambda run: None
        package_run(run)
        assert json.loads((run / "training_complete.json").read_text())["epochs_completed"] == 2
        print("SMOKE_PASSED", model, run, flush=True)




def check_comparison(yolo26_run, yolov5_run):
    """Exercise the full comparison on genuine smoke predictions; no baseline claim."""
    from csci_635_ml_project.baseline_comparison import build_comparison
    from csci_635_ml_project.baseline_data import records_for_split
    from csci_635_ml_project.baseline_reports import evaluate_report
    modern = Path(yolo26_run).resolve(); legacy = Path(yolov5_run).resolve()
    context = json.loads((modern / "run_manifest.json").read_text())
    data = Path(context["data"])
    manifest, docs = verify_dataset(data)
    old_manifest = json.loads((legacy / "dataset_manifest.json").read_text())
    assert manifest == old_manifest, "Smoke datasets differ"
    root = ROOT / "output_model_comparison/smoke/comparison" / uuid.uuid4().hex[:8]
    (root / "data").mkdir(parents=True)
    (root / "data/squirrel-v2-audited-coco").symlink_to(data, target_is_directory=True)
    reference = ROOT / "output/runs/20260926T153802Z-v2-f35e9e"
    rf = root / "reference_rfdetr"; rf.mkdir()
    save_json(rf / "dataset_manifest.json", manifest)
    for name in ("environment.json", "decision_threshold.json"):
        shutil.copyfile(reference / name, rf / name)
    completion = json.loads((reference / "training_complete.json").read_text())
    completion["best_checkpoint"] = str(reference / completion["best_checkpoint"])
    save_json(rf / "training_complete.json", completion)
    import pandas as pd
    inventory = pd.read_csv(reference / "reports/image_inventory.csv")
    selected_ids = {(split, im["id"]) for split, doc in docs.items() for im in doc["images"]}
    inventory = inventory[[ (row.split, row.image_id) in selected_ids for row in inventory.itertuples() ]]
    (rf / "reports").mkdir(exist_ok=True)
    inventory.to_csv(rf / "reports/image_inventory.csv", index=False)
    config = json.loads((reference / "config.json").read_text())
    config.update(benchmark_images=4, benchmark_warmup=2, cpu_threads=1)
    save_json(rf / "config.json", config)
    threshold = json.loads((rf / "decision_threshold.json").read_text())["threshold"]
    for split in ("valid", "test"):
        directory = rf / "reports" / split; directory.mkdir(parents=True)
        raw = json.loads((reference / "reports" / split / "predictions.json").read_text())
        selected = {str(im["id"]): raw[str(im["id"])] for im in docs[split]["images"]}
        save_json(directory / "predictions.json", selected)
        known = {int(k): [p for p in items if p["category_id"] <= 11] for k, items in selected.items()}
        evaluate_report(docs[split], records_for_split(data, split, docs[split]), known, threshold, directory)
    shutil.copyfile(reference / "reports/training_history.csv", rf / "reports/training_history.csv")
    output, table = build_comparison(root, dict(rfdetr=rf, yolo26=modern, yolov5=legacy), refresh_benchmarks=False)
    assert len(table) == 4
    assert (output / "reports.zip").is_file()
    (output / "SMOKE_ONLY.txt").write_text("Pipeline verification using tiny datasets. These scores are not model baselines.\n")
    print("COMPARISON_SMOKE_PASSED", output, flush=True)
    return output


if __name__ == "__main__": main()
