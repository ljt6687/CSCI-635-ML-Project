# Final held-out test report: RF-DETR Medium, YOLO26 Medium, and YOLOv5 Medium

Prepared October 5, 2026. This report summarizes the completed **11-class audited v2 experiments** using their saved best checkpoints and frozen validation-selected confidence thresholds. Test scores were recomputed from archived predictions and checked against the original reports; no models were retrained and no thresholds were selected using test results.

## Main findings

- **RF-DETR is the accuracy leader:** 66.11% COCO mAP50–95, 80.95% AP50, and 76.81% micro F1. Its mAP50–95 exceeds YOLO26’s primary head by **12.98 percentage points** and YOLOv5 by **15.71 points**.
- **YOLO26 is the stronger primary YOLO baseline:** 53.13% mAP5git@github.com:ljt6687/CSCI-635-ML-Project.git0–95 and 72.49% micro F1, versus YOLOv5’s 50.39% and 70.57%.
- RF-DETR has the highest per-class AP50–95 for **all 11 classes**. `generic_squirrel` has the lowest per-class AP50–95 for all three primary models.
- YOLO26’s secondary NMS-free head improves its mAP50–95 to **54.84%** and recall to **67.75%**, with lower precision and micro F1 than its primary head.
- Among the comparable saved YOLO benchmarks, **YOLOv5 is faster on GPU** and **YOLO26 is faster on CPU**. RF-DETR’s archived timing follows a different protocol, so these files do not support a fair three-model speed ranking.

## Dataset and evaluation protocol

| Split | Images | Annotated objects |
| --- | --- | --- |
| train | 5988 | 6379 |
| valid | 1282 | 1350 |
| test | 1284 | 1380 |
| Total | 8554 | 9109 |

The held-out test set contains **1,284 images and 1,380 ground-truth bounding boxes**, covering ten named squirrel species plus `generic_squirrel`. All three run manifests have the same dataset identity, class order, annotation hashes, and split counts. Current COCO annotations match the archived hashes. Object support in the per-class tables counts boxes, so it can exceed the number of images for a class.

- **COCO AP:** pycocotools bounding-box evaluation; AP50–95 averages IoU thresholds 0.50 to 0.95 in steps of 0.05. AP50 and AP75 use fixed IoU thresholds. COCO maxDets is `[1, 10, 100]`; saved prediction floor is 0.0001. AP uses the ranked saved predictions, independently of the operating confidence threshold.
- **Operating precision, recall, and F1:** matching IoU 0.50 at each model’s frozen threshold. Predictions are processed by descending confidence and matched to the best unused ground-truth box without a class restriction. A localized box with the wrong class counts as one false positive and one false negative.
- **Micro scores** aggregate TP/FP/FN across classes. **Macro scores** average each class’s precision, recall, or F1 equally. These are detection scores; ordinary image-classification accuracy and true-negative rates are not defined by these reports.
- **Confidence thresholds:** maximum validation micro F1 on a 0.01-spaced confidence grid; ties choose the higher threshold. Checkpoint and threshold were frozen before test evaluation.
- **AR1 / AR10 / AR100:** COCO average recall over IoU 0.50–0.95 with at most 1 / 10 / 100 detections per image. These differ from operating recall at IoU 0.50.
- **Mean matched IoU:** average overlap of localization matches above the confidence threshold, including matches whose predicted species is wrong. It is not restricted to correctly classified true positives.

Unless otherwise stated, all AP, AR, precision, recall, F1, and IoU values below are **percentages**, rounded to two decimal places. Confidence thresholds remain on the 0–1 scale. Differences are percentage points.

## Primary held-out test scores

| Metric | RF-DETR Medium | YOLO26 accuracy head | Original YOLOv5 Medium |
| --- | --- | --- | --- |
| COCO mAP50–95 | 66.11 | 53.13 | 50.39 |
| AP50 | 80.95 | 71.49 | 68.87 |
| AP75 | 73.44 | 58.69 | 55.65 |
| Micro precision | 84.26 | 81.15 | 80.37 |
| Micro recall | 70.58 | 65.51 | 62.90 |
| Micro F1 | 76.81 | 72.49 | 70.57 |
| Macro precision | 84.60 | 81.43 | 80.89 |
| Macro recall | 70.77 | 65.57 | 63.01 |
| Macro F1 | 76.94 | 72.44 | 70.51 |
| Mean matched IoU | 90.62 | 87.88 | 87.53 |
| Confidence threshold | 0.65 | 0.62 | 0.61 |
| True positives (TP) | 974 | 904 | 868 |
| False positives (FP) | 182 | 210 | 212 |
| False negatives (FN) | 406 | 476 | 512 |
| Predictions retained at threshold | 1156 | 1114 | 1080 |

RF-DETR correctly detects and classifies **70 more objects than YOLO26** and **106 more than YOLOv5**, with 28 and 30 fewer false positives, respectively. Each model satisfies `TP + FN = 1,380`.

## COCO scores by object size and detection limit

| Metric | RF-DETR Medium | YOLO26 accuracy head | Original YOLOv5 Medium |
| --- | --- | --- | --- |
| AP small | 17.05 | 2.71 | 2.60 |
| AP medium | 35.96 | 24.66 | 22.62 |
| AP large | 69.39 | 56.55 | 53.86 |
| AR1 | 72.34 | 58.79 | 57.29 |
| AR10 | 85.46 | 65.46 | 63.13 |
| AR100 | 86.63 | 65.80 | 63.28 |
| AR small | 51.90 | 7.38 | 3.81 |
| AR medium | 69.90 | 37.96 | 38.66 |
| AR large | 88.53 | 68.76 | 65.81 |

Test annotations contain **14 small**, **127 medium**, and **1,239 large** boxes (89.78% large), using standard COCO area cutoffs of 32² and 96² pixels in the exported annotation coordinates. The counts use `<32²`, `32²–<96²`, and `≥96²`; COCO’s evaluator includes boundary areas in adjacent ranges. Small-object estimates rely on few examples and should be interpreted cautiously. RF-DETR leads every size-specific AP/AR score in the primary comparison, but small-object AP remains low for all three models.

## Per-class test comparison

| Class | Ground-truth boxes | RF-DETR AP50–95 | YOLO26 AP50–95 | YOLOv5 AP50–95 |
| --- | --- | --- | --- | --- |
| Callosciurus erythraeus | 121 | 66.91 | 48.73 | 42.44 |
| Sciurus aureogaster | 123 | 64.30 | 47.71 | 46.26 |
| Sciurus carolinensis | 157 | 64.71 | 52.58 | 48.01 |
| Sciurus granatensis | 124 | 70.63 | 57.99 | 55.87 |
| Sciurus griseus | 132 | 66.76 | 53.56 | 55.51 |
| Sciurus lis | 94 | 63.17 | 52.21 | 49.95 |
| Sciurus niger | 126 | 66.70 | 53.25 | 48.55 |
| Sciurus vulgaris | 126 | 64.85 | 54.46 | 50.37 |
| Tamiasciurus douglasii | 131 | 68.00 | 63.54 | 62.69 |
| Tamiasciurus hudsonicus | 124 | 74.33 | 61.67 | 58.57 |
| generic_squirrel | 122 | 56.79 | 38.70 | 36.09 |

The largest RF-DETR AP50–95 advantage over YOLO26 is for `Callosciurus erythraeus` (**18.18 points**). Its strongest class is `Tamiasciurus hudsonicus` (**74.33%**). The strongest primary YOLO class is `Tamiasciurus douglasii` (**63.54%** for YOLO26; **62.69%** for YOLOv5).

### RF-DETR Medium

| Class | Support | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50–95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Callosciurus erythraeus | 121 | 84 | 20 | 37 | 80.77 | 69.42 | 74.67 | 77.04 | 66.91 |
| Sciurus aureogaster | 123 | 80 | 17 | 43 | 82.47 | 65.04 | 72.73 | 75.55 | 64.30 |
| Sciurus carolinensis | 157 | 103 | 30 | 54 | 77.44 | 65.61 | 71.03 | 76.92 | 64.71 |
| Sciurus granatensis | 124 | 92 | 14 | 32 | 86.79 | 74.19 | 80.00 | 87.49 | 70.63 |
| Sciurus griseus | 132 | 101 | 26 | 31 | 79.53 | 76.52 | 77.99 | 80.62 | 66.76 |
| Sciurus lis | 94 | 72 | 8 | 22 | 90.00 | 76.60 | 82.76 | 88.43 | 63.17 |
| Sciurus niger | 126 | 86 | 14 | 40 | 86.00 | 68.25 | 76.11 | 81.29 | 66.70 |
| Sciurus vulgaris | 126 | 88 | 19 | 38 | 82.24 | 69.84 | 75.54 | 80.06 | 64.85 |
| Tamiasciurus douglasii | 131 | 94 | 9 | 37 | 91.26 | 71.76 | 80.34 | 83.00 | 68.00 |
| Tamiasciurus hudsonicus | 124 | 103 | 12 | 21 | 89.57 | 83.06 | 86.19 | 88.75 | 74.33 |
| generic_squirrel | 122 | 71 | 13 | 51 | 84.52 | 58.20 | 68.93 | 71.29 | 56.79 |

### YOLO26 Medium (accuracy head)

| Class | Support | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50–95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Callosciurus erythraeus | 121 | 70 | 17 | 51 | 80.46 | 57.85 | 67.31 | 62.49 | 48.73 |
| Sciurus aureogaster | 123 | 78 | 30 | 45 | 72.22 | 63.41 | 67.53 | 64.59 | 47.71 |
| Sciurus carolinensis | 157 | 101 | 28 | 56 | 78.29 | 64.33 | 70.63 | 68.83 | 52.58 |
| Sciurus granatensis | 124 | 88 | 15 | 36 | 85.44 | 70.97 | 77.53 | 76.99 | 57.99 |
| Sciurus griseus | 132 | 92 | 24 | 40 | 79.31 | 69.70 | 74.19 | 71.92 | 53.56 |
| Sciurus lis | 94 | 68 | 11 | 26 | 86.08 | 72.34 | 78.61 | 79.13 | 52.21 |
| Sciurus niger | 126 | 85 | 31 | 41 | 73.28 | 67.46 | 70.25 | 75.22 | 53.25 |
| Sciurus vulgaris | 126 | 84 | 18 | 42 | 82.35 | 66.67 | 73.68 | 73.25 | 54.46 |
| Tamiasciurus douglasii | 131 | 95 | 13 | 36 | 87.96 | 72.52 | 79.50 | 78.40 | 63.54 |
| Tamiasciurus hudsonicus | 124 | 87 | 7 | 37 | 92.55 | 70.16 | 79.82 | 79.11 | 61.67 |
| generic_squirrel | 122 | 56 | 16 | 66 | 77.78 | 45.90 | 57.73 | 56.51 | 38.70 |

### Original YOLOv5 Medium

| Class | Support | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50–95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Callosciurus erythraeus | 121 | 61 | 11 | 60 | 84.72 | 50.41 | 63.21 | 56.25 | 42.44 |
| Sciurus aureogaster | 123 | 73 | 20 | 50 | 78.49 | 59.35 | 67.59 | 63.48 | 46.26 |
| Sciurus carolinensis | 157 | 93 | 40 | 64 | 69.92 | 59.24 | 64.14 | 62.72 | 48.01 |
| Sciurus granatensis | 124 | 88 | 17 | 36 | 83.81 | 70.97 | 76.86 | 74.75 | 55.87 |
| Sciurus griseus | 132 | 92 | 19 | 40 | 82.88 | 69.70 | 75.72 | 75.05 | 55.51 |
| Sciurus lis | 94 | 67 | 10 | 27 | 87.01 | 71.28 | 78.36 | 79.88 | 49.95 |
| Sciurus niger | 126 | 79 | 32 | 47 | 71.17 | 62.70 | 66.67 | 64.64 | 48.55 |
| Sciurus vulgaris | 126 | 78 | 25 | 48 | 75.73 | 61.90 | 68.12 | 66.97 | 50.37 |
| Tamiasciurus douglasii | 131 | 100 | 12 | 31 | 89.29 | 76.34 | 82.30 | 81.76 | 62.69 |
| Tamiasciurus hudsonicus | 124 | 83 | 12 | 41 | 87.37 | 66.94 | 75.80 | 77.23 | 58.57 |
| generic_squirrel | 122 | 54 | 14 | 68 | 79.41 | 44.26 | 56.84 | 54.83 | 36.09 |

## Error breakdown at the frozen thresholds

| Error type | RF-DETR Medium | YOLO26 accuracy head | Original YOLOv5 Medium |
| --- | --- | --- | --- |
| Localized objects assigned the wrong class | 120 | 112 | 118 |
| Ground-truth objects without a localization match | 286 | 364 | 394 |
| Predictions without a ground-truth localization match | 62 | 98 | 94 |

A species confusion contributes to both FP and FN. Thus `FP = wrong-class matches + unmatched predictions` and `FN = wrong-class matches + unmatched ground truth`. RF-DETR’s recall advantage comes mainly from fewer unmatched ground-truth objects; it does not have the fewest wrong-class matches.

**RF-DETR Medium:** `Sciurus aureogaster` → `Sciurus carolinensis` (7); `Sciurus niger` → `Sciurus carolinensis` (6); `Sciurus griseus` → `Sciurus carolinensis` (6). Directions are ground truth → prediction.

**YOLO26 Medium (accuracy head):** `Sciurus griseus` → `Sciurus carolinensis` (10); `Tamiasciurus hudsonicus` → `Sciurus vulgaris` (6); `Sciurus niger` → `Sciurus griseus` (5). Directions are ground truth → prediction.

**Original YOLOv5 Medium:** `Sciurus griseus` → `Sciurus carolinensis` (9); `Tamiasciurus hudsonicus` → `Sciurus carolinensis` (6); `Sciurus niger` → `Sciurus carolinensis` (5). Directions are ground truth → prediction.

## Validation-to-test change

| Model | Validation mAP50–95 | Test mAP50–95 | Change (pp) | Validation micro F1 | Test micro F1 | Change (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| RF-DETR Medium | 67.11 | 66.11 | -1.01 | 77.83 | 76.81 | -1.01 |
| YOLO26 Medium (accuracy head) | 52.84 | 53.13 | +0.29 | 72.31 | 72.49 | +0.18 |
| Original YOLOv5 Medium | 51.69 | 50.39 | -1.30 | 71.77 | 70.57 | -1.20 |

The primary ranking is consistent across validation and test. These are descriptive changes from one split and one seed; the saved artifacts do not provide confidence intervals or repeated-run uncertainty.

## YOLO26 secondary NMS-free head

This is an alternate head of the **same YOLO26 best checkpoint**, with its own validation-selected threshold of **0.40**. It is included separately from the primary accuracy head.

| Metric | Accuracy head | NMS-free head | Change (pp unless counts) |
| --- | --- | --- | --- |
| mAP50–95 | 53.13 | 54.84 | +1.71 |
| AP50 | 71.49 | 73.64 | +2.15 |
| AP75 | 58.69 | 60.25 | +1.57 |
| Micro precision | 81.15 | 76.58 | -4.57 |
| Micro recall | 65.51 | 67.75 | +2.25 |
| Micro F1 | 72.49 | 71.90 | -0.60 |
| Macro precision | 81.43 | 76.87 | -4.56 |
| Macro recall | 65.57 | 67.88 | +2.31 |
| Macro F1 | 72.44 | 71.88 | -0.56 |
| Mean matched IoU | 87.88 | 87.44 | -0.44 |
| AP small | 2.71 | 2.94 | +0.23 |
| AP medium | 24.66 | 26.05 | +1.39 |
| AP large | 56.55 | 58.31 | +1.76 |
| AR1 | 58.79 | 64.69 | +5.90 |
| AR10 | 65.46 | 75.36 | +9.90 |
| AR100 | 65.80 | 75.76 | +9.96 |
| AR small | 7.38 | 34.05 | +26.67 |
| AR medium | 37.96 | 50.29 | +12.33 |
| AR large | 68.76 | 78.36 | +9.60 |
| TP | 904 | 935 | +31 objects |
| FP | 210 | 286 | +76 objects |
| FN | 476 | 445 | -31 objects |

The NMS-free head gains **1.71 points mAP50–95** and **2.25 points recall**, with **4.57 points lower precision** and **0.60 points lower micro F1**. It yields 31 additional true positives and 76 additional false positives at its separately frozen threshold.

### NMS-free per-class results

| Class | Support | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50–95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Callosciurus erythraeus | 121 | 75 | 23 | 46 | 76.53 | 61.98 | 68.49 | 66.66 | 51.61 |
| Sciurus aureogaster | 123 | 80 | 35 | 43 | 69.57 | 65.04 | 67.23 | 67.00 | 50.37 |
| Sciurus carolinensis | 157 | 98 | 33 | 59 | 74.81 | 62.42 | 68.06 | 71.89 | 54.39 |
| Sciurus granatensis | 124 | 92 | 19 | 32 | 82.88 | 74.19 | 78.30 | 80.32 | 59.64 |
| Sciurus griseus | 132 | 94 | 40 | 38 | 70.15 | 71.21 | 70.68 | 77.03 | 57.04 |
| Sciurus lis | 94 | 68 | 21 | 26 | 76.40 | 72.34 | 74.32 | 76.11 | 50.37 |
| Sciurus niger | 126 | 88 | 33 | 38 | 72.73 | 69.84 | 71.26 | 76.63 | 55.36 |
| Sciurus vulgaris | 126 | 88 | 29 | 38 | 75.21 | 69.84 | 72.43 | 73.52 | 53.90 |
| Tamiasciurus douglasii | 131 | 99 | 18 | 32 | 84.62 | 75.57 | 79.84 | 81.22 | 66.38 |
| Tamiasciurus hudsonicus | 124 | 89 | 20 | 35 | 81.65 | 71.77 | 76.39 | 80.83 | 63.25 |
| generic_squirrel | 122 | 64 | 15 | 58 | 81.01 | 52.46 | 63.68 | 58.86 | 40.88 |

## Inference latency and resources

### Comparable saved YOLO benchmarks

All six saved YOLO CPU/GPU records have identical benchmark protocols: the same 200 held-out test image IDs, seed 42, 20 warmup calls, batch size 1, FP32, one CPU thread, TF32 disabled, native PyTorch inference with standard backend fusion and no compilation. Timing includes preprocessing, inference, and postprocessing from decoded RGB input; image decoding is excluded. Hardware: **NVIDIA GeForce RTX 5060 Ti** GPU and **Intel Xeon W-2145 @ 3.70 GHz** CPU. FPS is `1,000 / mean latency`; it is sequential batch-one throughput.

| Model / head | Device | Mean (ms) | Median (ms) | p95 (ms) | FPS | Peak inference VRAM (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Medium (accuracy head) | CUDA | 23.91 | 23.97 | 24.61 | 41.82 | 0.420 |
| YOLO26 Medium (accuracy head) | CPU | 116.48 | 116.05 | 127.25 | 8.58 | — |
| Original YOLOv5 Medium | CUDA | 18.61 | 18.67 | 19.17 | 53.73 | 0.224 |
| Original YOLOv5 Medium | CPU | 400.45 | 401.27 | 409.08 | 2.50 | — |
| YOLO26 Medium (NMS-free head) | CUDA | 23.56 | 23.56 | 24.30 | 42.44 | 0.420 |
| YOLO26 Medium (NMS-free head) | CPU | 127.92 | 126.11 | 141.53 | 7.82 | — |

YOLOv5’s GPU mean latency is **18.61 ms**, versus YOLO26’s **23.91 ms**. On CPU, YOLO26 averages **116.48 ms**, versus YOLOv5’s **400.45 ms**. The NMS-free head averages **23.56 ms** on GPU and **127.92 ms** on CPU; it offers only a small GPU latency reduction in this saved benchmark and is slower on CPU.

### RF-DETR archived evaluation timing

| Device | Timed test images | Mean (ms) | p95 (ms) |
| --- | --- | --- | --- |
| CUDA | 1284 | 15.91 | 16.69 |

The RF-DETR notebook times synchronized `model.predict` calls across the full test split after one warmup, with image decoding outside the timer and prediction floor 0.0001. Its saved timing does not document the same FP32/TF32/runtime controls as the YOLO protocol. **Do not directly rank RF-DETR’s 15.91 ms against the YOLO means.** No completed common three-model benchmark exists under `output_model_comparison/runs/` at report generation; a common RF-DETR benchmark would be needed to establish a three-model speed leader.

### Training and artifact context

| Model | Input size (px) | Epochs completed | Epoch limit | Training batch | Summed epoch time (h) | Peak recorded training VRAM (GiB) | Best checkpoint (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF-DETR Medium | 576 | 50 | 50 | 8 × 2 | 5.27 | 4.07 | 127.70 |
| YOLO26 Medium (accuracy head) | 640 | 140 | 200 | 16 | 3.61 | 8.45 | 42.00 |
| Original YOLOv5 Medium | 640 | 177 | 200 | 16 | 4.47 | 4.98 | 40.23 |

Epoch time is the sum of recorded epoch durations, excluding setup, probes, evaluation, benchmarking, and other overhead. YOLO26 and YOLOv5 used early-stopping patience 30. RF-DETR’s batch is 8 with gradient accumulation 2 (effective batch 16). Recorded YOLO inference parameter counts are **21,789,850** for YOLO26 and **20,893,344** for YOLOv5; a comparable RF-DETR inference count was not saved. Checkpoint sizes reflect serialization and saved precision, not just model capacity.

## Interpretation and limitations

- This compares complete training recipes on one dataset and seed 42. RF-DETR used 576-pixel input and class-aware training sampling for 50 epochs; YOLO used 640-pixel input, natural sampling, and longer training budgets. Pretraining, augmentation, optimizers, capacity, and postprocessing differ. The observed ranking does not isolate architecture effects.
- The audited split groups known observation and perceptual-risk relationships and removes reviewed duplicates. This reduces known leakage; it does not prove independence from every related scene, photographer, or pretraining source. Validation and test retain their natural distributions.
- Large boxes dominate this held-out set, and only 14 small boxes are present. Scores on this split do not establish performance on a new geographic source, deployment camera, or edge device.
- RF-DETR’s archived test predictions contain **91 low-confidence outputs** for unmapped output index 11 / category ID 12 (`cls_11`), outside the eleven-class taxonomy. Their highest score is **0.02314**, below the frozen 0.65 threshold. These entries were excluded when deriving per-class reports; all twelve original COCO scores and all original operating scores, including mean matched IoU, reproduced within 1e-9.
- Final detection metrics use the shared COCO evaluator. Native training AP implementations and loss scales differ, so training loss values are not used for cross-model quality ranking. No statistical significance claim is made.

## Reproducibility and source artifacts

The final runs are:

| Model | Run | Best checkpoint | Selection criterion | Frozen threshold |
| --- | --- | --- | --- | --- |
| RF-DETR Medium | `output/runs/20260926T153802Z-v2-f35e9e` | `checkpoint_best_total.pth` | Validation mAP50–95 | 0.65 |
| YOLO26 Medium (accuracy head) | `output_yolo26/runs/20260928T202126Z-e14d4f` | `native/weights/best.pt` | Validation mAP50–95 | 0.62 |
| Original YOLOv5 Medium | `output_yolov5/runs/20260929T154035Z-d7f2b6` | `native/weights/best.pt` | 0.1 × validation AP50 + 0.9 × validation mAP50–95 | 0.61 |

Dataset fingerprint: `96af70a0474fdebe578301d8703c0da2bc0e9b718ebab86a580c7ec883bc5b0b`.

Test annotation SHA-256: `d9926aaaa3b39fa452da5de4a0585bf221ef4dc6780088b805bb99d1f43dde85`.

| Model | Best checkpoint SHA-256 |
| --- | --- |
| RF-DETR Medium | `50502f65c6316dd67e996b6fce3b821dd4056e8208ba1cc4eca8faecb2488c95` |
| YOLO26 Medium (accuracy head) | `ea72f3f84f810db85d652d7b9a8942b0174c9caa0b9d4593b16cb81503f58f42` |
| Original YOLOv5 Medium | `bea84a654c15db4bd6a02eb978fed10724b9f3659d582c2d224c8a20a3eadb9c` |

All three checkpoint files were hashed and matched their training-completion records. Dataset manifests and current annotation hashes were verified. Archived predictions cover the full test image-ID set. The shared evaluator reproduced all twelve saved COCO AP/AR scores and all seven operating values for each primary model and the YOLO26 NMS-free head within 1e-9. All saved YOLO per-class metrics were also reproduced; RF-DETR’s per-class and macro metrics were derived using that evaluator.

**RF-DETR Medium sources:** [test metrics](/home/ss3813/github/ML/CSCI-635-ML-Project/output/runs/20260926T153802Z-v2-f35e9e/reports/test/metrics.json), [validation metrics](/home/ss3813/github/ML/CSCI-635-ML-Project/output/runs/20260926T153802Z-v2-f35e9e/reports/valid/metrics.json), [archived test predictions](/home/ss3813/github/ML/CSCI-635-ML-Project/output/runs/20260926T153802Z-v2-f35e9e/reports/test/predictions.json), [frozen threshold](/home/ss3813/github/ML/CSCI-635-ML-Project/output/runs/20260926T153802Z-v2-f35e9e/decision_threshold.json), [training completion](/home/ss3813/github/ML/CSCI-635-ML-Project/output/runs/20260926T153802Z-v2-f35e9e/training_complete.json), [training history](/home/ss3813/github/ML/CSCI-635-ML-Project/output/runs/20260926T153802Z-v2-f35e9e/reports/training_history.csv).

**YOLO26 Medium (accuracy head) sources:** [test metrics](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolo26/runs/20260928T202126Z-e14d4f/reports/test/metrics.json), [validation metrics](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolo26/runs/20260928T202126Z-e14d4f/reports/valid/metrics.json), [archived test predictions](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolo26/runs/20260928T202126Z-e14d4f/reports/test/predictions.json), [frozen threshold](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolo26/runs/20260928T202126Z-e14d4f/decision_threshold.json), [training completion](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolo26/runs/20260928T202126Z-e14d4f/training_complete.json), [training history](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolo26/runs/20260928T202126Z-e14d4f/reports/training_history.csv).

**Original YOLOv5 Medium sources:** [test metrics](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolov5/runs/20260929T154035Z-d7f2b6/reports/test/metrics.json), [validation metrics](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolov5/runs/20260929T154035Z-d7f2b6/reports/valid/metrics.json), [archived test predictions](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolov5/runs/20260929T154035Z-d7f2b6/reports/test/predictions.json), [frozen threshold](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolov5/runs/20260929T154035Z-d7f2b6/decision_threshold.json), [training completion](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolov5/runs/20260929T154035Z-d7f2b6/training_complete.json), [training history](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolov5/runs/20260929T154035Z-d7f2b6/reports/training_history.csv).

**Additional sources:** [YOLO26 NMS-free test metrics](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolo26/runs/20260928T202126Z-e14d4f/reports/end2end/test/metrics.json), [YOLO26 benchmarks](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolo26/runs/20260928T202126Z-e14d4f/benchmarks/accuracy/cuda.json), [YOLOv5 benchmarks](/home/ss3813/github/ML/CSCI-635-ML-Project/output_yolov5/runs/20260929T154035Z-d7f2b6/benchmarks/accuracy/cuda.json), [audited dataset manifest](/home/ss3813/github/ML/CSCI-635-ML-Project/data/squirrel-v2-audited-coco/.complete.json), [test annotations](/home/ss3813/github/ML/CSCI-635-ML-Project/data/squirrel-v2-audited-coco/test/_annotations.coco.json), [common metric implementation](/home/ss3813/github/ML/CSCI-635-ML-Project/src/csci_635_ml_project/baseline_reports.py).

A [machine-readable verification snapshot](/home/ss3813/github/ML/CSCI-635-ML-Project/reports/final_held_out_test_metrics.json) accompanies this report, including full-precision metrics, all per-class rows, confusion matrices (ground-truth rows × predicted columns, with background last), and excluded RF-DETR output counts. Original run artifacts were left unchanged. Source links point to this workspace; run directories are Git-ignored and should be retained with the report for full reproducibility.
