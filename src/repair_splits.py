"""Dataset Repair & Leakage Prevention Pipeline.

Merges train, valid, and test splits into a unified dataset, deduplicates
exact identical copies ("ditto duplicates"), clusters near-identical and burst
images by perceptual hash similarity (pHash distance <= threshold), and performs
a balanced, burst-isolated split.

Key guarantees:
1. Zero cross-split leakage: Connected components of similar images (distance <= 4)
   are strictly confined to a single split (never split across train, val, test).
2. Burst diversity protection: Large bursts (> 2 images) are routed to `train` so
   no single burst can dominate or skew the `valid` or `test` evaluation.
3. Ditto deduplication: Exact duplicate copies (pHash distance 0) are deduplicated.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
import imagehash


def deduplicate_ditto_copies(image_records: List[Dict]) -> Tuple[List[Dict], int]:
    """Remove exact duplicate copies (identical pHash distance 0)."""
    seen_hashes = {}
    unique_records = []
    removed_count = 0
    for r in image_records:
        h = r['phash']
        if h in seen_hashes:
            removed_count += 1
        else:
            seen_hashes[h] = r
            unique_records.append(r)
    return unique_records, removed_count


def cluster_similar_images(records: List[Dict], max_distance: int = 4) -> List[List[int]]:
    """Group images into connected components where pHash Hamming distance <= max_distance."""
    hashes = np.array([int(r['phash'], 16) for r in records], dtype=np.uint64)
    popcount = np.array([int(x).bit_count() for x in range(256)], dtype=np.uint8)
    n = len(records)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    chunk_size = 500
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        h_chunk = hashes[i:end_i, None]
        for j in range(i, n, chunk_size):
            end_j = min(j + chunk_size, n)
            h_other = hashes[None, j:end_j]
            xor = (h_chunk ^ h_other).view(np.uint8).reshape(-1, 8)
            dists = popcount[xor].sum(axis=1).reshape(end_i - i, end_j - j)
            if i == j:
                mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
            else:
                mask = np.ones_like(dists, dtype=bool)
            r_idx, c_idx = np.where(mask & (dists <= max_distance))
            for r, c in zip(r_idx, c_idx):
                G.add_edge(i + r, j + c)

    return [list(c) for c in nx.connected_components(G)]


def assign_clusters_to_splits(
    clusters: List[List[int]],
    val_target: int = 380,
    test_target: int = 380,
    max_eval_cluster_size: int = 2,
    seed: int = 42
) -> Dict[str, List[List[int]]]:
    """Assign clusters to train, val, and test splits with burst-skew protection.

    Clusters larger than max_eval_cluster_size are allocated to `train` to prevent
    burst sequences from skewing the evaluation sets. Val and test are sampled from
    small, diverse clusters and singletons.
    """
    rng = random.Random(seed)

    large_clusters = [c for c in clusters if len(c) > max_eval_cluster_size]
    small_clusters = [c for c in clusters if len(c) <= max_eval_cluster_size]
    rng.shuffle(small_clusters)

    val_clusters: List[List[int]] = []
    test_clusters: List[List[int]] = []
    train_clusters: List[List[int]] = list(large_clusters)

    val_count = 0
    test_count = 0

    for c in small_clusters:
        if val_count < val_target:
            val_clusters.append(c)
            val_count += len(c)
        elif test_count < test_target:
            test_clusters.append(c)
            test_count += len(c)
        else:
            train_clusters.append(c)

    return {
        'train': train_clusters,
        'valid': val_clusters,
        'test': test_clusters
    }


def repair_dataset_splits(
    raw_dir: Path,
    clean_dir: Path,
    val_target: int = 380,
    test_target: int = 380,
    max_distance: int = 4,
    max_eval_cluster_size: int = 2,
    seed: int = 42
) -> Dict[str, int]:
    """Execute the end-to-end dataset repair and split generation."""
    raw_dir = Path(raw_dir)
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all original COCO documents and images
    docs = {}
    ann_map = {}
    image_records = []

    for split in ['train', 'valid', 'test']:
        doc_path = raw_dir / split / '_annotations.coco.json'
        if not doc_path.exists():
            raise FileNotFoundError(f'Missing {doc_path}')
        doc = json.loads(doc_path.read_text())
        docs[split] = doc

        for ann in doc.get('annotations', []):
            ann_map.setdefault((split, ann['image_id']), []).append(ann)

        for img in doc.get('images', []):
            img_path = raw_dir / split / img['file_name']
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                im.load()
                rgb = im.convert('RGB')
                w, h = rgb.size
                phash = str(imagehash.phash(rgb))
                pixel_hash = hashlib.sha256(f'{w}x{h}'.encode() + rgb.tobytes()).hexdigest()

            image_records.append({
                'orig_split': split,
                'orig_id': img['id'],
                'file_name': img['file_name'],
                'path': str(img_path),
                'width': w,
                'height': h,
                'phash': phash,
                'pixel_hash': pixel_hash
            })

    # 2. Deduplicate exact duplicate copies
    unique_records, removed_dups = deduplicate_ditto_copies(image_records)

    # 3. Cluster similar images (pHash distance <= max_distance)
    clusters = cluster_similar_images(unique_records, max_distance=max_distance)

    # 4. Partition clusters into train, valid, test
    splits = assign_clusters_to_splits(
        clusters,
        val_target=val_target,
        test_target=test_target,
        max_eval_cluster_size=max_eval_cluster_size,
        seed=seed
    )

    # 5. Write out clean COCO splits and hard-link images
    stats = {
        'initial_images': len(image_records),
        'deduplicated_copies': removed_dups,
        'unique_images': len(unique_records),
        'total_clusters': len(clusters)
    }

    for split_name, split_clusters in splits.items():
        split_dir = clean_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        new_images = []
        new_annotations = []
        new_img_id = 1
        new_ann_id = 1

        for c in split_clusters:
            for idx in c:
                row = unique_records[idx]
                src_path = Path(row['path'])
                dst_path = split_dir / row['file_name']

                if dst_path.exists():
                    dst_path.unlink()
                try:
                    os.link(src_path, dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)

                new_images.append({
                    'id': new_img_id,
                    'file_name': row['file_name'],
                    'width': row['width'],
                    'height': row['height'],
                    'license': 1
                })

                for ann in ann_map.get((row['orig_split'], row['orig_id']), []):
                    new_ann = dict(ann)
                    new_ann['id'] = new_ann_id
                    new_ann['image_id'] = new_img_id
                    new_annotations.append(new_ann)
                    new_ann_id += 1

                new_img_id += 1

        doc = {
            'info': docs['train'].get('info', {}),
            'licenses': docs['train'].get('licenses', []),
            'categories': docs['train'].get('categories', []),
            'images': new_images,
            'annotations': new_annotations
        }
        (split_dir / '_annotations.coco.json').write_text(json.dumps(doc, indent=2))
        stats[f'{split_name}_images'] = len(new_images)
        stats[f'{split_name}_annotations'] = len(new_annotations)
        stats[f'{split_name}_clusters'] = len(split_clusters)

    # 6. Save dataset manifest marker
    manifest = {
        'version': 6,
        'workspace': 'root-and-nut',
        'project': 'squirrel-re-id-training-v1-fzpbr',
        'format': 'coco',
        'license': 'CC BY 4.0',
        'source': 'Repaired leak-free splits with burst grouping and duplicate removal',
        'annotation_hashes': {
            s: hashlib.sha256((clean_dir / s / '_annotations.coco.json').read_bytes()).hexdigest()
            for s in ['train', 'valid', 'test']
        }
    }
    (clean_dir / '.complete.json').write_text(json.dumps(manifest, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description='Repair dataset splits to guarantee 0 cross-split leakage.')
    parser.add_argument('--raw_dir', type=Path, default=Path('data/squirrel-v6-coco'))
    parser.add_argument('--clean_dir', type=Path, default=Path('data/squirrel-v6-clean'))
    parser.add_argument('--val_target', type=int, default=380)
    parser.add_argument('--test_target', type=int, default=380)
    parser.add_argument('--max_distance', type=int, default=4)
    parser.add_argument('--max_eval_cluster_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    stats = repair_dataset_splits(
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
        val_target=args.val_target,
        test_target=args.test_target,
        max_distance=args.max_distance,
        max_eval_cluster_size=args.max_eval_cluster_size,
        seed=args.seed
    )
    print('Repair complete:')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
