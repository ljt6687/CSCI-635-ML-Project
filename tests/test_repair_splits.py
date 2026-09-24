import unittest
import numpy as np
from src.repair_splits import (
    deduplicate_ditto_copies,
    cluster_similar_images,
    assign_clusters_to_splits
)


class TestRepairSplits(unittest.TestCase):
    def test_deduplicate_ditto_copies(self):
        records = [
            {'file_name': 'a.jpg', 'source': 'img_1', 'phash': '0000000000000000'},
            {'file_name': 'b.jpg', 'source': 'img_1', 'phash': '0000000000000000'},  # Duplicate of a
            {'file_name': 'c.jpg', 'source': 'img_2', 'phash': '0000000000000000'},  # Same hash, different source
            {'file_name': 'd.jpg', 'source': 'img_3', 'phash': 'ffffffffffffffff'}
        ]
        unique, removed = deduplicate_ditto_copies(records)
        self.assertEqual(len(unique), 3)
        self.assertEqual(removed, 1)
        self.assertEqual([r['file_name'] for r in unique], ['a.jpg', 'c.jpg', 'd.jpg'])

    def test_cluster_similar_images_by_source_and_distance(self):
        records = [
            # Same source 'seqA' but different phash (e.g. 90-deg rotation, dist=32)
            {'file_name': 'seqA_0.jpg', 'source': 'seqA', 'phash': '0000000000000000'},
            {'file_name': 'seqA_rot.jpg', 'source': 'seqA', 'phash': 'ffff000000000000'},  # dist=16
            # Different source 'seqB', near distance to seqA_0 (dist=1)
            {'file_name': 'seqB_0.jpg', 'source': 'seqB', 'phash': '0000000000000001'},
            # Completely different image
            {'file_name': 'seqC_0.jpg', 'source': 'seqC', 'phash': 'ffffffffffffffff'}
        ]
        clusters = cluster_similar_images(records, max_distance=2)
        cluster_sets = [set(c) for c in clusters]
        # seqA_0, seqA_rot, and seqB_0 should all be in one connected component
        self.assertTrue(any(s == {0, 1, 2} for s in cluster_sets))
        self.assertTrue(any(s == {3} for s in cluster_sets))

    def test_assign_clusters_burst_protection_and_zero_leakage(self):
        # Create clusters: large bursts (> 4 images) and small clusters
        large_cluster_1 = list(range(10))
        large_cluster_2 = list(range(10, 18))
        small_clusters = [[i, i + 1] for i in range(18, 100, 2)]  # 41 clusters of size 2 (82 images)
        all_clusters = [large_cluster_1, large_cluster_2] + small_clusters
        total_images = sum(len(c) for c in all_clusters)  # 100 images

        splits = assign_clusters_to_splits(
            all_clusters,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            max_eval_cluster_size=4,
            seed=42
        )

        train_clusters = splits['train']
        val_clusters = splits['valid']
        test_clusters = splits['test']

        # 1. Large bursts (> 4) MUST be in train
        self.assertIn(large_cluster_1, train_clusters)
        self.assertIn(large_cluster_2, train_clusters)
        self.assertNotIn(large_cluster_1, val_clusters)
        self.assertNotIn(large_cluster_2, test_clusters)

        # 2. Strict isolation: no image appears in more than one split
        train_images = {idx for c in train_clusters for idx in c}
        val_images = {idx for c in val_clusters for idx in c}
        test_images = {idx for c in test_clusters for idx in c}

        self.assertEqual(len(train_images & val_images), 0)
        self.assertEqual(len(train_images & test_images), 0)
        self.assertEqual(len(val_images & test_images), 0)

        # 3. All images accounted for
        self.assertEqual(len(train_images | val_images | test_images), total_images)

        # 4. Valid and test are close to 15% (15-16 images each), train has remainder (~70%)
        self.assertGreaterEqual(len(val_images), 15)
        self.assertGreaterEqual(len(test_images), 15)
        self.assertGreaterEqual(len(train_images), 68)


if __name__ == '__main__':
    unittest.main()

