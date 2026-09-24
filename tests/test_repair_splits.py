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
            {'file_name': 'a.jpg', 'phash': '0000000000000000'},
            {'file_name': 'b.jpg', 'phash': '0000000000000000'},  # Duplicate
            {'file_name': 'c.jpg', 'phash': '0000000000000001'},
            {'file_name': 'd.jpg', 'phash': 'ffffffffffffffff'}
        ]
        unique, removed = deduplicate_ditto_copies(records)
        self.assertEqual(len(unique), 3)
        self.assertEqual(removed, 1)
        self.assertEqual([r['file_name'] for r in unique], ['a.jpg', 'c.jpg', 'd.jpg'])

    def test_cluster_similar_images(self):
        records = [
            {'file_name': '0.jpg', 'phash': '0000000000000000'},
            {'file_name': '1.jpg', 'phash': '0000000000000001'},  # dist=1 to 0
            {'file_name': '2.jpg', 'phash': '0000000000000003'},  # dist=1 to 1
            {'file_name': '3.jpg', 'phash': '00000000000000ff'},  # dist=6 to 2
            {'file_name': '4.jpg', 'phash': 'ffffffffffffffff'}   # dist=64 to 0
        ]
        clusters = cluster_similar_images(records, max_distance=2)
        # 0, 1, 2 should be in one connected component
        cluster_sets = [set(c) for c in clusters]
        self.assertTrue(any(s == {0, 1, 2} for s in cluster_sets))
        self.assertTrue(any(s == {3} for s in cluster_sets))
        self.assertTrue(any(s == {4} for s in cluster_sets))

    def test_assign_clusters_burst_protection_and_zero_leakage(self):
        # Create clusters: one large burst of 10 images, and 20 singletons
        large_cluster = list(range(10))
        singletons = [[i] for i in range(10, 30)]
        all_clusters = [large_cluster] + singletons

        splits = assign_clusters_to_splits(
            all_clusters,
            val_target=4,
            test_target=4,
            max_eval_cluster_size=2,
            seed=42
        )

        train_clusters = splits['train']
        val_clusters = splits['valid']
        test_clusters = splits['test']

        # 1. Large burst (> 2) MUST be in train
        self.assertIn(large_cluster, train_clusters)
        self.assertNotIn(large_cluster, val_clusters)
        self.assertNotIn(large_cluster, test_clusters)

        # 2. Strict isolation: no image appears in more than one split
        train_images = {idx for c in train_clusters for idx in c}
        val_images = {idx for c in val_clusters for idx in c}
        test_images = {idx for c in test_clusters for idx in c}

        self.assertEqual(len(train_images & val_images), 0)
        self.assertEqual(len(train_images & test_images), 0)
        self.assertEqual(len(val_images & test_images), 0)

        # 3. All images accounted for
        self.assertEqual(len(train_images | val_images | test_images), 30)

        # 4. Target counts met
        self.assertEqual(len(val_images), 4)
        self.assertEqual(len(test_images), 4)
        self.assertEqual(len(train_images), 22)


if __name__ == '__main__':
    unittest.main()
