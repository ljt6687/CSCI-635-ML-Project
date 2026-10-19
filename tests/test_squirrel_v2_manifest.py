"""Focused contract tests for deterministic squirrel v2 manifest builder."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts import build_squirrel_v2_manifest as builder


class ManifestBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        # 1. Pilot iNaturalist directory and metadata
        self.inat_dir = self.root / "data" / "squirrel_sources" / "inaturalist"
        self.inat_dir.mkdir(parents=True)
        self.meta_dir = self.root / "research" / "squirrel_expansion" / "metadata"
        self.meta_dir.mkdir(parents=True)

        self.obs_file = self.meta_dir / "inat_observations.jsonl"
        obs_records = [
            {
                "observation_id": 101,
                "scientific_name": "Sciurus carolinensis",
                "common_name": "Eastern Gray Squirrel",
                "observation_url": "https://www.inaturalist.org/observations/101",
                "photos": [
                    {
                        "photo_id": 201,
                        "license": "cc-by",
                        "attribution": "(c) Author One, CC BY",
                        "original_url": "https://inat.org/201.jpg",
                    }
                ],
            },
            {
                "observation_id": 102,
                "scientific_name": "Glaucomys volans",  # Should map to generic_squirrel
                "common_name": "Southern Flying Squirrel",
                "observation_url": "https://www.inaturalist.org/observations/102",
                "photos": [
                    {
                        "photo_id": 202,
                        "license": "cc0",
                        "attribution": "Public Domain",
                        "original_url": "https://inat.org/202.jpg",
                    }
                ],
            },
        ]
        self.obs_file.write_text("\n".join(json.dumps(r) for r in obs_records) + "\n")

        self.inat_manifest = self.inat_dir / "manifest.jsonl"
        manifest_records = [
            {
                "observation_id": 101,
                "photo_id": 201,
                "scientific_name": "Sciurus carolinensis",
                "filename": "101_201.jpg",
                "observation_url": "https://www.inaturalist.org/observations/101",
                "photo_license": "cc-by",
                "attribution": "(c) Author One, CC BY",
            },
            {
                "observation_id": 102,
                "photo_id": 202,
                "scientific_name": "Glaucomys volans",
                "filename": "102_202.jpg",
                "observation_url": "https://www.inaturalist.org/observations/102",
                "photo_license": "cc0",
                "attribution": "Public Domain",
            },
        ]
        self.inat_manifest.write_text("\n".join(json.dumps(r) for r in manifest_records) + "\n")
        Image.new("RGB", (64, 48), "blue").save(self.inat_dir / "101_201.jpg")
        Image.new("RGB", (64, 48), "green").save(self.inat_dir / "102_202.jpg")

        # 2. Bulk iNaturalist directory and metadata
        self.bulk_dir = self.root / "data" / "squirrel_sources" / "inaturalist_bulk"
        self.bulk_dir.mkdir(parents=True)
        self.scale_meta_dir = self.root / "research" / "squirrel_scale" / "metadata" / "images"
        self.scale_meta_dir.mkdir(parents=True)

        self.bulk_obs_file = self.scale_meta_dir / "observations.jsonl"
        bulk_obs_records = [
            {
                "observation_id": 301,
                "class": "Sciurus lis",
                "scientific_name": "Sciurus lis",
                "photos": [{"photo_id": 401, "license": "cc-by", "attribution": "(c) User, CC BY", "original_url": "https://bulk.org/401.jpg"}],
            },
            {
                "observation_id": 302,
                "class": "squirrel_generic",  # Normalized class
                "scientific_name": "Sciurus aberti",  # Specific raw species
                "photos": [{"photo_id": 402, "license": "cc0", "attribution": "CC0", "original_url": "https://bulk.org/402.jpg"}],
            },
            {
                "observation_id": 303,
                "class": "Sciurus niger",
                "scientific_name": "Sciurus niger cinereus",  # Subspecies
                "photos": [{"photo_id": 403, "license": "cc-by", "attribution": "CC BY", "original_url": "https://bulk.org/403.jpg"}],
            },
        ]
        self.bulk_obs_file.write_text("\n".join(json.dumps(r) for r in bulk_obs_records) + "\n")

        self.bulk_manifest = self.scale_meta_dir / "manifest.jsonl"
        bulk_manifest_records = [
            # 1. Normal still bulk image
            {
                "observation_id": 301,
                "photo_id": 401,
                "class": "Sciurus lis",
                "scientific_name": "Sciurus lis",
                "path": "data/squirrel_sources/inaturalist_bulk/301_401.jpg",
                "format": "JPEG",
                "width": 800,
                "height": 600,
                "sha256": "fake_sha_301",
                "photo_license": "cc-by",
                "source": "new_bulk",
            },
            # 2. Generic squirrel image with distinct raw species
            {
                "observation_id": 302,
                "photo_id": 402,
                "class": "squirrel_generic",
                "scientific_name": "Sciurus aberti",
                "path": "data/squirrel_sources/inaturalist_bulk/302_402.jpg",
                "format": "JPEG",
                "width": 1024,
                "height": 768,
                "sha256": "fake_sha_302",
                "photo_license": "cc0",
                "source": "new_bulk",
            },
            # 3. Subspecies image (should not fragment category)
            {
                "observation_id": 303,
                "photo_id": 403,
                "class": "Sciurus niger",
                "scientific_name": "Sciurus niger cinereus",
                "path": "data/squirrel_sources/inaturalist_bulk/303_403.jpg",
                "format": "JPEG",
                "width": 640,
                "height": 480,
                "sha256": "fake_sha_303",
                "photo_license": "cc-by",
                "source": "new_bulk",
            },
            # 4. Pilot overlap row (must be skipped to avoid double count)
            {
                "observation_id": 101,
                "photo_id": 201,
                "class": "Sciurus carolinensis",
                "scientific_name": "Sciurus carolinensis",
                "path": "data/squirrel_sources/inaturalist/101_201.jpg",
                "format": "JPEG",
                "width": 64,
                "height": 48,
                "sha256": "fake_sha_pilot",
                "source": "existing_pilot",
            },
            # 5. Missing bulk path (must be reported, not fabricated)
            {
                "observation_id": 999,
                "photo_id": 888,
                "class": "Sciurus vulgaris",
                "scientific_name": "Sciurus vulgaris",
                "path": "data/squirrel_sources/inaturalist_bulk/999_888.jpg",
                "format": "JPEG",
                "width": 500,
                "height": 500,
                "sha256": "fake_sha_missing",
                "source": "new_bulk",
            },
            # 6. Animated GIF original (must be excluded from still manifest)
            {
                "observation_id": 777,
                "photo_id": 666,
                "class": "Sciurus carolinensis",
                "scientific_name": "Sciurus carolinensis",
                "path": "data/squirrel_sources/inaturalist_bulk/777_666.gif",
                "format": "GIF",
                "width": 300,
                "height": 300,
                "sha256": "fake_sha_gif",
                "source": "new_bulk",
            },
        ]
        self.bulk_manifest.write_text("\n".join(json.dumps(r) for r in bulk_manifest_records) + "\n")

        Image.new("RGB", (800, 600), "red").save(self.bulk_dir / "301_401.jpg")
        Image.new("RGB", (1024, 768), "purple").save(self.bulk_dir / "302_402.jpg")
        Image.new("RGB", (640, 480), "yellow").save(self.bulk_dir / "303_403.jpg")
        (self.bulk_dir / "777_666.gif").write_bytes(b"GIF89a...")  # Animated GIF file

        # 3. Harry Baines acquisition and images
        self.hb_raw_dir = self.root / "data" / "squirrel_sources" / "harrybaines_squirrels" / "raw"
        self.hb_raw_dir.mkdir(parents=True)
        hb_acq = self.root / "data" / "squirrel_sources" / "harrybaines_squirrels" / "acquisition.json"
        hb_acq.write_text(json.dumps({
            "source_url": "https://www.kaggle.com/datasets/harrybaines/squirrels",
            "provider_license": "CC0: Public Domain",
        }))
        Image.new("RGB", (100, 100), "red").save(self.hb_raw_dir / "IMG_1001.png")
        Image.new("RGB", (100, 100), "red").save(self.hb_raw_dir / "IMG_1002.png")
        Image.new("RGB", (100, 100), "red").save(self.hb_raw_dir / "IMG_2050.png")

        # 4. Meyer Trailcam acquisition and images
        self.meyer_samples_dir = self.root / "data" / "squirrel_sources" / "meyer_trailcam" / "raw" / "DatasetSamples"
        self.meyer_squirrel_dir = self.meyer_samples_dir / "Squirrel" / "cam1"
        self.meyer_squirrel_dir.mkdir(parents=True)
        self.meyer_coyote_dir = self.meyer_samples_dir / "Coyote" / "cam1"
        self.meyer_coyote_dir.mkdir(parents=True)

        meyer_acq = self.root / "data" / "squirrel_sources" / "meyer_trailcam" / "acquisition.json"
        meyer_acq.write_text(json.dumps({
            "source_url": "https://www.kaggle.com/datasets/jimmeyer645/trailcam-dataset-samples-from-larger-dataset",
            "provider_license": "CDLA-Permissive-1.0",
        }))
        Image.new("RGB", (120, 90), "gray").save(self.meyer_squirrel_dir / "IMG_0001_sd1.JPG")
        Image.new("RGB", (120, 90), "yellow").save(self.meyer_squirrel_dir / "IMG_0001_sd1-rendered.JPG")
        (self.meyer_squirrel_dir / "IMG_0001_sd1.xml").write_text("<annotation><object><name>Squirrel</name></object></annotation>")
        (self.meyer_squirrel_dir / ".DS_Store").write_bytes(b"junk")
        Image.new("RGB", (120, 90), "brown").save(self.meyer_coyote_dir / "COYOTE_0001.JPG")

    def test_schema_and_required_fields(self) -> None:
        """Ensure all output rows strictly adhere to the required JSON schema."""
        out_path = self.root / "output" / "manifest.jsonl"
        # 2 inat + 3 bulk + 3 hb + 1 meyer = 9 rows total
        rows, report = builder.build_manifest(self.root, out_path, spot_check_sample_size=10)

        self.assertEqual(len(rows), 9)
        self.assertTrue(out_path.exists())

        for row in rows:
            for field in builder.REQUIRED_FIELDS:
                self.assertIn(field, row)
                self.assertTrue(row[field], f"Field {field} must not be empty")

            self.assertIn("image_id", row)
            self.assertIn("image_path", row)
            self.assertIn("source", row)
            self.assertIn("species_label", row)
            self.assertIn("scientific_name", row)
            self.assertIn("label_basis", row)
            self.assertIn("species_label_provisional", row)
            self.assertTrue(row["species_label_provisional"])
            self.assertIn("group_id", row)
            self.assertIn("status_flags", row)
            self.assertIn("image_sha256", row)
            self.assertIn("width", row)
            self.assertIn("height", row)
            self.assertIn("image_format", row)
            self.assertIn("image_error", row)
            self.assertIn("metadata_error", row)

            resolved = self.root / row["image_path"]
            self.assertTrue(resolved.exists(), f"Image path does not exist: {resolved}")

    def test_determinism(self) -> None:
        """Building manifest multiple times must yield byte-for-byte identical output."""
        out1 = self.root / "out1.jsonl"
        out2 = self.root / "out2.jsonl"
        builder.build_manifest(self.root, out1, spot_check_sample_size=10)
        builder.build_manifest(self.root, out2, spot_check_sample_size=10)
        self.assertEqual(out1.read_bytes(), out2.read_bytes())

    def test_unique_image_ids_and_paths(self) -> None:
        """Verify no duplicate image_ids or image_paths exist."""
        rows, _ = builder.build_manifest(self.root)
        image_ids = [r["image_id"] for r in rows]
        image_paths = [r["image_path"] for r in rows]
        self.assertEqual(len(image_ids), len(set(image_ids)))
        self.assertEqual(len(image_paths), len(set(image_paths)))

    def test_duplicate_and_overlap_logic(self) -> None:
        """Pilot overlap rows in bulk manifest must be skipped and not double-counted."""
        rows, report = builder.build_manifest(self.root)
        bulk_rep = report["inaturalist_bulk"]
        self.assertEqual(bulk_rep["skipped_pilot_overlap"], 1)

        # Confirm 101_201.jpg appears exactly once in the entire manifest
        matches_101 = [r for r in rows if r.get("observation_id") == 101 and r.get("photo_id") == 201]
        self.assertEqual(len(matches_101), 1)
        self.assertEqual(matches_101[0]["source"], "inaturalist")

    def test_missing_bulk_files_reported_not_fabricated(self) -> None:
        """Missing bulk files must be reported in missing_on_disk and omitted from manifest."""
        rows, report = builder.build_manifest(self.root)
        bulk_rep = report["inaturalist_bulk"]
        missing = bulk_rep["missing_on_disk"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["observation_id"], 999)
        self.assertEqual(missing[0]["photo_id"], 888)

        # Verify no fabricated row exists in the manifest for 999_888.jpg
        self.assertFalse(any("999_888" in r["image_path"] for r in rows))

    def test_gif_animations_excluded(self) -> None:
        """GIF animations must be explicitly excluded from still image manifest."""
        rows, report = builder.build_manifest(self.root)
        bulk_rep = report["inaturalist_bulk"]
        self.assertEqual(bulk_rep["skipped_gif_animations"], 1)
        self.assertFalse(any(r["image_path"].endswith(".gif") for r in rows))

    def test_normalized_class_mapping_and_subspecies(self) -> None:
        """Verify normalized species labels, generic_squirrel mapping, and subspecies preservation."""
        rows, _ = builder.build_manifest(self.root)

        # 1. squirrel_generic mapped to generic_squirrel, raw scientific name preserved
        row_302 = next(r for r in rows if r.get("observation_id") == 302)
        self.assertEqual(row_302["species_label"], "generic_squirrel")
        self.assertEqual(row_302["scientific_name"], "Sciurus aberti")

        # 2. Subspecies Sciurus niger cinereus keeps normalized species Sciurus niger
        row_303 = next(r for r in rows if r.get("observation_id") == 303)
        self.assertEqual(row_303["species_label"], "Sciurus niger")
        self.assertEqual(row_303["scientific_name"], "Sciurus niger cinereus")

        # 3. Pilot Glaucomys volans normalized to generic_squirrel, preserving raw scientific name
        row_102 = next(r for r in rows if r.get("observation_id") == 102)
        self.assertEqual(row_102["species_label"], "generic_squirrel")
        self.assertEqual(row_102["scientific_name"], "Glaucomys volans")

    def test_spot_check_with_pillow(self) -> None:
        """Spot check validates dimensions and catches corrupted files."""
        # Corrupt one bulk image
        corrupt_path = self.bulk_dir / "301_401.jpg"
        corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE")

        _, report = builder.build_manifest(self.root, spot_check_sample_size=10)
        bulk_rep = report["inaturalist_bulk"]
        self.assertGreaterEqual(len(bulk_rep["spot_check_failures"]), 1)
        self.assertIn("301_401.jpg", bulk_rep["spot_check_failures"][0]["path"])

    def test_harrybaines_and_meyer_sources(self) -> None:
        """Verify Harry Baines user_confirmed labels and Meyer trailcam filtering."""
        rows, report = builder.build_manifest(self.root)
        hb_rows = [r for r in rows if r["source"] == "harrybaines"]
        meyer_rows = [r for r in rows if r["source"] == "meyer_trailcam"]

        self.assertEqual(len(hb_rows), 3)
        self.assertTrue(all(r["species_label"] == "Sciurus carolinensis" for r in hb_rows))
        self.assertTrue(all(r["label_basis"] == "user_confirmed" for r in hb_rows))

        self.assertEqual(len(meyer_rows), 1)
        self.assertEqual(meyer_rows[0]["species_label"], "generic_squirrel")
        self.assertEqual(meyer_rows[0]["label_basis"], "generic_source")
        self.assertNotIn("-rendered", meyer_rows[0]["image_path"])

        # Check Meyer exclusions
        meyer_rep = report["meyer_trailcam"]
        self.assertEqual(meyer_rep["skipped_rendered_derivatives"], 1)
        self.assertEqual(meyer_rep["skipped_xml_annotations"], 1)
        self.assertIn("Coyote", meyer_rep["excluded_other_animal_folders"])

    def test_missing_metadata_raises_clear_error(self) -> None:
        """Missing authoritative metadata or manifests raises FileNotFoundError."""
        self.bulk_manifest.unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            builder.build_manifest(self.root)
        self.assertIn("manifest.jsonl", str(ctx.exception))

    def test_cli_execution(self) -> None:
        """Test invoking builder main() function via simulated CLI arguments."""
        out_file = self.root / "cli_out.jsonl"
        exit_code = builder.main(["--repo-root", str(self.root), "--output", str(out_file), "--spot-check-size", "5"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(out_file.exists())
        lines = out_file.read_text().splitlines()
        self.assertEqual(len(lines), 9)


if __name__ == "__main__":
    unittest.main()
