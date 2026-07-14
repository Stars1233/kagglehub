"""Test helper functions in kagglehub."""

import pathlib
import tempfile
import threading
import time
from unittest import mock

from kagglesdk.blobs.types.blob_api_service import ApiBlobType

from kagglehub.gcs_upload import filtered_walk, normalize_patterns, upload_files_and_directories
from tests.fixtures import BaseTestCase


class TesModelsHelpers(BaseTestCase):
    def testnormalize_patterns(self) -> None:
        default_patterns = [".git/", ".cache/", ".gitignore"]
        self.assertEqual(
            normalize_patterns(default=default_patterns, additional=None),
            [".git/*", ".cache/*", ".gitignore"],
        )
        self.assertEqual(
            normalize_patterns(default=default_patterns, additional=["original/", "*/*.txt", "doc/readme.txt"]),
            [".git/*", ".cache/*", ".gitignore", "original/*", "*/*.txt", "doc/readme.txt"],
        )

    def test_filtered_walk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_p = pathlib.Path(tmp_dir)

            # files to upload
            (tmp_dir_p / "a" / "b").mkdir(parents=True)
            (tmp_dir_p / "weights.txt").touch()
            (tmp_dir_p / "a" / "a.txt").touch()
            (tmp_dir_p / "a" / "b" / "b.txt").touch()
            (tmp_dir_p / "a" / "b" / ".bb").touch()
            expected_files = {
                tmp_dir_p / "weights.txt",
                tmp_dir_p / "a" / "a.txt",
                tmp_dir_p / "a" / "b" / "b.txt",
                tmp_dir_p / "a" / "b" / ".bb",
            }

            # files to ignore
            (tmp_dir_p / ".git").mkdir(parents=True)
            (tmp_dir_p / ".git" / "file").write_text("hidden git file")
            (tmp_dir_p / ".gitignore").write_text("none")

            (tmp_dir_p / "a" / ".git").mkdir(parents=True)
            (tmp_dir_p / "a" / "b" / ".git").mkdir(parents=True)
            (tmp_dir_p / "a" / "b" / ".git" / "abgit.txt").write_text("abgit")

            (tmp_dir_p / "a" / "b" / ".hidden").touch()

            (tmp_dir_p / "original" / "fp8").mkdir(parents=True)
            (tmp_dir_p / "original" / "fp8" / "weights").touch()
            (tmp_dir_p / "original" / "fp16").mkdir(parents=True)
            (tmp_dir_p / "original" / "fp16" / "weights").touch()

            # filtered walk
            ignore_patterns = normalize_patterns(
                default=[".git/", "*/.git/", ".gitignore", "*/.hidden", "original/"], additional=None
            )
            walked_files = []
            for dir_path, _, file_names in filtered_walk(base_dir=tmp_dir, ignore_patterns=ignore_patterns):
                for file_name in file_names:
                    walked_files.append(pathlib.Path(dir_path) / file_name)
            self.assertEqual(set(walked_files), expected_files)

    def test_upload_files_and_directories_uploads_files_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_p = pathlib.Path(tmp_dir)
            for file_name in ["a.txt", "b.txt", "c.txt"]:
                (tmp_dir_p / file_name).touch()

            lock = threading.Lock()
            active_uploads = 0
            max_active_uploads = 0

            def fake_upload_file(file_path: str, *, quiet: bool, item_type: ApiBlobType) -> str:
                nonlocal active_uploads, max_active_uploads
                _ = quiet, item_type
                with lock:
                    active_uploads += 1
                    max_active_uploads = max(max_active_uploads, active_uploads)
                time.sleep(0.05)
                with lock:
                    active_uploads -= 1
                return pathlib.Path(file_path).name

            with mock.patch("kagglehub.gcs_upload._upload_file", side_effect=fake_upload_file):
                upload_dir = upload_files_and_directories(
                    tmp_dir,
                    ignore_patterns=[],
                    item_type=ApiBlobType.DATASET,
                    quiet=True,
                )

            self.assertGreater(max_active_uploads, 1)
            self.assertCountEqual(upload_dir.files, ["a.txt", "b.txt", "c.txt"])
