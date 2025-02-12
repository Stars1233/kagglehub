import os
from typing import Optional

import kagglehub
from kagglehub.cache import DATASETS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_dataset_handle
from tests.fixtures import BaseTestCase

from .server_stubs import dataset_download_stub as stub
from .server_stubs import serv
from .utils import AUTO_COMPRESSED_FILE_NAME, create_test_cache

INVALID_ARCHIVE_DATASET_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata/versions/2"
UNVERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata"
TEST_FILEPATH = "foo.txt"
TEST_CONTENTS = "foo"
AUTO_COMPRESSED_CONTENTS = """"shape","degrees","sides","color","date"
"square",360,4,"blue","2024-12-17"
"circle",360,,"red","2023-08-01"
"triangle",180,3,"green","2022-01-05"
"""

EXPECTED_DATASET_SUBDIR = os.path.join(
    DATASETS_CACHE_SUBFOLDER, "sarahjeffreson", "featured-spotify-artiststracks-with-metadata", "versions", "2"
)
EXPECTED_DATASET_SUBPATH = os.path.join(
    DATASETS_CACHE_SUBFOLDER,
    "sarahjeffreson",
    "featured-spotify-artiststracks-with-metadata",
    "versions",
    "2",
)


class TestHttpDatasetDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _download_dataset_and_assert_downloaded(
        self,
        d: str,
        dataset_handle: str,
        expected_subdir_or_subpath: str,
        expected_files: Optional[list[str]] = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        # Download the full datasets and ensure all files are there.
        dataset_path = kagglehub.dataset_download(dataset_handle, **kwargs)

        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), dataset_path)

        if not expected_files:
            expected_files = ["foo.txt"]
        self.assertEqual(sorted(expected_files), sorted(os.listdir(dataset_path)))

        # Assert that the archive file has been deleted
        archive_path = get_cached_archive_path(parse_dataset_handle(dataset_handle))
        self.assertFalse(os.path.exists(archive_path))

    def _download_test_file_and_assert_downloaded(self, d: str, dataset_handle: str, **kwargs) -> None:  # noqa: ANN003
        dataset_path = kagglehub.dataset_download(dataset_handle, path=TEST_FILEPATH, **kwargs)
        self.assertEqual(os.path.join(d, EXPECTED_DATASET_SUBPATH, TEST_FILEPATH), dataset_path)
        with open(dataset_path) as dataset_file:
            self.assertEqual(TEST_CONTENTS, dataset_file.read())

    def _download_test_file_and_assert_downloaded_auto_compressed(
        self,
        d: str,
        dataset_handle: str,
        **kwargs,  # noqa: ANN003
    ) -> None:
        dataset_path = kagglehub.dataset_download(dataset_handle, path=AUTO_COMPRESSED_FILE_NAME, **kwargs)
        self.assertEqual(os.path.join(d, EXPECTED_DATASET_SUBPATH, AUTO_COMPRESSED_FILE_NAME), dataset_path)
        with open(dataset_path) as dataset_file:
            self.assertEqual(AUTO_COMPRESSED_CONTENTS, dataset_file.read())

    def test_unversioned_dataset_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(d, UNVERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR)

    def test_versioned_dataset_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(d, VERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR)

    def test_versioned_dataset_targz_archive_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(
                d,
                stub.TARGZ_ARCHIVE_HANDLE,
                f"{DATASETS_CACHE_SUBFOLDER}/{stub.TARGZ_ARCHIVE_HANDLE}",
                expected_files=[f"{i}.txt" for i in range(1, 51)],
            )

    def test_versioned_dataset_download_bad_archive(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.dataset_download(INVALID_ARCHIVE_DATASET_HANDLE)

    def test_versioned_dataset_download_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, VERSIONED_DATASET_HANDLE)

    def test_versioned_dataset_download_with_auto_compressed_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded_auto_compressed(d, VERSIONED_DATASET_HANDLE)

    def test_unversioned_dataset_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(
                d, UNVERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR, force_download=True
            )

    def test_versioned_dataset_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(
                d, VERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR, force_download=True
            )

    def test_versioned_dataset_full_download_with_file_already_cached(self) -> None:
        with create_test_cache() as d:
            # Download a single file first
            kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path=TEST_FILEPATH)
            self._download_dataset_and_assert_downloaded(d, VERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR)

    def test_unversioned_dataset_full_download_with_file_already_cached(self) -> None:
        with create_test_cache() as d:
            # Download a single file first
            kagglehub.dataset_download(UNVERSIONED_DATASET_HANDLE, path=TEST_FILEPATH)
            self._download_dataset_and_assert_downloaded(d, UNVERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR)
