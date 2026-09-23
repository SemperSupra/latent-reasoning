#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from tools.hydrate_pinned_lfs import (
    normalize_repository,
    parse_lfs_pointer,
    validate_oid,
    validate_repo_path,
    validate_revision,
)


class PinnedLFSHydratorTests(unittest.TestCase):
    def test_repository_normalization(self):
        self.assertEqual(
            normalize_repository("da03/Internalize_CoT_Step_by_Step"),
            (
                "da03/Internalize_CoT_Step_by_Step",
                "https://github.com/da03/Internalize_CoT_Step_by_Step.git",
            ),
        )
        self.assertEqual(
            normalize_repository("https://github.com/da03/Internalize_CoT_Step_by_Step.git")[0],
            "da03/Internalize_CoT_Step_by_Step",
        )

    def test_rejects_non_github_repository(self):
        with self.assertRaises(ValueError):
            normalize_repository("https://example.com/a/b.git")

    def test_exact_revision_required(self):
        good="e06a32ee5e4cd117171daeb4755d2a97ece62761"
        self.assertEqual(validate_revision(good),good)
        with self.assertRaises(ValueError):
            validate_revision("main")

    def test_pointer_parsing(self):
        pointer=parse_lfs_pointer(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c\n"
            "size 87805358\n"
        )
        self.assertEqual(
            pointer["oid_sha256"],
            "0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c",
        )
        self.assertEqual(pointer["size_bytes"],87805358)

    def test_path_boundary(self):
        self.assertEqual(validate_repo_path("data/gsm8k/train.txt"),"data/gsm8k/train.txt")
        for bad in ("/etc/passwd","../secret","data/../secret","has space/file"):
            with self.assertRaises(ValueError):
                validate_repo_path(bad)

    def test_oid_accepts_optional_prefix(self):
        oid="0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c"
        self.assertEqual(validate_oid(oid),oid)
        self.assertEqual(validate_oid("sha256:"+oid),oid)


if __name__=="__main__":
    unittest.main()
