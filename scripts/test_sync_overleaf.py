"""Exercise scoped sync, conflicting edits, and retry using local Git repositories."""

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sync_overleaf as sync


class OverleafSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bylw-overleaf-test-")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repo, self.bridge, self.remote = root / "research", root / "overleaf", root / "remote.git"
        for repo, branch in ((self.repo, "main"), (self.bridge, "master")):
            repo.mkdir()
            sync.git(repo, "init", "-b", branch)
            sync.git(repo, "config", "user.name", "Sync Test")
            sync.git(repo, "config", "user.email", "sync-test@example.invalid")
        source, target = self.repo / sync.SOURCE, self.bridge / sync.TARGET
        for folder in (source, target):
            (folder / "chapters").mkdir(parents=True)
            (folder / "Thesis_main.tex").write_text("original\n", encoding="utf-8")
            (folder / "chapters" / "obsolete.tex").write_text("obsolete\n", encoding="utf-8")
        (self.bridge / "other-paper.tex").write_text("keep this paper\n", encoding="utf-8")
        self.commit(self.repo)
        self.commit(self.bridge)
        sync.git(self.bridge, "remote", "add", "origin", sync.OVERLEAF_URL)
        sync.git(self.bridge, "update-ref", "refs/remotes/origin/master", "HEAD")
        sync.git(self.bridge, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/master")
        sync.git(root, "init", "--bare", str(self.remote))
        sync.git(self.bridge, "push", str(self.remote), "HEAD:master")
        self.args = argparse.Namespace(command="push", repo=self.repo, overleaf_dir=self.bridge,
                                       source_ref="HEAD", no_fetch=True)
        self.real_git = sync.git
        self.fail_push = False
        self.quiet_sync()

    @staticmethod
    def commit(repo):
        sync.git(repo, "add", "--all")
        sync.git(repo, "commit", "-m", "Test fixture")

    def quiet_sync(self):
        with redirect_stdout(io.StringIO()):
            sync.synchronize(self.args)

    def network(self, repo, *args, **kwargs):
        if repo == self.bridge and args[0] in {"push", "fetch"}:
            if args[0] == "push" and self.fail_push:
                self.fail_push = False
                raise sync.SyncError("Simulated connection loss")
            args = (args[0], str(self.remote), *args[2:])
        return self.real_git(repo, *args, **kwargs)

    def edit_source(self):
        source = self.repo / sync.SOURCE
        (source / "Thesis_main.tex").write_text("revised\n", encoding="utf-8")
        (source / "chapters" / "new.tex").write_text("new chapter\n", encoding="utf-8")
        (source / "chapters" / "obsolete.tex").unlink()
        self.commit(self.repo)

    def test_push_add_update_delete_preserves_other_paper(self):
        self.edit_source()
        with patch.object(sync, "git", side_effect=self.network):
            self.quiet_sync()
        target = self.bridge / sync.TARGET
        self.assertEqual((target / "Thesis_main.tex").read_text(), "revised\n")
        self.assertTrue((target / "chapters" / "new.tex").exists())
        self.assertFalse((target / "chapters" / "obsolete.tex").exists())
        self.assertEqual((self.bridge / "other-paper.tex").read_text(), "keep this paper\n")
        self.assertEqual(sync.git_text(self.bridge, "rev-parse", "HEAD"),
                         sync.git_text(self.remote, "rev-parse", "master"))

    def test_overleaf_edit_blocks_overwrite(self):
        target = self.bridge / sync.TARGET / "Thesis_main.tex"
        target.write_text("advisor edit\n", encoding="utf-8")
        self.commit(self.bridge)
        sync.git(self.bridge, "update-ref", "refs/remotes/origin/master", "HEAD")
        self.edit_source()
        with self.assertRaisesRegex(sync.SyncError, "Overleaf was edited"):
            self.quiet_sync()
        self.assertEqual(target.read_text(), "advisor edit\n")

    def test_main_default_branch_is_detected(self):
        sync.git(self.bridge, "branch", "-m", "main")
        sync.git(self.bridge, "update-ref", "refs/remotes/origin/main", "HEAD")
        sync.git(self.bridge, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
        self.edit_source()
        with patch.object(sync, "git", side_effect=self.network):
            self.quiet_sync()
        self.assertEqual(sync.git_text(self.remote, "rev-parse", "main"),
                         sync.git_text(self.bridge, "rev-parse", "HEAD"))

    def test_failed_push_can_be_retried_without_new_commit(self):
        self.edit_source()
        self.fail_push = True
        with patch.object(sync, "git", side_effect=self.network):
            with self.assertRaisesRegex(sync.SyncError, "connection loss"):
                self.quiet_sync()
            pending = sync.git_text(self.bridge, "rev-parse", "HEAD")
            self.quiet_sync()
            self.assertEqual(sync.git_text(self.bridge, "rev-parse", "HEAD"), pending)
        self.assertEqual(sync.git_text(self.remote, "rev-parse", "master"), pending)

    def test_token_detection_precedes_file_changes(self):
        target = self.bridge / sync.TARGET
        original = (target / "Thesis_main.tex").read_bytes()
        before = {"Thesis_main.tex": b"original\n"}
        after = {"Thesis_main.tex": b"olp_" + b"A" * 24}
        with self.assertRaisesRegex(sync.SyncError, "contains an Overleaf token"):
            sync.apply_files(target, before, after)
        self.assertEqual((target / "Thesis_main.tex").read_bytes(), original)

    def test_line_endings_do_not_create_false_changes(self):
        self.assertEqual(sync.fingerprints({"a.tex": b"a\r\nb\r\n"}),
                         sync.fingerprints({"a.tex": b"a\nb\n"}))


if __name__ == "__main__":
    unittest.main()
