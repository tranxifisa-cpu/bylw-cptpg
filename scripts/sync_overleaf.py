"""Sync the committed Chinese thesis to its folder in the existing Overleaf project."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
from urllib.parse import urlsplit


PROJECT_ID = "6946a3f58266e8759a116aa4"
SOURCE = "output_paper/\u5e94\u7edf\u4f5c\u6587\u6a21\u677f"
TARGET = "\u4e2d\u6587\u521d\u7a3fv1"
OVERLEAF_URL = f"https://git@git.overleaf.com/{PROJECT_ID}"
TOKEN = re.compile(rb"olp_[A-Za-z0-9]{20,}")
BUILD_SUFFIXES = (
    ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out",
    ".synctex.gz", ".toc", ".lof", ".lot", ".nav", ".snm", ".vrb",
)
TEXT_SUFFIXES = {".tex", ".bib", ".bst", ".sty", ".cls", ".md", ".m", ".txt", ".csv"}
HOOK = """#!/bin/sh
# overleaf-sync: prevent credentials from entering Git history.
git grep --cached --text --quiet -E 'olp_[A-Za-z0-9]{20,}' -- .
result=$?
if [ "$result" -eq 0 ]; then
    echo 'Overleaf token detected in staged files; commit refused.' >&2
    exit 1
fi
if [ "$result" -ne 1 ]; then exit "$result"; fi
"""


class SyncError(RuntimeError):
    pass


def git(repo: Path, *args: str, data: bytes | None = None, check: bool = True,
        overleaf: bool = False) -> subprocess.CompletedProcess:
    command = ["git", "-C", str(repo)]
    if overleaf:
        command += ["-c", "http.version=HTTP/1.1"]
        if os.name == "nt":
            command += ["-c", "http.sslBackend=openssl"]
    environment = os.environ.copy()
    environment.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    result = subprocess.run(command + list(args), input=data, capture_output=True,
                            env=environment, timeout=180)
    if check and result.returncode:
        detail = TOKEN.sub(b"<REDACTED>", result.stderr).decode("utf-8", "replace").strip()
        raise SyncError(detail or f"git {args[0]} failed")
    return result


def git_text(repo: Path, *args: str, **kwargs) -> str:
    return git(repo, *args, **kwargs).stdout.decode("utf-8", "replace").strip()


def included(path: str) -> bool:
    name = PurePosixPath(path).name
    return (name not in {".DS_Store", "Thumbs.db"}
            and not name.lower().endswith(BUILD_SUFFIXES + (".docx",)))


def read_files(repo: Path, ref: str, prefix: str) -> dict[str, bytes]:
    archive = git(repo, "archive", "--format=tar", ref, "--", prefix).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if not item.name.startswith(prefix + "/"):
                continue
            relative = item.name[len(prefix) + 1:]
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                raise SyncError("Unsafe path in Git archive")
            if item.isdir() or not included(relative):
                continue
            if not item.isfile():
                raise SyncError(f"Only regular files can be synchronized: {relative}")
            files[relative] = stream.extractfile(item).read()
    if not files or "Thesis_main.tex" not in files:
        raise SyncError(f"Thesis_main.tex was not found under {prefix}")
    return files


def fingerprints(files: dict[str, bytes]) -> dict[str, str]:
    result = {}
    for path, content in files.items():
        if PurePosixPath(path).suffix.lower() in TEXT_SUFFIXES:
            content = content.removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        result[path] = hashlib.sha256(content).hexdigest()
    return result


def changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(path for path in before.keys() | after.keys()
                  if before.get(path) != after.get(path))


def destination(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise SyncError(f"Path leaves the thesis folder: {relative}")
    return path


def apply_files(root: Path, before: dict[str, bytes], after: dict[str, bytes]) -> None:
    for content in after.values():
        if TOKEN.search(content):
            raise SyncError("A source file contains an Overleaf token; synchronization refused")
    for path in before.keys() | after.keys():
        target = destination(root, path)
        if target.is_symlink():
            raise SyncError(f"Symlink in destination: {path}")
    for path in sorted(before.keys() - after.keys()):
        destination(root, path).unlink()
    for path in changed(fingerprints(before), fingerprints(after)):
        if path in after:
            target = destination(root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(after[path])


def save_state(path: Path, source_commit: str, overleaf_commit: str,
               files: dict[str, str]) -> None:
    state = {"version": 1, "project_id": PROJECT_ID, "source": SOURCE,
             "target": TARGET, "source_commit": source_commit,
             "overleaf_commit": overleaf_commit, "files": files}
    path.write_text(json.dumps(state, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def authenticate(repo: Path) -> None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise SyncError("Run setup in your own interactive terminal to enter a new token privately")
    helpers = git_text(repo, "config", "--get-all", "credential.helper", check=False).splitlines()
    if not helpers or any(helper.strip().split(" ")[0] == "store" for helper in helpers):
        raise SyncError("Configure a secure Git credential helper first (Windows: manager)")
    print("Paste once and press Enter. No characters will be displayed.")
    token = getpass.getpass("New Overleaf token (input hidden): ")
    if not TOKEN.fullmatch(token.encode()):
        raise SyncError("The input is not an Overleaf Git authentication token")
    credential = f"protocol=https\nhost=git.overleaf.com\nusername=git\npassword={token}\n\n"
    git(repo, "credential", "approve", data=credential.encode())
    del credential, token


def prepare_bridge(repo: Path, bridge: Path, setup: bool, fetch: bool) -> bool:
    if setup:
        authenticate(repo)
    if not (bridge / ".git").is_dir():
        if not setup:
            raise SyncError("Overleaf is not cloned yet. Run: python scripts/sync_overleaf.py setup")
        if bridge.exists():
            raise SyncError(f"Clone destination already exists: {bridge}")
        git(repo, "clone", OVERLEAF_URL, str(bridge), overleaf=True)
    remote = urlsplit(git_text(bridge, "remote", "get-url", "origin"))
    if (remote.scheme != "https" or remote.hostname != "git.overleaf.com"
            or remote.path.rstrip("/") != "/" + PROJECT_ID
            or remote.password or remote.username not in (None, "git")
            or remote.query or remote.fragment):
        raise SyncError("The clone must use this project's token-free Overleaf HTTPS URL")
    if git_text(bridge, "status", "--porcelain"):
        raise SyncError("The Overleaf clone has uncommitted changes; resolve them before syncing")
    remote_ref = git_text(bridge, "symbolic-ref", "--quiet", "--short",
                          "refs/remotes/origin/HEAD", check=False)
    if not remote_ref.startswith("origin/"):
        raise SyncError("The default Overleaf branch could not be determined")
    branch = remote_ref[len("origin/"):]
    if git_text(bridge, "branch", "--show-current") != branch:
        raise SyncError(f"The Overleaf clone must be on its {branch} branch")
    if fetch:
        git(bridge, "fetch", "origin", f"{branch}:refs/remotes/origin/{branch}", overleaf=True)
    head = git_text(bridge, "rev-parse", "HEAD")
    remote_head = git_text(bridge, "rev-parse", remote_ref)
    if head != remote_head:
        if not git(bridge, "merge-base", "--is-ancestor", "HEAD", remote_ref, check=False).returncode:
            git(bridge, "merge", "--ff-only", remote_ref)
        elif not git(bridge, "merge-base", "--is-ancestor", remote_ref, "HEAD", check=False).returncode:
            paths = git(bridge, "diff", "--name-only", "-z", f"{remote_ref}..HEAD").stdout.split(b"\0")
            if any(path and not path.decode("utf-8").startswith(TARGET + "/") for path in paths):
                raise SyncError("Unpushed Overleaf commits change files outside the Chinese thesis")
            return True
        else:
            raise SyncError("Overleaf history diverged. Merge the remote edits before syncing")
    return False


def configure_bridge(repo: Path, bridge: Path) -> None:
    hook = bridge / ".git" / "hooks" / "pre-commit"
    if not hook.exists():
        hook.write_text(HOOK, encoding="utf-8", newline="\n")
        hook.chmod(0o755)
    for key in ("user.name", "user.email"):
        if not git_text(bridge, "config", "--get", key, check=False):
            value = git_text(repo, "config", "--get", key, check=False)
            if value:
                git(bridge, "config", key, value)


def synchronize(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    bridge = (args.overleaf_dir or repo.parent / "paper-overleaf").resolve()
    if bridge == repo or bridge.is_relative_to(repo / "output_paper"):
        raise SyncError("Use a separate clone outside the paper source directory")
    if not args.no_fetch:
        git(repo, "fetch", "origin", "main:refs/remotes/origin/main")
    source_commit = git_text(repo, "rev-parse", "--verify", args.source_ref + "^{commit}")
    source = read_files(repo, source_commit, SOURCE)
    source_hashes = fingerprints(source)
    pending = prepare_bridge(repo, bridge, args.command == "setup", not args.no_fetch)
    branch = git_text(bridge, "branch", "--show-current")
    if args.command != "status":
        configure_bridge(repo, bridge)
    target = read_files(bridge, "HEAD", TARGET)
    target_hashes = fingerprints(target)
    state_path = bridge / ".git" / "overleaf-sync-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if state and (state.get("project_id"), state.get("source"), state.get("target")) != (PROJECT_ID, SOURCE, TARGET):
        raise SyncError("The saved synchronization baseline belongs to a different mapping")
    differences = changed(target_hashes, source_hashes)
    remote_edits = changed(state["files"], target_hashes) if state else []
    print(f"GitHub source: {args.source_ref} ({source_commit[:12]})")
    print(f"Overleaf clone: {bridge}")
    print(f"Mapping: {SOURCE} -> {TARGET}; {len(source)} source files")
    for path in differences:
        verb = "ADD" if path not in target else "DELETE" if path not in source else "UPDATE"
        print(f"{verb}: {path}")
    if args.command == "status":
        print(f"{len(differences)} file changes; baseline {'present' if state else 'not initialized'}")
        if remote_edits:
            print("Overleaf changes since the baseline: " + ", ".join(remote_edits))
        if pending:
            print("A local Overleaf commit is waiting to be pushed")
        return
    if differences and not state:
        raise SyncError("Initial source and Overleaf files differ; compare them before establishing the baseline")
    if differences and remote_edits:
        raise SyncError("Overleaf was edited since the baseline: " + ", ".join(remote_edits)
                        + ". Incorporate these edits into GitHub before pushing")
    if pending and differences:
        raise SyncError("An older sync commit is still unpushed; resolve it before syncing a newer source")
    if differences:
        apply_files(bridge / TARGET, target, source)
        git(bridge, "add", "--all", "--", TARGET)
        if not git(bridge, "diff", "--cached", "--quiet", check=False).returncode:
            raise SyncError("Git produced no staged change despite differing source files")
        git(bridge, "commit", "-m", f"Sync Chinese thesis from GitHub {source_commit[:12]}")
        if fingerprints(read_files(bridge, "HEAD", TARGET)) != source_hashes:
            raise SyncError("Committed Overleaf files do not match the source")
    if differences or pending:
        git(bridge, "push", "origin", f"HEAD:{branch}", overleaf=True)
        git(bridge, "fetch", "origin", f"{branch}:refs/remotes/origin/{branch}", overleaf=True)
        if git_text(bridge, "rev-parse", "HEAD") != git_text(bridge, "rev-parse", f"origin/{branch}"):
            raise SyncError("Overleaf advanced during verification; run status to inspect the new version")
    save_state(state_path, source_commit, git_text(bridge, "rev-parse", "HEAD"), source_hashes)
    print("Synchronized and verified" if differences or pending else "Already identical; baseline saved")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "status", "push"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overleaf-dir", type=Path)
    parser.add_argument("--source-ref", default="origin/main")
    parser.add_argument("--no-fetch", action="store_true", help="Inspect already-fetched refs without network access")
    args = parser.parse_args()
    try:
        synchronize(args)
    except (SyncError, OSError, subprocess.TimeoutExpired, ValueError) as error:
        message = TOKEN.sub(b"<REDACTED>", str(error).encode("utf-8")).decode("utf-8")
        print(f"Error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
