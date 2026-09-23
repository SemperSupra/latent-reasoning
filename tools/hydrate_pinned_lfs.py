#!/usr/bin/env python3
"""Hydrate one exact Git LFS object from a pinned GitHub revision.

The tool is intentionally narrow: one LFS path plus optional ordinary companion
paths. It performs a no-shell sparse checkout, activates the historical LFS
attribute locally, verifies the committed pointer, and verifies the hydrated
payload before emitting a receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
OID_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_repository(value: str) -> tuple[str, str]:
    """Return (owner/repo, https clone URL) for a GitHub repository."""
    if value.startswith("https://github.com/"):
        parsed=urlparse(value)
        if parsed.scheme != "https" or parsed.netloc != "github.com":
            raise ValueError("repository must be hosted on github.com over HTTPS")
        path=parsed.path.strip("/")
        if path.endswith(".git"):
            path=path[:-4]
    else:
        path=value.strip("/")

    parts=path.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repository must be owner/repo or an https://github.com/owner/repo URL")
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise ValueError("repository owner/name contains unsupported characters")

    slug="/".join(parts)
    return slug, f"https://github.com/{slug}.git"


def validate_revision(value: str) -> str:
    if not REVISION_RE.fullmatch(value):
        raise ValueError("revision must be an exact 40-character lowercase Git SHA")
    return value


def validate_oid(value: str) -> str:
    value=value.removeprefix("sha256:")
    if not OID_RE.fullmatch(value):
        raise ValueError("LFS OID must be a 64-character lowercase SHA-256")
    return value


def validate_repo_path(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"invalid repository-relative path: {value!r}")
    if any(ch.isspace() for ch in value):
        raise ValueError("v1 hydrator does not accept whitespace in repository paths")
    pure=PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    return pure.as_posix()


def parse_lfs_pointer(text: str) -> dict[str, object]:
    fields={}
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("version "):
            fields["version"]=line.removeprefix("version ")
        elif line.startswith("oid sha256:"):
            fields["oid_sha256"]=line.removeprefix("oid sha256:")
        elif line.startswith("size "):
            fields["size_bytes"]=int(line.removeprefix("size "))

    if fields.get("version") != "https://git-lfs.github.com/spec/v1":
        raise ValueError("not a Git LFS v1 pointer")
    oid=fields.get("oid_sha256")
    if not isinstance(oid,str) or not OID_RE.fullmatch(oid):
        raise ValueError("pointer lacks a valid SHA-256 OID")
    size=fields.get("size_bytes")
    if not isinstance(size,int) or size < 0:
        raise ValueError("pointer lacks a valid size")
    return fields


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv: list[str], *, cwd: Path | None=None, env: dict[str,str] | None=None) -> str:
    result=subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def git(repo: Path, *args: str, env: dict[str,str] | None=None) -> str:
    return run(["git","-C",str(repo),*args],env=env)


def prepare_checkout(
    *,
    repository: str,
    revision: str,
    lfs_path: str,
    includes: list[str],
    checkout_dir: Path,
) -> tuple[str,str]:
    slug,clone_url=normalize_repository(repository)
    revision=validate_revision(revision)
    lfs_path=validate_repo_path(lfs_path)
    includes=[validate_repo_path(path) for path in includes]

    if checkout_dir.exists():
        if any(checkout_dir.iterdir()):
            raise ValueError(f"checkout directory is not empty: {checkout_dir}")
    else:
        checkout_dir.mkdir(parents=True)

    clone_env=os.environ.copy()
    clone_env["GIT_LFS_SKIP_SMUDGE"]="1"
    run(
        ["git","clone","--filter=blob:none","--no-checkout",clone_url,str(checkout_dir)],
        env=clone_env,
    )
    git(checkout_dir,"lfs","install","--local")

    info=checkout_dir/".git"/"info"
    info.mkdir(parents=True,exist_ok=True)
    (info/"attributes").write_text(
        f"{lfs_path} filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    git(checkout_dir,"sparse-checkout","init","--no-cone")
    sparse=[f"/{lfs_path}",*[f"/{path}" for path in includes]]
    (info/"sparse-checkout").write_text("\n".join(sparse)+"\n",encoding="utf-8")

    checkout_env=os.environ.copy()
    checkout_env.pop("GIT_LFS_SKIP_SMUDGE",None)
    git(checkout_dir,"checkout",revision,env=checkout_env)

    head=git(checkout_dir,"rev-parse","HEAD").strip()
    if head != revision:
        raise RuntimeError(f"checkout revision mismatch: {head} != {revision}")

    return slug,clone_url


def hydrate(
    *,
    repository: str,
    revision: str,
    lfs_path: str,
    expected_oid: str,
    expected_size: int,
    includes: list[str],
    checkout_dir: Path,
) -> dict:
    revision=validate_revision(revision)
    lfs_path=validate_repo_path(lfs_path)
    includes=[validate_repo_path(path) for path in includes]
    expected_oid=validate_oid(expected_oid)
    if expected_size < 0:
        raise ValueError("expected size must be non-negative")

    slug,_=prepare_checkout(
        repository=repository,
        revision=revision,
        lfs_path=lfs_path,
        includes=includes,
        checkout_dir=checkout_dir,
    )

    pointer_text=git(checkout_dir,"show",f"{revision}:{lfs_path}")
    pointer=parse_lfs_pointer(pointer_text)
    if pointer["oid_sha256"] != expected_oid:
        raise RuntimeError(
            f"LFS pointer OID mismatch: {pointer['oid_sha256']} != {expected_oid}"
        )
    if pointer["size_bytes"] != expected_size:
        raise RuntimeError(
            f"LFS pointer size mismatch: {pointer['size_bytes']} != {expected_size}"
        )

    hydrated=checkout_dir/lfs_path
    if not hydrated.is_file():
        raise RuntimeError(f"hydrated path missing: {hydrated}")
    actual_size=hydrated.stat().st_size
    actual_oid=sha256_file(hydrated)
    if actual_size != expected_size:
        raise RuntimeError(f"hydrated size mismatch: {actual_size} != {expected_size}")
    if actual_oid != expected_oid:
        raise RuntimeError(f"hydrated SHA-256 mismatch: {actual_oid} != {expected_oid}")

    companion_receipts=[]
    for path in includes:
        target=checkout_dir/path
        if not target.is_file():
            raise RuntimeError(f"companion path missing: {target}")
        companion_receipts.append({
            "path":path,
            "size_bytes":target.stat().st_size,
            "sha256":sha256_file(target),
        })

    return {
        "schema_version":1,
        "evidence_class":"pinned-git-lfs-hydration",
        "repository":slug,
        "revision":revision,
        "lfs":{
            "path":lfs_path,
            "pointer":pointer,
            "hydrated_size_bytes":actual_size,
            "hydrated_sha256":actual_oid,
        },
        "companions":companion_receipts,
        "method":{
            "checkout":"sparse",
            "attribute_scope":"local .git/info/attributes",
            "shell_invocation":False,
        },
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--repository",required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--lfs-path",required=True)
    parser.add_argument("--oid",required=True)
    parser.add_argument("--size",type=int,required=True)
    parser.add_argument("--include",action="append",default=[])
    parser.add_argument("--checkout-dir",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()

    if args.checkout_dir.exists():
        if any(args.checkout_dir.iterdir()):
            raise SystemExit(f"checkout directory must be absent or empty: {args.checkout_dir}")
        args.checkout_dir.rmdir()

    receipt=hydrate(
        repository=args.repository,
        revision=args.revision,
        lfs_path=args.lfs_path,
        expected_oid=args.oid,
        expected_size=args.size,
        includes=args.include,
        checkout_dir=args.checkout_dir,
    )
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(receipt,sort_keys=True))


if __name__=="__main__":
    main()
