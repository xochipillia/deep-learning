#!/usr/bin/env python3
"""Minimal boto3 directory sync (the Studio image may not ship the AWS CLI).

    python3 s3sync.py up   <local_dir> <s3_prefix>     # local -> s3://$REPL_BUCKET/<prefix>
    python3 s3sync.py down <s3_prefix> <local_dir>     # s3 -> local

Uploads skip files whose size and MD5 ETag already match. Uses the execution
role; no credentials involved.
"""
import hashlib
import os
import sys
from pathlib import Path

import boto3

s3 = boto3.client("s3")
B = os.environ["REPL_BUCKET"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote(prefix: str) -> dict:
    out, tok = {}, None
    while True:
        kw = {"Bucket": B, "Prefix": prefix.rstrip("/") + "/"}
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            out[o["Key"]] = (o["Size"], o["ETag"].strip('"'))
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


def up(local: str, prefix: str) -> None:
    rem, n = remote(prefix), 0
    for f in sorted(Path(local).rglob("*")):
        if not f.is_file():
            continue
        key = f"{prefix.rstrip('/')}/{f.relative_to(local).as_posix()}"
        cur = rem.get(key)
        if cur and cur[0] == f.stat().st_size and (cur[1] == md5(f) or "-" in cur[1]):
            continue
        s3.upload_file(str(f), B, key); n += 1
    print(f"uploaded {n} files to s3://<bucket>/{prefix}")


def down(prefix: str, local: str) -> None:
    n = 0
    for key in remote(prefix):
        dest = Path(local) / key[len(prefix.rstrip('/')) + 1:]
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(B, key, str(dest)); n += 1
    print(f"downloaded {n} files from s3://<bucket>/{prefix}")


if __name__ == "__main__":
    {"up": up, "down": down}[sys.argv[1]](sys.argv[2], sys.argv[3])
