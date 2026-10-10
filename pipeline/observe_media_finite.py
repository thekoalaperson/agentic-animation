#!/usr/bin/env python3
"""Bounded blind media descriptions through the studio's existing review lane.

Observations are evidence, never approvals. Each job makes at most one model
call; all inputs and responses are hashed. No generation or publication occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

if __package__:
    from ._common import env_key
    from ._common import upload_ref
else:
    from _common import env_key
    from _common import upload_ref

ENDPOINT = "https://api.kie.ai/gemini-3-6-flash-openai/v1/chat/completions"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def run(path: Path, count: int, submit: bool):
    path = path.resolve()
    plan = json.loads(path.read_text())
    plan_sha = sha(path)
    jobs = plan["jobs"]
    if not jobs or len(jobs) != count or count > 160:
        raise ValueError("command count must equal the concrete review job count (1–160)")
    root_value = plan.get("projectRoot")
    if not isinstance(root_value, str) or not root_value or Path(root_value).is_absolute():
        raise ValueError("projectRoot must be explicitly relative to the observation manifest")
    root = (path.parent / root_value).resolve()
    if not root.is_dir() or not path.is_relative_to(root):
        raise ValueError("Observation manifest must live inside its project root")
    output = path.parent / (path.stem + "-observations")
    ids = set()
    for job in jobs:
        if not re.fullmatch(r"[a-z0-9-]+", job["id"]) or job["id"] in ids:
            raise ValueError("duplicate or malformed review job ID")
        ids.add(job["id"])
        if not job["prompt"] or not 1 <= len(job["media"]) <= 12:
            raise ValueError("review requires a concrete prompt and 1–12 media files")
        for media in job["media"]:
            source = (root / media["path"]).resolve()
            source.relative_to(root)
            if sha(source) != media["sha256"]:
                raise ValueError(f"review input changed: {source}")
            if media.get("url") and not media["url"].startswith("https://"):
                raise ValueError("provider media URL must be HTTPS")
    if not submit:
        return {"jobs": len(jobs), "networkCalls": 0, "output": str(output)}
    if plan.get("financialApproval") is not True:
        raise ValueError("Explicit financialApproval for this finite review plan is required")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        calls = 0
        for job in jobs:
            target = output / (job["id"] + ".json")
            if target.exists():
                saved = json.loads(target.read_text())
                if saved.get("planSha256") != plan_sha or saved.get("status") != "COMPLETED":
                    raise ValueError(f"review needs reconciliation before any retry: {target}")
                continue
            parts = [{"type": "text", "text": job["prompt"]}]
            sources = []
            for index, media in enumerate(job["media"], 1):
                source = root / media["path"]
                url = media.get("url") or upload_ref(source)
                parts += [{"type": "text", "text": f"Media {index}:"},
                          {"type": "image_url", "image_url": {"url": url}}]
                sources.append({**media, "url": url})
            request = {"model": "gemini-3-6-flash", "stream": False,
                       "max_tokens": job.get("maxTokens", 2400),
                       "messages": [{"role": "user", "content": parts}]}
            state = {"version": 1, "status": "POSTING_UNCERTAIN", "jobId": job["id"],
                     "planSha256": plan_sha, "sources": sources, "request": request,
                     "startedAt": time.time(), "advisoryOnly": True}
            save(target, state)
            req = urllib.request.Request(ENDPOINT, data=json.dumps(request).encode(),
                headers={"Authorization": "Bearer " + env_key("KIE_KEY"),
                         "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
            state.update(response=result, finishedAt=time.time())
            text = ((result.get("choices") or [{}])[0].get("message") or {}).get("content")
            if not text:
                text = "\n".join(p.get("text", "") for c in result.get("candidates", [])
                                 for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
            if not text:
                state["status"] = "RESPONSE_INVALID"
                save(target, state)
                raise ValueError(f"review returned no observation: {job['id']}")
            state.update(status="COMPLETED", observation=text,
                         creditsConsumed=result.get("credits_consumed"))
            save(target, state)
            calls += 1
            print(json.dumps({"id": job["id"], "status": "COMPLETED",
                              "creditsConsumed": state["creditsConsumed"]}), flush=True)
        return {"jobs": len(jobs), "reviewCalls": calls, "output": str(output)}
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.jobs, args.submit)))


if __name__ == "__main__":
    main()
