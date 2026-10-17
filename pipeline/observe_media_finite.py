#!/usr/bin/env python3
"""Bounded blind media descriptions through the studio's existing review lane.

Observations are evidence, never approvals. Each job makes at most one model
call; all inputs and responses are hashed. No generation or publication occurs.
A submission needs a hash-bound approval record for lane ``media-observation``
binding ``{"jobs": <count>, "credits": <plan maxCredits>}``; the plan's own
financialApproval never authorizes. The run stops once reported credits pass
the plan's ``maxCredits``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

if __package__:
    from ._common import env_key
    from ._common import upload_ref
    from .paid_approval import PaidApprovalError, validate_paid_approval
else:
    from _common import env_key
    from _common import upload_ref
    from paid_approval import PaidApprovalError, validate_paid_approval

ENDPOINT = "https://api.kie.ai/gemini-3-6-flash-openai/v1/chat/completions"
APPROVAL_LANE = "media-observation"


def approval_scope(plan):
    """The finite execution limits an approval record must bind exactly."""
    ceiling = plan.get("maxCredits")
    if (isinstance(ceiling, bool) or not isinstance(ceiling, (int, float))
            or not ceiling > 0 or ceiling == float("inf")):
        raise ValueError("observation plan needs an explicit positive maxCredits ceiling")
    return {"jobs": len(plan["jobs"]), "credits": ceiling}


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


def run(path: Path, count: int, submit: bool, approval_admission: Path | None = None):
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
    approval = None
    if submit or approval_admission is not None:
        try:
            approval = validate_paid_approval(
                approval_admission, episode=plan.get("episode"), lane=APPROVAL_LANE,
                manifest_path=path, manifest_sha256=plan_sha,
                finite_execution=approval_scope(plan))
        except PaidApprovalError as exc:
            raise ValueError(str(exc)) from exc
    if not submit:
        return {"jobs": len(jobs), "networkCalls": 0, "output": str(output),
                "financialApproval": plan.get("financialApproval") is True,
                "financialAuthorized": approval is not None}
    if approval is None:
        raise ValueError("submission requires a current approval record (--approval-admission); "
                         "the plan's financialApproval never authorizes spending")
    spent = 0.0
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
                spent += float(saved.get("creditsConsumed") or 0)
                continue
            if spent >= plan["maxCredits"]:
                raise ValueError(f"reported credits {spent} reached maxCredits "
                                 f"{plan['maxCredits']}; stopped before {job['id']}")
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
            spent += float(state["creditsConsumed"] or 0)
            print(json.dumps({"id": job["id"], "status": "COMPLETED",
                              "creditsConsumed": state["creditsConsumed"]}), flush=True)
        return {"jobs": len(jobs), "reviewCalls": calls, "output": str(output),
                "reportedCredits": spent,
                "approvalAdmissionSha256": approval["sha256"]}
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--approval-admission", type=Path,
                        help="hash-bound paid authorization record (paid_approval.py)")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.jobs, args.submit, args.approval_admission)))


if __name__ == "__main__":
    main()
