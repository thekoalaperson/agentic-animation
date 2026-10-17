#!/usr/bin/env python3
"""Coordinate one authorized finite Suno batch on a qualified GPU host.

Credentials arrive on stdin and remain in memory. No top-up and no automatic
generation retry; persisted task IDs own recovery. Media decoding stays remote.

Every invocation, --poll-only included, writes a new
credit-preflight-attempt-NNN.json and execution-receipt-attempt-NNN.json;
execution-receipt.json lists every attempt's wallet readings.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

from kie_music_finite import execute, load_plan, _atomic_json, _sha_file
from _common import adopt_legacy_record, attempt_records, write_attempt_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--max-requests", type=int, required=True)
    ap.add_argument("--max-credits", type=int, required=True)
    ap.add_argument("--poll-only", action="store_true")
    ap.add_argument("--approval-admission", type=Path)
    a = ap.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Run media execution on qualified GPU worker, not the Mac")
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                   check=True, capture_output=True, timeout=30)
    manifest = a.manifest.resolve()
    # load_plan verifies authority and creativeBindings, so a dry run sees a change.
    plan = load_plan(manifest, a.max_requests, a.max_credits)
    doc = plan["doc"]
    if not (plan["authority"] or {}).get("authority"):
        raise SystemExit("This wrapper requires an authority and authoritySha256 binding")
    key = json.load(sys.stdin)["kieApiKey"]
    folder = manifest.parent
    for stem in ("execution-receipt", "credit-preflight"):
        adopt_legacy_record(folder, stem)

    def receipt(status, before, after, **extra):
        write_attempt_json(folder, "execution-receipt", {
            "manifestSha256": plan["sha256"], "status": status, "pollOnly": a.poll_only,
            "walletBeforeThisInvocation": before, "walletAfter": after,
            "observedWalletDelta": before - after, "topUpAttempted": False,
            "accountingNote": "Shared wallet observation, not guaranteed per-job attribution",
            **extra})
        _atomic_json(folder / "execution-receipt.json", {
            "manifestSha256": plan["sha256"], "status": status,
            "attempts": attempt_records(folder, "execution-receipt"),
            "preflights": attempt_records(folder, "credit-preflight")})

    def wallet():
        req = urllib.request.Request("https://api.kie.ai/api/v1/chat/credit",
            headers={"Authorization": "Bearer " + key, "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)
        if out.get("code") != 200:
            raise RuntimeError("Wallet unavailable; do not submit")
        return float(out["data"])

    state_path = manifest.with_suffix(".state.json")
    before = wallet()
    if not a.poll_only:
        if state_path.exists():
            raise SystemExit("Existing batch: use --poll-only, never start a duplicate")
        if before < plan["expectedCredits"]:
            raise SystemExit("Insufficient existing credits; no top-up/overage")
    write_attempt_json(folder, "credit-preflight", {
        "manifestSha256": plan["sha256"], "availableCredits": before,
        "expectedCredits": plan["expectedCredits"], "authoritySha256": doc["authoritySha256"],
        "pollOnly": a.poll_only, "computeHost": platform.node(), "topUpAttempted": False})
    if not a.poll_only:
        print(json.dumps(execute(manifest, a.max_requests, a.max_credits,
                                 "submit", api_key=key, approval_admission=a.approval_admission)), flush=True)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        result = execute(manifest, a.max_requests, a.max_credits, "poll", api_key=key, approval_admission=a.approval_admission)
        state = json.loads(state_path.read_text())
        statuses = {k: v["status"] for k, v in state["items"].items()}
        print(json.dumps({"poll": result, "statuses": statuses}), flush=True)
        if all(s == "DOWNLOADED" for s in statuses.values()):
            receipt("COMPLETED", before, wallet(), requests=len(statuses),
                    candidates=sum(len(i["candidates"]) for i in state["items"].values()))
            return
        time.sleep(15)
    receipt("DEADLINE", before, wallet())
    raise SystemExit("Observation deadline reached; recover the same task IDs with --poll-only")


if __name__ == "__main__":
    main()
