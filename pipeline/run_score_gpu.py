#!/usr/bin/env python3
"""Coordinate one authorized finite Suno batch on a qualified GPU host.

Credentials arrive on stdin and remain in memory. No top-up and no automatic
generation retry; persisted task IDs own recovery. Media decoding stays remote.
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
    plan = load_plan(manifest, a.max_requests, a.max_credits)
    doc = plan["doc"]
    authority = (manifest.parent / doc["authority"]).resolve()
    if _sha_file(authority) != doc["authoritySha256"]:
        raise SystemExit("Authority bytes changed")
    for binding in doc.get("creativeBindings", []):
        if _sha_file((manifest.parent / binding["path"]).resolve()) != binding["sha256"]:
            raise SystemExit("Creative/timing input changed")
    key = json.load(sys.stdin)["kieApiKey"]

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
        _atomic_json(manifest.parent / "credit-preflight.json", {
            "manifestSha256": plan["sha256"], "availableCredits": before,
            "expectedCredits": plan["expectedCredits"], "authoritySha256": doc["authoritySha256"],
            "computeHost": platform.node(), "topUpAttempted": False})
        print(json.dumps(execute(manifest, a.max_requests, a.max_credits,
                                 "submit", api_key=key, approval_admission=a.approval_admission)), flush=True)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        result = execute(manifest, a.max_requests, a.max_credits, "poll", api_key=key, approval_admission=a.approval_admission)
        state = json.loads(state_path.read_text())
        statuses = {k: v["status"] for k, v in state["items"].items()}
        print(json.dumps({"poll": result, "statuses": statuses}), flush=True)
        if all(s == "DOWNLOADED" for s in statuses.values()):
            after = wallet()
            _atomic_json(manifest.parent / "execution-receipt.json", {
                "manifestSha256": plan["sha256"], "status": "COMPLETED",
                "requests": len(statuses), "candidates": sum(len(i["candidates"]) for i in state["items"].values()),
                "walletBeforeThisInvocation": before, "walletAfter": after,
                "observedWalletDelta": before-after, "pollOnly": a.poll_only,
                "accountingNote": "Shared wallet observation, not guaranteed per-job attribution",
                "topUpAttempted": False})
            return
        time.sleep(15)
    raise SystemExit("Observation deadline reached; recover the same task IDs with --poll-only")


if __name__ == "__main__":
    main()
