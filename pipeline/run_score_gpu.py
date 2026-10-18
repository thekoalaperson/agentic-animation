#!/usr/bin/env python3
"""Coordinate one authorized finite Suno batch on a qualified GPU host.

Credentials arrive on stdin and remain in memory. No top-up and no automatic
generation retry; persisted task IDs own recovery. Media decoding stays remote.

Every invocation, --poll-only included, writes a new
credit-preflight-attempt-NNN.json and execution-receipt-attempt-NNN.json;
execution-receipt.json lists every attempt's wallet readings. A full receipt or
preflight left by the older single-file wrapper is first kept as an attempt
marked legacyRecord.

A fresh batch posts one cue, waits for it to land, records the wallet delta
beside creditsPerRequest (report-only: the Suno poll carries no per-task
credits), then posts the rest. If this wrapper stops between the two, the
remainder is posted by kie_music_finite.py --submit (which posts only PLANNED
cues) and observed with --poll-only here; --poll-only itself never posts.

A cue that extends another waits (WAITING_PARENT) until its parent lands; this
wrapper keeps posting and observing waves until no cue is left waiting, and
stops if a wave posts nothing.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

from kie_music_finite import cue_credits, execute, load_plan, _atomic_json, _sha_file
from _common import adopt_legacy_record, attempt_records, write_attempt_json


UNPOSTED = {"PLANNED", "WAITING_PARENT"}


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

    def run(operation):
        return execute(manifest, a.max_requests, a.max_credits, operation,
                       api_key=key, approval_admission=a.approval_admission)

    def observe():
        """Poll until every posted cue has landed; PLANNED cues are not posted yet."""
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            result = run("poll")
            state = json.loads(state_path.read_text())
            statuses = {k: v["status"] for k, v in state["items"].items()}
            print(json.dumps({"poll": result, "statuses": statuses}), flush=True)
            if all(s in UNPOSTED | {"DOWNLOADED"} for s in statuses.values()):
                return state, statuses
            time.sleep(15)
        receipt("DEADLINE", before, wallet())
        raise SystemExit("Observation deadline reached; recover the same task IDs with --poll-only")

    if not a.poll_only:
        print(json.dumps(run("submit")), flush=True)
    state, statuses = observe()
    planned = [k for k, s in statuses.items() if s in UNPOSTED]
    if planned and a.poll_only:
        receipt("AWAITING_REMAINDER", before, wallet(), notYetPosted=planned)
        landed = "Posted cues have landed" if len(planned) < len(statuses) else "No cue is posted yet"
        raise SystemExit(f"{landed}; post the rest with kie_music_finite.py "
                         "--submit, then observe them here with --poll-only")
    if planned:
        pilot = [k for k, s in statuses.items() if s == "DOWNLOADED"]
        if not pilot:
            raise SystemExit("No cue was posted; inspect the batch state before submitting")
        after_pilot = wallet()
        rate = doc["creditsPerRequest"]
        delta = before - after_pilot
        receipt("FIRST_CUE_LANDED", before, after_pilot, firstCues=pilot,
                creditsPerRequest=rate, observedCreditsPerLandedCue=delta / len(pilot),
                comparison="report-only: the Suno poll records no per-task credits; "
                           "a shared wallet delta is an observation, not a receipt")
        available = after_pilot
    while planned:
        remaining = sum(cue_credits(doc, cue) for cue in plan["cues"] if cue["id"] in planned)
        if available < remaining:
            raise SystemExit("Insufficient existing credits for the remaining cues; no top-up/overage")
        print(json.dumps(run("submit")), flush=True)
        state, statuses = observe()
        still = [k for k, s in statuses.items() if s in UNPOSTED]
        if still == planned:
            receipt("AWAITING_REMAINDER", before, wallet(), notYetPosted=still)
            raise SystemExit("A wave posted nothing; inspect the batch state before submitting")
        planned = still
        if planned:
            available = wallet()
    receipt("COMPLETED", before, wallet(), requests=len(statuses),
            candidates=sum(len(i["candidates"]) for i in state["items"].values()))


if __name__ == "__main__":
    main()
