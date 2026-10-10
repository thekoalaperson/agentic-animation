#!/usr/bin/env python3
"""Finite, resumable ElevenLabs Hindi narration batch runner.

The command is a dry run unless ``--submit`` is present.  It intentionally
does not query subscriptions, enable overages, top up an account, choose a
fallback format, or retry a POST.  An interrupted POST is cost-ambiguous and
therefore remains blocked for manual reconciliation.

Manifest schema (item paths are relative to the manifest directory and must
remain below ``episodeRoot``):

    {
      "version": 1,
      "language": "hi",
      "voiceId": "...",
      "modelId": "eleven_multilingual_v2",
      "outputFormat": "mp3_44100_192",
      "voiceSettings": {
        "stability": 0.6,
        "similarity_boost": 0.8,
        "style": 0.0,
        "use_speaker_boost": true,
        "speed": 0.9
      },
      "items": [
        {"id": "narration-part-01", "textFile": "part-01-hi.txt",
         "output": "audio/part-01-hi.mp3"}
      ]
    }

An item may contain ``text`` instead of ``textFile``, but never both.  Voice,
model, output format and voice settings are root-level so a batch cannot
silently drift between items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

if __package__:
    from ._common import env_key
    from .paid_approval import PaidApprovalError, validate_paid_approval
else:
    from _common import env_key
    from paid_approval import PaidApprovalError, validate_paid_approval


API_ROOT = "https://api.elevenlabs.io/v1/text-to-speech"
ALLOWED_MODEL = "eleven_multilingual_v2"
ALLOWED_FORMAT = "mp3_44100_192"
STATE_VERSION = 1
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class PlanError(ValueError):
    """The manifest or durable state is unsafe to execute."""


class UncertainPost(RuntimeError):
    """A POST may have reached ElevenLabs and must not be retried."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_new_bytes(path: Path, raw: bytes) -> None:
    """Publish new bytes atomically without ever replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.new")
    try:
        with tmp.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        # Linking a complete same-directory temp file makes publication atomic,
        # while link()'s EEXIST behavior closes the check-then-replace race.
        os.link(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _now() -> float:
    return time.time()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _verified_binding(manifest_path: Path, doc: dict[str, Any],
                      path_key: str, sha_key: str) -> dict[str, str] | None:
    raw_path = doc.get(path_key)
    raw_sha = doc.get(sha_key)
    if raw_path is None and raw_sha is None:
        return None
    if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha, str):
        raise PlanError(f"{path_key} and {sha_key} must be supplied together")
    rel = Path(raw_path)
    if rel.is_absolute():
        raise PlanError(f"{path_key} must be relative")
    base_mode = doc.get("provenancePathBase")
    if base_mode == "manifest":
        base = manifest_path.parent
    elif base_mode in {"episode", "project"}:
        episode_value = doc.get("projectRoot", doc.get("episodeRoot"))
        if (not isinstance(episode_value, str) or not episode_value
                or Path(episode_value).is_absolute()):
            raise PlanError("episode provenance requires a relative episodeRoot")
        base = (manifest_path.parent / episode_value).resolve()
        if not base.is_dir():
            raise PlanError("episode provenance root is missing")
    else:
        raise PlanError(
            "provenancePathBase must be 'project', 'manifest' or 'episode' when provenance is declared"
        )
    path = (base / rel).resolve()
    if not _is_below(path, base):
        raise PlanError(f"{path_key} escapes its declared provenance base")
    if not path.is_file():
        raise PlanError(f"{path_key} is missing: {path}")
    actual = _sha256_file(path)
    if actual != raw_sha.lower():
        raise PlanError(f"{path_key} changed after the narration manifest was frozen")
    return {"kind": path_key, "path": str(path), "sha256": actual}


def _validate_voice_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError("voiceSettings must be an object")
    expected = {"stability", "similarity_boost", "style",
                "use_speaker_boost", "speed"}
    if set(value) != expected:
        raise PlanError(f"voiceSettings keys must be exactly {sorted(expected)}")
    for key in ("stability", "similarity_boost", "style"):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)):
            raise PlanError(f"voiceSettings.{key} must be numeric")
        if not 0 <= float(value[key]) <= 1:
            raise PlanError(f"voiceSettings.{key} must be between 0 and 1")
    if not isinstance(value["use_speaker_boost"], bool):
        raise PlanError("voiceSettings.use_speaker_boost must be boolean")
    if isinstance(value["speed"], bool) or not isinstance(value["speed"], (int, float)):
        raise PlanError("voiceSettings.speed must be numeric")
    if not 0.7 <= float(value["speed"]) <= 1.2:
        raise PlanError("voiceSettings.speed must be between 0.7 and 1.2")
    return dict(value)


def _hindi_text(text: Any, item_id: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise PlanError(f"{item_id}: narration text must be non-empty")
    text = text.strip()
    return text


def load_plan(manifest_path: Path, max_characters: int) -> dict[str, Any]:
    if max_characters <= 0:
        raise PlanError("--max-characters must be a positive integer")
    manifest_path = manifest_path.resolve()
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read manifest: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise PlanError("manifest version must be 1")
    if not isinstance(doc.get("financialApproval"), bool):
        raise PlanError("financialApproval must be an explicit boolean")
    if doc.get("language") != "hi":
        raise PlanError("finite narration batches are Hindi-only (language must be 'hi')")
    voice_id = doc.get("voiceId")
    if not isinstance(voice_id, str) or not SAFE_ID.fullmatch(voice_id):
        raise PlanError("voiceId is missing or malformed")
    if doc.get("modelId") != ALLOWED_MODEL:
        raise PlanError(f"modelId must be {ALLOWED_MODEL}")
    if doc.get("outputFormat") != ALLOWED_FORMAT:
        raise PlanError(f"outputFormat must be {ALLOWED_FORMAT}; no quality fallback")
    settings = _validate_voice_settings(doc.get("voiceSettings"))
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        raise PlanError("items must be a non-empty array")

    root = manifest_path.parent
    episode_root_raw = doc.get("episodeRoot", ".")
    if not isinstance(episode_root_raw, str) or not episode_root_raw:
        raise PlanError("episodeRoot must be a non-empty relative path")
    if Path(episode_root_raw).is_absolute():
        raise PlanError("episodeRoot must be relative to the manifest")
    episode_root = (root / episode_root_raw).resolve()
    if not episode_root.is_dir():
        raise PlanError(f"episodeRoot is not a directory: {episode_root}")
    bindings = [
        binding for binding in (
            _verified_binding(manifest_path, doc, "sourcePlan", "sourcePlanSha256"),
            _verified_binding(
                manifest_path, doc, "productionOverrides", "productionOverridesSha256"
            ),
        ) if binding is not None
    ]
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    outputs: set[Path] = set()
    total = 0
    allowed_item_keys = {"id", "text", "textFile", "output"}
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            raise PlanError(f"item {pos}: must be an object")
        extra = set(item) - allowed_item_keys
        if extra:
            raise PlanError(f"item {pos}: unsupported per-item fields {sorted(extra)}")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not SAFE_ID.fullmatch(item_id):
            raise PlanError(f"item {pos}: id is missing or malformed")
        if item_id in ids:
            raise PlanError(f"duplicate item id: {item_id}")
        ids.add(item_id)
        has_text = "text" in item
        has_file = "textFile" in item
        if has_text == has_file:
            raise PlanError(f"{item_id}: specify exactly one of text or textFile")
        if has_file:
            if not isinstance(item["textFile"], str) or not item["textFile"]:
                raise PlanError(f"{item_id}: textFile must be a non-empty relative path")
            if Path(item["textFile"]).is_absolute():
                raise PlanError(f"{item_id}: textFile must be relative")
            text_path = (root / item["textFile"]).resolve()
            if not _is_below(text_path, episode_root):
                raise PlanError(f"{item_id}: textFile escapes episodeRoot")
            try:
                raw_text = text_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PlanError(f"{item_id}: cannot read textFile: {exc}") from exc
        else:
            text_path = None
            raw_text = item["text"]
        text = _hindi_text(raw_text, item_id)
        output_value = item.get("output")
        if not isinstance(output_value, str) or not output_value:
            raise PlanError(f"{item_id}: output must be a non-empty relative path")
        output_rel = Path(output_value)
        if output_rel.is_absolute() or output_rel.suffix.lower() != ".mp3":
            raise PlanError(f"{item_id}: output must be a relative .mp3 path")
        output = (root / output_rel).resolve()
        if not _is_below(output, episode_root):
            raise PlanError(f"{item_id}: output escapes episodeRoot")
        if output in outputs:
            raise PlanError(f"duplicate output path: {output_value}")
        outputs.add(output)
        chars = len(text)
        total += chars
        normalized.append({"id": item_id, "text": text, "textFile":
                           str(text_path) if text_path else None,
                           "output": str(output), "characters": chars})
    if total > max_characters:
        raise PlanError(f"aggregate characters {total} exceed explicit cap {max_characters}")
    declared_total = doc.get("aggregateCharacters")
    if (not isinstance(declared_total, int) or isinstance(declared_total, bool)
            or declared_total <= 0):
        raise PlanError("aggregateCharacters must be a positive integer")
    if declared_total != total:
        raise PlanError(
            f"declared aggregateCharacters {declared_total} does not equal computed {total}"
        )
    if max_characters != declared_total:
        raise PlanError(
            f"--max-characters must equal frozen aggregateCharacters {declared_total}"
        )

    config = {"language": "hi", "voiceId": voice_id,
              "modelId": ALLOWED_MODEL, "outputFormat": ALLOWED_FORMAT,
              "voiceSettings": settings}
    return {"manifest": str(manifest_path), "manifestSha256": _sha256_file(manifest_path),
            "config": config, "configSha256": _canonical_sha(config),
            "items": normalized, "aggregateCharacters": total,
            "maxCharacters": max_characters,
            "episodeRoot": str(episode_root),
            "sourceBindings": bindings,
            "sourceBindingsSha256": _canonical_sha(bindings),
            "financialApproval": doc.get("financialApproval") is True,
            "episode": doc.get("episode"),
            "approvalAdmissionSha256": None, "approvalAuthorizationId": None}


def _initial_state(plan: dict[str, Any]) -> dict[str, Any]:
    return {"version": STATE_VERSION, "manifest": plan["manifest"],
            "manifestSha256": plan["manifestSha256"],
            "configSha256": plan["configSha256"],
            "aggregateCharacters": plan["aggregateCharacters"],
            "sourceBindingsSha256": plan["sourceBindingsSha256"],
            "approvalAdmissionSha256": (
                plan["approvalAdmission"]["sha256"]
                if plan.get("approvalAdmission") else None
            ),
            "approvalAuthorizationId": (
                plan["approvalAdmission"]["authorizationId"]
                if plan.get("approvalAdmission") else None
            ),
            "maxCharacters": plan["maxCharacters"], "createdAt": _now(),
            "updatedAt": _now(),
            "items": {item["id"]: {"status": "PLANNED",
                                     "characters": item["characters"],
                                     "output": item["output"]}
                      for item in plan["items"]}}


def _load_or_create_state(plan: dict[str, Any], state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"cannot read durable state: {exc}") from exc
        if state.get("version") != STATE_VERSION:
            raise PlanError("durable state version mismatch")
        for field in ("manifestSha256", "configSha256", "aggregateCharacters",
                      "maxCharacters", "sourceBindingsSha256",
                      "approvalAdmissionSha256", "approvalAuthorizationId"):
            if state.get(field) != plan[field]:
                raise PlanError(f"durable state does not match current plan: {field}")
        if set(state.get("items", {})) != {x["id"] for x in plan["items"]}:
            raise PlanError("durable state item set does not match manifest")
        return state
    state = _initial_state(plan)
    _atomic_json(state_path, state)
    return state


def validate_mp3(path: Path, minimum_seconds: float = 0.05) -> dict[str, Any]:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format",
             "-of", "json", str(path)], check=True, capture_output=True, text=True)
        info = json.loads(probe.stdout)
        audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        if not audio:
            raise PlanError(f"response contains no audio stream: {path}")
        duration = float((info.get("format") or {}).get("duration") or
                         audio[0].get("duration") or 0)
        if duration < minimum_seconds:
            raise PlanError(f"response audio is empty or truncated: {path}")
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-map", "0:a:0", "-f", "null", "-"], check=True,
                       capture_output=True)
    except (OSError, subprocess.CalledProcessError, ValueError,
            json.JSONDecodeError) as exc:
        raise PlanError(f"MP3 decode validation failed for {path}: {exc}") from exc
    return {"durationSeconds": duration, "audioStreams": len(audio),
            "codec": audio[0].get("codec_name")}


def _post_once(url: str, body: dict[str, Any], api_key: str) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"xi-api-key": api_key, "Content-Type": "application/json",
                 "Accept": "audio/mpeg"}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
        return raw, {"status": response.status,
                     "requestId": response.headers.get("request-id"),
                     "contentType": response.headers.get("content-type")}


def _persist(state_path: Path, state: dict[str, Any]) -> None:
    state["updatedAt"] = _now()
    _atomic_json(state_path, state)


def _verify_resumable(plan: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for item in plan["items"]:
        record = state["items"][item["id"]]
        status = record.get("status")
        output = Path(item["output"])
        if status == "COMPLETED":
            expected = record.get("sha256")
            if not output.is_file() or not expected or _sha256_file(output) != expected:
                raise PlanError(f"{item['id']}: completed output missing or hash changed")
            continue
        if status != "PLANNED":
            raise UncertainPost(
                f"{item['id']}: state is {status}; reconcile it manually, never auto-retry")
        if output.exists():
            raise PlanError(f"{item['id']}: untracked output already exists: {output}")
        pending.append(item)
    return pending


def execute(manifest_path: Path, max_characters: int, submit: bool,
            state_path: Path | None = None,
            post_fn: Callable[[str, dict[str, Any], str],
                              tuple[bytes, dict[str, Any]]] | None = None,
            api_key: str | None = None,
            approval_admission: Path | None = None) -> dict[str, Any]:
    plan = load_plan(manifest_path, max_characters)
    try:
        approval = validate_paid_approval(
            approval_admission,
            episode=plan["episode"],
            lane="elevenlabs-narration",
            manifest_path=Path(plan["manifest"]),
            manifest_sha256=plan["manifestSha256"],
            finite_execution={
                "items": len(plan["items"]),
                "characters": plan["aggregateCharacters"],
            },
        )
    except PaidApprovalError as exc:
        raise PlanError(str(exc)) from exc
    plan["approvalAdmission"] = approval
    plan["approvalAdmissionSha256"] = approval["sha256"] if approval else None
    plan["approvalAuthorizationId"] = approval["authorizationId"] if approval else None
    plan["financialAuthorized"] = plan["financialApproval"] or approval is not None
    default_state = Path(plan["manifest"]).with_suffix(".eleven-state.json")
    state_path = (state_path or default_state).resolve()
    if state_path != default_state.resolve():
        raise PlanError(
            f"custom state paths are forbidden for paid batches; use canonical {default_state}"
        )
    if not submit:
        return {"mode": "DRY_RUN", "networkCalls": 0,
                "financialApproval": plan["financialApproval"],
                "financialAuthorized": plan["financialAuthorized"],
                "approvalAdmissionSha256": (
                    approval["sha256"] if approval else None
                ),
                "manifest": plan["manifest"],
                "aggregateCharacters": plan["aggregateCharacters"],
                "maxCharacters": plan["maxCharacters"],
                "items": [{"id": x["id"], "characters": x["characters"],
                           "output": x["output"]} for x in plan["items"]]}

    if not plan["financialAuthorized"]:
        raise PlanError(
            "financialApproval is false and no current approval admission was supplied; "
            "submission is gated"
        )

    # The lock is owned by the immutable manifest, never by a caller-selected
    # state path.  That makes a second runner for the same paid batch collide.
    lock_path = Path(plan["manifest"]).with_suffix(".eleven-execution.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PlanError(f"batch lock exists; verify no runner is active: {lock_path}") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.fsync(lock_fd)
        os.close(lock_fd)
        state = _load_or_create_state(plan, state_path)
        pending = _verify_resumable(plan, state)
        if not pending:
            return {"mode": "SUBMIT", "networkCalls": 0,
                    "status": "ALREADY_COMPLETED", "state": str(state_path)}
        fn = post_fn or _post_once
        if api_key is None:
            try:
                api_key = env_key("ELEVENLABS_API_KEY")
            except (SystemExit, OSError, ValueError) as exc:
                raise PlanError(str(exc)) from exc
        if not api_key:
            raise PlanError("ELEVENLABS_API_KEY is missing from the explicit credential sources")
        pending_root = state_path.parent / f".{state_path.stem}-responses"
        calls = 0
        config = plan["config"]
        url = (f"{API_ROOT}/{urllib.parse.quote(config['voiceId'], safe='')}"
               f"?output_format={urllib.parse.quote(config['outputFormat'], safe='')}")
        for item in pending:
            record = state["items"][item["id"]]
            body = {"text": item["text"], "model_id": config["modelId"],
                    "voice_settings": config["voiceSettings"]}
            record.update(status="POSTING_UNCERTAIN", postAttempt=1,
                          requestSha256=_canonical_sha(body), postStartedAt=_now())
            _persist(state_path, state)  # durable before the network boundary
            calls += 1
            try:
                raw, response_meta = fn(url, body, api_key)
            except urllib.error.HTTPError as exc:
                detail = exc.read(1024).decode("utf-8", "replace")
                if 400 <= exc.code < 500:
                    record.update(status="HTTP_4XX_REFUSED", httpStatus=exc.code,
                                  responseExcerpt=detail, postFinishedAt=_now())
                    _persist(state_path, state)
                    raise PlanError(f"{item['id']}: HTTP {exc.code}; automatic retry forbidden") from exc
                record.update(status="UNCERTAIN_HTTP_ERROR", httpStatus=exc.code,
                              responseExcerpt=detail, postFinishedAt=_now())
                _persist(state_path, state)
                raise UncertainPost(f"{item['id']}: HTTP {exc.code}; reconcile before retry") from exc
            except Exception as exc:
                record.update(status="UNCERTAIN_NETWORK_ERROR",
                              error=f"{type(exc).__name__}: {exc}", postFinishedAt=_now())
                _persist(state_path, state)
                raise UncertainPost(f"{item['id']}: POST outcome uncertain; reconcile before retry") from exc

            response_sha = _sha256_bytes(raw)
            staged = pending_root / f"{item['id']}-{response_sha[:16]}.mp3"
            _atomic_bytes(staged, raw)
            record.update(status="RESPONSE_CAPTURED", postFinishedAt=_now(),
                          responseSha256=response_sha, stagedResponse=str(staged),
                          response=response_meta)
            _persist(state_path, state)  # durable immediately after response capture
            try:
                # A tiny but decodable MP3 is not a narration response.  This
                # conservative Hindi-character floor catches severe truncation
                # while leaving wide room for speaking-rate variation.
                media = validate_mp3(staged, max(0.5, item["characters"] / 40.0))
            except Exception as exc:
                record.update(status="RESPONSE_INVALID", validationError=str(exc))
                _persist(state_path, state)
                raise
            record.update(status="RESPONSE_VALIDATED", media=media)
            _persist(state_path, state)
            output = Path(item["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                record.update(status="OUTPUT_CONFLICT")
                _persist(state_path, state)
                raise PlanError(f"refusing to overwrite output created during run: {output}")
            # The final temp file must live beside the destination so publication
            # remains atomic even if the work tree crosses filesystem boundaries.
            # Keep the captured response as reconciliation evidence.
            try:
                _atomic_new_bytes(output, raw)
            except FileExistsError as exc:
                record.update(status="OUTPUT_CONFLICT")
                _persist(state_path, state)
                raise PlanError(f"refusing to overwrite output created during run: {output}") from exc
            final_sha = _sha256_file(output)
            if final_sha != response_sha:
                record.update(status="OUTPUT_HASH_MISMATCH", outputSha256=final_sha)
                _persist(state_path, state)
                raise PlanError(f"{item['id']}: atomic output hash mismatch")
            record.update(status="COMPLETED", outputSha256=final_sha,
                          sha256=final_sha, completedAt=_now())
            _persist(state_path, state)
        return {"mode": "SUBMIT", "networkCalls": calls, "status": "COMPLETED",
                "state": str(state_path), "aggregateCharacters": plan["aggregateCharacters"],
                "approvalAdmissionSha256": approval["sha256"] if approval else None}
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-characters", type=int, required=True,
                        help="explicit hard cap for all manifest text")
    parser.add_argument("--submit", action="store_true",
                        help="perform at most one POST for each pending item")
    parser.add_argument("--approval-admission", type=Path,
                        help="hash-bound external paid authorization record")
    args = parser.parse_args(argv)
    try:
        result = execute(
            args.manifest,
            args.max_characters,
            args.submit,
            approval_admission=args.approval_admission,
        )
    except (PlanError, UncertainPost) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
