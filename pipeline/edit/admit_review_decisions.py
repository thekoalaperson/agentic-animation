#!/usr/bin/env python3
"""Write the source-bound record that admits reviewed takes to the timeline.

The timeline admits a dialogue or score take only through a schema-2 review
export, and H3 picture only through a ledger verdict. This tool writes either
one from a plain decisions file, recording whatever review was actually done.
It makes no media, calls no provider and never decides anything itself: a
take with no decision row is left as it was, and a note never becomes a
decision.

Decisions file (JSON)::

    {
      "reviewedBy": "agent:demo-reviewer",
      "reviewer": {"kind": "agent", "name": "demo-reviewer"},
      "method": "frame sheet",
      "decisions": [
        {"itemId": "demo-s003", "decision": "approved",
         "reviewedAt": "2026-01-01T09:00:00Z", "sha256": "<reviewed bytes>",
         "speechWindow": {"start": 1.0, "end": 3.5}, "note": "free text"}
      ]
    }

``reviewedBy`` is required, on the file or on every row, and is recorded
verbatim. A person's name goes there only in that person's own words; an agent
writes ``agent:<name>``. ``reviewer`` (``{kind: human|agent, name}`` or a list
of them) and ``method`` (open wording: full playback with sound, frame sheet,
transcript against script, speaker similarity, model observation, animatic
context, or anything else) are optional on the file or a row. ``reviewedAt`` is
a UTC ISO-8601 time. ``sha256`` and ``sidecarSha256`` name the bytes the
reviewer saw; when given they must match. Optional ``note`` and
``judgedAgainst`` are copied as written.

Per lane, a row also carries:

* ``dialogue`` -- ``itemId`` is the job ID in the dialogue manifest, or in the
  row's own ``executionManifest`` for a pilot or correction batch. Optional
  ``sceneId`` (default ``itemId``), ``reuseReason`` and ``speechWindow``.
* ``score`` -- ``itemId`` is the cue ID and ``take`` its candidate index. A
  placement already authored for that take is pointed at the new record;
  ``timelineStart`` and ``timelineDuration`` on the row add one.
* ``h3`` -- ``itemId`` is the job ID in ``--audit`` (``review_finite.py``'s
  ``local-audit.json``) and ``sceneId`` the storyboard row it covers, which may
  be left out only when it equals ``itemId``. Optional ``in``, ``out``,
  ``speed``, ``role`` and ``origin``; the origin defaults to the job's grouped
  ingest receipt when the audit names one.

The tool refuses when any source or sidecar hash has changed since it was
recorded, and refuses the whole file if any row is unusable.

Outputs, never overwritten: ``work/review/<lane>-review-vN.json`` for dialogue
and score, ``work/review/h3-review-ledger-vN.json`` for H3. The timeline inputs
are then rewritten to point at them, after the previous bytes are kept beside
them as ``timeline-inputs-vN.json``.

A review export's ``reviewFingerprint`` is the canonical SHA-256 (sorted keys,
compact separators, UTF-8) of ``{"reviewType", "sourceBindings", "reviews"}``
taken from the export itself. ``sourceBindings`` are the documents the timeline
checks for that lane, computed by the timeline's own ``review_binding_paths``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import atomic_new_bytes, canonical_sha256, is_below, sha256_file
from build_episode_timeline import (
    ACCEPTED, PICTURE_ROUTES, _dialogue_entries, _ledger_entries, _normal_verdict,
    _portable_path, _resolve_path, _state_path_matches, lane_documents,
    review_binding_paths,
)

LANES = ("dialogue", "score", "h3")
REVIEWER_KINDS = {"human", "agent"}
PASS_THROUGH = ("note", "judgedAgainst")


class AdmissionError(ValueError):
    """The decisions cannot be written as an honest, source-bound record."""


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"{label} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"{label} must be a JSON object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _next_version(directory: Path, stem: str) -> Path:
    pattern = re.compile(re.escape(stem) + r"-v(\d+)\.json\Z")
    used = [int(m.group(1)) for p in directory.glob(stem + "-v*.json")
            if (m := pattern.fullmatch(p.name))] if directory.is_dir() else []
    return directory / f"{stem}-v{max(used, default=0) + 1}.json"


def _utc(value: Any, item: str) -> str:
    text = str(value or "")
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        moment = None
    if (moment is None or moment.tzinfo is None or moment.utcoffset() != timezone.utc.utcoffset(None)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}.*", text)):
        raise AdmissionError(f"{item}: reviewedAt must be a UTC ISO-8601 time, got {value!r}")
    return text


def _reviewer(value: Any, item: str) -> Any:
    rows = value if isinstance(value, list) else [value]
    for row in rows:
        if (not isinstance(row, dict) or row.get("kind") not in REVIEWER_KINDS
                or not isinstance(row.get("name"), str) or not row["name"].strip()):
            raise AdmissionError(f"{item}: reviewer must be {{kind: human|agent, name}} or a list of them")
    return value


def _method(value: Any, item: str) -> Any:
    rows = value if isinstance(value, list) else [value]
    if not rows or any(not isinstance(row, str) or not row.strip() for row in rows):
        raise AdmissionError(f"{item}: method must be free text or a list of it")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AdmissionError(f"{label} must be a finite number")
    return float(value)


def _review_row(top: dict[str, Any], row: dict[str, Any], item: str) -> dict[str, Any]:
    """The fields every lane records about who decided what, and when."""
    decision = row.get("decision")
    if not isinstance(decision, str) or not decision.strip():
        raise AdmissionError(f"{item}: every row needs a decision; a note is never read as one")
    reviewed_by = row.get("reviewedBy", top.get("reviewedBy"))
    if not isinstance(reviewed_by, str) or not reviewed_by.strip():
        raise AdmissionError(f"{item}: reviewedBy is required, recorded verbatim")
    result = {"decision": decision, "reviewedAt": _utc(row.get("reviewedAt"), item),
              "reviewedBy": reviewed_by}
    reviewer = row.get("reviewer", top.get("reviewer"))
    if reviewer is None and reviewed_by.startswith("agent:") and reviewed_by[6:].strip():
        reviewer = {"kind": "agent", "name": reviewed_by[6:].strip()}
    if reviewer is not None:
        result["reviewer"] = _reviewer(reviewer, item)
    method = row.get("method", top.get("method"))
    if method is not None:
        result["method"] = _method(method, item)
    for key in PASS_THROUGH:
        if row.get(key) is not None:
            result[key] = row[key]
    return result


def _seen(row: dict[str, Any], key: str, actual: str, item: str, label: str) -> None:
    declared = row.get(key)
    if declared is not None and str(declared).lower() != actual:
        raise AdmissionError(f"{item}: the reviewed {label} hash {declared} is not the current file {actual}")


class Admission:
    def __init__(self, episode: Path, config_path: Path, decisions_path: Path, lane: str,
                 audit_path: Path | None = None):
        if lane not in LANES:
            raise AdmissionError(f"lane must be one of {', '.join(LANES)}")
        self.episode = episode.resolve()
        self.config_path = config_path.resolve()
        self.decisions_path = decisions_path.resolve()
        self.lane = lane
        self.audit_path = audit_path.resolve() if audit_path else None
        self.config = _read(self.config_path, "timeline inputs") if self.config_path.is_file() else {}
        self.decisions = _read(self.decisions_path, "decisions file")
        if self.decisions.get("lane", lane) != lane:
            raise AdmissionError(f"decisions file is for lane {self.decisions['lane']!r}, not {lane!r}")
        rows = self.decisions.get("decisions")
        if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
            raise AdmissionError("decisions must be a non-empty list of rows")
        self.rows = rows
        self.documents = {label: self.path(raw) for label, raw in
                          lane_documents(self.episode, self.episode, self.config_path, self.config).items()}
        self.report: dict[str, list[str]] = {"admitted": [], "notAdmitted": [], "unplaced": []}

    def path(self, raw: Any, owner: Path | None = None) -> Path:
        return _resolve_path(self.episode, self.episode, owner or self.config_path, str(raw))

    def rel(self, path: Path) -> str:
        return _portable_path(path, self.episode)

    def contained(self, path: Path, item: str, label: str) -> Path:
        if not is_below(path, self.episode):
            raise AdmissionError(f"{item}: {label} must sit below the project: {path}")
        if not path.is_file():
            raise AdmissionError(f"{item}: {label} is missing: {path}")
        return path

    def bindings(self, paths: list[Path]) -> list[dict[str, str]]:
        result = []
        for path in paths:
            self.contained(path, self.lane, "bound document")
            result.append({"path": self.rel(path), "sha256": sha256_file(path)})
        return result

    def tally(self, item: str, decision: str) -> None:
        self.report["admitted" if _normal_verdict(decision) in ACCEPTED else "notAdmitted"].append(item)

    # -- dialogue ---------------------------------------------------------
    def dialogue(self) -> tuple[dict[str, Any], dict[str, Any]]:
        canonical = self.documents["dialogueManifest"]
        production = self.documents.get("productionOverrides")
        outputs = dict(self.config.get("dialogueOutputs") or {})
        reviews, binding_paths, errors, seen = [], [], [], set()
        for index, row in enumerate(self.rows):
            item = str(row.get("itemId") or f"row {index}")
            try:
                if item in seen:
                    raise AdmissionError(f"{item}: decided twice in one file")
                seen.add(item)
                record = _review_row(self.decisions, row, item)
                execution = self.path(row["executionManifest"]) if row.get("executionManifest") else canonical
                self.contained(execution, item, "execution manifest")
                job = _dialogue_entries(_read(execution, "dialogue manifest")).get(item)
                if job is None or not job.get("outputStem"):
                    raise AdmissionError(f"{item}: no job with an outputStem in {self.rel(execution)}")
                stem = self.path(job["outputStem"])
                source = self.contained(stem.with_name(stem.name + ".mp4"), item, "take")
                sidecar = self.contained(stem.with_name(stem.name + ".vid.json"), item, "sidecar")
                source_sha, sidecar_sha = sha256_file(source), sha256_file(sidecar)
                meta = _read(sidecar, f"{item} sidecar")
                if meta.get("outputSha256") != source_sha or meta.get("sha256") != source_sha:
                    raise AdmissionError(f"{item}: the take's bytes changed since its sidecar was written")
                if meta.get("executionManifestSha256") != sha256_file(execution):
                    raise AdmissionError(f"{item}: the sidecar binds another execution manifest")
                _seen(row, "sha256", source_sha, item, "take")
                _seen(row, "sidecarSha256", sidecar_sha, item, "sidecar")
                window = row.get("speechWindow")
                if window is not None:
                    if not isinstance(window, dict):
                        raise AdmissionError(f"{item}: speechWindow must be {{start, end}} in seconds")
                    start = _number(window.get("start"), f"{item}: speechWindow start")
                    end = _number(window.get("end"), f"{item}: speechWindow end")
                    if not 0 <= start < end:
                        raise AdmissionError(f"{item}: speechWindow must run forward from zero or later")
                    record["speechWindow"] = {"start": start, "end": end}
                record.update(itemId=item, source={
                    "path": self.rel(source), "sha256": source_sha,
                    "sidecarPath": self.rel(sidecar), "sidecarSha256": sidecar_sha})
                reviews.append(record)
                for path in review_binding_paths("dialogue", dialogue_path=canonical,
                                                 execution_path=execution, production_path=production):
                    if path not in binding_paths:
                        binding_paths.append(path)
                scene = str(row.get("sceneId") or item)
                previous = outputs.get(scene) if isinstance(outputs.get(scene), dict) else {}
                # Editorial choices made on this exact take survive a re-review.
                entry = dict(previous) if previous.get("sha256") == source_sha else {}
                entry.update(path=self.rel(source), verdict=record["decision"], sha256=source_sha,
                             sidecar=self.rel(sidecar), sidecarSha256=sidecar_sha)
                if "speechWindow" in record:
                    entry["speechWindow"] = record["speechWindow"]
                for key in ("executionManifest", "executionManifestSha256", "sourceSceneId"):
                    entry.pop(key, None)
                if execution != canonical:
                    entry.update(executionManifest=self.rel(execution),
                                 executionManifestSha256=sha256_file(execution))
                if scene != item:
                    entry["sourceSceneId"] = item
                    if row.get("reuseReason"):
                        entry["reuseReason"] = row["reuseReason"]
                outputs[scene] = entry
                self.tally(scene, record["decision"])
            except (AdmissionError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        if errors:
            raise AdmissionError("; ".join(errors))
        return self.export(reviews, binding_paths), {"dialogueOutputs": outputs}

    # -- score ------------------------------------------------------------
    def score(self) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = self.path(self.config.get("scoreManifest", "work/music-score-v1/execution-manifest.json"))
        self.contained(manifest, "score", "score manifest")
        state_path = self.contained(manifest.with_suffix(".state.json"), "score", "score finite state")
        state = _read(state_path, "score finite state")
        if state.get("executionManifestSha256") != sha256_file(manifest):
            raise AdmissionError("score finite state does not bind the current score manifest")
        cues = {str(cue.get("id")) for cue in _read(manifest, "score manifest").get("cues", [])}
        original_root = self.config.get("scoreStateEpisodeRoot")
        placements = [dict(p) for p in self.config.get("scorePlacements") or []]
        reviews, errors, seen = [], [], set()
        for index, row in enumerate(self.rows):
            item = str(row.get("itemId") or f"row {index}")
            try:
                take = row.get("take")
                if isinstance(take, bool) or not isinstance(take, int) or take <= 0:
                    raise AdmissionError(f"{item}: take must be the candidate's positive index")
                label = f"{item} take {take}"
                if (item, take) in seen:
                    raise AdmissionError(f"{label}: decided twice in one file")
                seen.add((item, take))
                if item not in cues:
                    raise AdmissionError(f"{item}: no such cue in the score manifest")
                record = _review_row(self.decisions, row, label)
                rows = [c for c in ((state.get("items") or {}).get(item) or {}).get("candidates", [])
                        if c.get("index") == take and c.get("status") == "COMPLETE"]
                if len(rows) != 1:
                    raise AdmissionError(f"{label}: the finite state holds no single completed candidate")
                candidate = rows[0]
                track = self.locate(candidate.get("file"), original_root, label, "track")
                sidecar = self.locate(candidate.get("sidecar"), original_root, label, "sidecar")
                track_sha, sidecar_sha = sha256_file(track), sha256_file(sidecar)
                if track_sha != candidate.get("sha256") or sidecar_sha != candidate.get("sidecarSha256"):
                    raise AdmissionError(f"{label}: the track or sidecar changed since the finite state recorded it")
                meta = _read(sidecar, f"{label} sidecar")
                if (meta.get("sha256") != track_sha or meta.get("cueId") != item
                        or meta.get("candidateIndex") != take):
                    raise AdmissionError(f"{label}: the sidecar names other bytes or another take")
                _seen(row, "sha256", track_sha, label, "track")
                _seen(row, "sidecarSha256", sidecar_sha, label, "sidecar")
                record.update(itemId=item, take=take, source={
                    "path": self.rel(track), "sha256": track_sha,
                    "sidecarPath": self.rel(sidecar), "sidecarSha256": sidecar_sha})
                reviews.append(record)
                bound = dict(track=self.rel(track), trackSha256=track_sha,
                             sidecar=self.rel(sidecar), sidecarSha256=sidecar_sha)
                matched = [p for p in placements if p.get("movement") == item and p.get("take") == take]
                for placement in matched:
                    placement.update(bound)
                if not matched and row.get("timelineStart") is not None and row.get("timelineDuration") is not None:
                    placement = {"movement": item, "take": take, **bound,
                                 "timelineStart": _number(row["timelineStart"], f"{label}: timelineStart"),
                                 "timelineDuration": _number(row["timelineDuration"], f"{label}: timelineDuration")}
                    if row.get("placementId"):
                        placement["id"] = row["placementId"]
                    placements.append(placement)
                    matched = [placement]
                if not matched:
                    self.report["unplaced"].append(label)
                for placement in matched:
                    placement["_new"] = True
                self.tally(label, record["decision"])
            except (AdmissionError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        if errors:
            raise AdmissionError("; ".join(errors))
        binding_paths = review_binding_paths(
            "score", dialogue_path=self.documents["dialogueManifest"],
            narration_path=self.documents.get("narrationManifest"),
            score_manifest_path=manifest, production_path=self.documents.get("productionOverrides"))
        export = self.export(reviews, binding_paths)
        return export, {"scorePlacements": placements}

    def locate(self, recorded: Any, original_root: Any, item: str, label: str) -> Path:
        """Find a worker-recorded file below this project, by exact relative identity."""
        raw = Path(str(recorded or ""))
        if not str(recorded or ""):
            raise AdmissionError(f"{item}: the finite state records no {label}")
        if not raw.is_absolute():
            path = (self.episode / raw).resolve()
        elif is_below(raw, self.episode) or not original_root:
            path = raw.resolve()
        else:
            try:
                path = (self.episode / raw.relative_to(Path(str(original_root)))).resolve()
            except ValueError:
                path = raw.resolve()
        if not _state_path_matches(recorded, path, self.episode, original_root) and raw.is_absolute():
            raise AdmissionError(f"{item}: {label} {recorded} is not below this project; "
                                 "declare scoreStateEpisodeRoot for a relocated project")
        return self.contained(path, item, label)

    # -- h3 ---------------------------------------------------------------
    def h3(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.audit_path is None:
            raise AdmissionError("the h3 lane needs --audit, review_finite.py's local-audit.json")
        audit_path = self.contained(self.audit_path, "h3", "local audit")
        audit = _read(audit_path, "local audit")
        outputs = {str(r.get("id")): r for r in audit.get("outputs", []) if isinstance(r, dict)}
        board_path = self.documents["storyboard"]
        board = _read(board_path, "storyboard").get("stills") or []
        still_ids = [str(r.get("still_id")) for r in board if isinstance(r, dict)]
        raw_ledger = self.config.get("h3ReviewLedger", "work/h3-review-ledger-v1.json")
        previous_path = self.path(raw_ledger)
        if previous_path.is_file():
            old = _read(previous_path, "H3 review ledger")
            ledger = {k: v for k, v in old.items() if k not in ("entries", "shots")
                      and ("entries" in old or "shots" in old or not isinstance(v, dict))}
            ledger["entries"] = {k: dict(v, segments=list(v.get("segments") or []))
                                 for k, v in _ledger_entries(old).items()}
            ledger.setdefault("schemaVersion", 1)
        else:
            ledger = {"schemaVersion": 2, "entries": {}}
        version = ledger["schemaVersion"]
        if isinstance(version, bool) or not isinstance(version, int):
            version = 2  # as the timeline does: an unreadable version gets the stricter rule
        entries, errors, seen = ledger["entries"], [], set()
        for index, row in enumerate(self.rows):
            item = str(row.get("itemId") or f"row {index}")
            try:
                if item in seen:
                    raise AdmissionError(f"{item}: decided twice in one file")
                seen.add(item)
                record = _review_row(self.decisions, row, item)
                scene = row.get("sceneId") or (item if item in still_ids else None)
                if scene is None:
                    raise AdmissionError(f"{item}: name the storyboard row it covers as sceneId")
                if scene not in still_ids:
                    raise AdmissionError(f"{item}: sceneId {scene!r} is not a storyboard row")
                output = outputs.get(item)
                if not output or output.get("status") != "technical-pass" or not isinstance(output.get("silent"), dict):
                    raise AdmissionError(f"{item}: the local audit holds no technical-pass silent output")
                silent = output["silent"]
                recorded = Path(str(silent.get("file") or ""))
                source = recorded if is_below(recorded, self.episode) and recorded.is_file() \
                    else audit_path.parent / "silent" / recorded.name
                self.contained(source, item, "silent picture")
                source_sha = sha256_file(source)
                if source_sha != silent.get("sha256"):
                    raise AdmissionError(f"{item}: the silent picture changed since the local audit")
                _seen(row, "sha256", source_sha, item, "silent picture")
                segment = {"source": self.rel(source), "sourceSha256": source_sha,
                           "verdict": record.pop("decision"), "h3Item": item, **record}
                for key in ("in", "out", "speed"):
                    if row.get(key) is not None:
                        segment[key] = _number(row[key], f"{item}: {key}")
                if row.get("role") is not None:
                    segment["role"] = str(row["role"])
                origin = row.get("origin")
                if origin is None and silent.get("ingestReceiptSha256"):
                    receipt = self.contained(audit_path.parent / "silent" / "ingest-receipt.json",
                                             item, "ingest receipt")
                    if sha256_file(receipt) != silent["ingestReceiptSha256"]:
                        raise AdmissionError(f"{item}: the ingest receipt changed since the local audit")
                    origin = {"kind": "h3-ingest", "record": self.rel(receipt),
                              "recordSha256": silent["ingestReceiptSha256"]}
                if origin is not None:
                    segment["origin"] = origin
                elif version >= 2 and _normal_verdict(segment["verdict"]) in ACCEPTED:
                    raise AdmissionError(f"{item}: a version-{version} ledger admits a segment only with "
                                         "its origin; give origin on the row or ingest with a receipt")
                entry = entries.setdefault(scene, {"pictureRoute": "h3", "segments": []})
                if str(entry.get("pictureRoute", "")).lower() not in PICTURE_ROUTES:
                    raise AdmissionError(f"{item}: {scene} is routed {entry.get('pictureRoute')!r}, not H3 picture")
                kept = [s for s in entry["segments"] if s.get("sourceSha256") != source_sha]
                entry["segments"] = kept + [segment]
                self.tally(f"{scene} {item}", segment["verdict"])
            except (AdmissionError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        if errors:
            raise AdmissionError("; ".join(errors))
        # A speaker row is always a lip-sync take; its route follows from the
        # storyboard, not from any review, and the timeline needs the entry.
        for row in board:
            scene = str(row.get("still_id"))
            if isinstance(row, dict) and row.get("speaker") and scene not in entries:
                entries[scene] = {"pictureRoute": "grok-dialogue", "segments": []}
                ledger.setdefault("routesFromStoryboard", []).append(scene)
        ledger["admission"] = {
            "writer": "admit_review_decisions.py",
            "decisions": {"path": self.rel(self.decisions_path), "sha256": sha256_file(self.decisions_path)},
            "localAudit": {"path": self.rel(audit_path), "sha256": sha256_file(audit_path)},
            **({"previousLedger": {"path": self.rel(previous_path), "sha256": sha256_file(previous_path)}}
               if previous_path.is_file() else {}),
        }
        return ledger, {}

    # -- shared -----------------------------------------------------------
    def export(self, reviews: list[dict[str, Any]], binding_paths: list[Path]) -> dict[str, Any]:
        body = {"reviewType": self.lane, "sourceBindings": self.bindings(binding_paths), "reviews": reviews}
        return {"schemaVersion": 2, **body, "reviewFingerprint": canonical_sha256(body),
                "writer": "admit_review_decisions.py",
                "decisions": {"path": self.rel(self.decisions_path),
                              "sha256": sha256_file(self.decisions_path)}}

    def run(self, write: bool = True) -> dict[str, Any]:
        document, changes = getattr(self, self.lane)()
        review_dir = self.episode / "work/review"
        target = _next_version(review_dir, "h3-review-ledger" if self.lane == "h3" else f"{self.lane}-review")
        config = dict(self.config)
        config.update(changes)
        if self.lane == "h3":
            config["h3ReviewLedger"] = self.rel(target)
        result = {"lane": self.lane, "record": str(target), "written": write, **self.report}
        if not write:
            return result
        raw = _json_bytes(document)
        atomic_new_bytes(target, raw)
        sha = sha256_file(target)
        evidence = {"reviewExport": self.rel(target), "reviewExportSha256": sha,
                    "reviewFingerprint": document.get("reviewFingerprint")}
        if self.lane == "dialogue":
            for row in self.rows:
                config["dialogueOutputs"][str(row.get("sceneId") or row["itemId"])].update(evidence)
        elif self.lane == "score":
            for placement in config["scorePlacements"]:
                if placement.pop("_new", False):
                    placement.update(evidence)
        if self.config_path.is_file():
            archive = _next_version(self.config_path.parent, self.config_path.stem)
            atomic_new_bytes(archive, self.config_path.read_bytes())
            result["previousTimelineInputs"] = str(archive)
        temporary = self.config_path.with_name(f".{self.config_path.name}.partial")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(_json_bytes(config))
        temporary.replace(self.config_path)
        result.update(recordSha256=sha, timelineInputs=str(self.config_path))
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--decisions", type=Path, required=True, help="the plain decisions file")
    parser.add_argument("--input", type=Path,
                        help="timeline inputs to update (default PROJECT/work/assembly-v1/timeline-inputs.json)")
    parser.add_argument("--audit", type=Path, help="h3 lane: review_finite.py's local-audit.json")
    parser.add_argument("--check", action="store_true", help="validate and report; write nothing")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    config = args.input or project / "work/assembly-v1/timeline-inputs.json"
    try:
        result = Admission(project, config, args.decisions, args.lane, args.audit).run(write=not args.check)
    except (AdmissionError, FileExistsError) as exc:
        raise SystemExit(f"refused: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
