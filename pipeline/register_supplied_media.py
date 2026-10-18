#!/usr/bin/env python3
"""Register media from any hand: a recorded voice take, a composed cue, picture.

Writes ``<file>.supplied.json`` beside the file: its SHA-256, measured probe
facts, who made or supplied it, the rights it is used under, and whether it may
be sent to a provider. It never overwrites a sidecar, and it never fills a
field the operator did not give.

The sidecar is a declaration. The maker and rights fields turn a false claim
into a recorded false statement; they do not prove anything. A file whose
sidecar says ``providerUpload: forbidden`` is refused by the shared uploader,
which guards against accidental upload and is not DRM. Admission into a cut
still goes through the timeline's own stream checks and a source-bound review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

if __package__:
    from ._common import atomic_new_bytes, is_below, project_root, sha256_file
else:
    from _common import atomic_new_bytes, is_below, project_root, sha256_file

SCHEMA_VERSION = 1
KIND = "supplied_media"
SUFFIX = ".supplied.json"
AUDIO_TYPES = {"voice-take", "music-cue", "sfx"}
VIDEO_TYPES = {"picture", "reference-clip"}
MEDIA_TYPES = AUDIO_TYPES | VIDEO_TYPES
PROVIDER_UPLOAD = ("allowed", "forbidden")


class SuppliedMediaError(ValueError):
    """The file or its declaration cannot be registered or trusted."""


def sidecar_for(path: Path) -> Path:
    return path.with_name(path.name + SUFFIX)


def probe(path: Path) -> dict[str, Any]:
    try:
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, check=True, timeout=120).stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SuppliedMediaError(f"ffprobe cannot read {path}: {exc}") from exc
    streams = info.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    facts: dict[str, Any] = {
        "durationSeconds": float((info.get("format") or {}).get("duration") or 0.0),
        "videoStreams": len(video), "audioStreams": len(audio),
    }
    if video:
        facts.update(videoCodec=video[0].get("codec_name"), width=video[0].get("width"),
                     height=video[0].get("height"), fps=video[0].get("r_frame_rate"))
    if audio:
        facts.update(audioCodec=audio[0].get("codec_name"),
                     sampleRate=int(audio[0].get("sample_rate") or 0),
                     channels=audio[0].get("channels"))
    return facts


def _bound(project: Path, value: Path, label: str) -> dict[str, str]:
    path = value.resolve()
    if not is_below(path, project) or not path.is_file():
        raise SuppliedMediaError(f"{label} must be an existing file inside the project: {path}")
    return {"path": path.relative_to(project).as_posix(), "sha256": sha256_file(path)}


def build_record(project: Path, media: Path, *, media_type: str, maker: str | None,
                 supplied_by: str | None, rights_basis: str, provider_upload: str,
                 performer: str | None = None, composer: str | None = None,
                 rights_evidence: Path | None = None, digital_source_type: str | None = None,
                 language: str | None = None, performed_text: str | None = None,
                 derived_from: Path | None = None, note: str | None = None,
                 probe_fn=probe) -> dict[str, Any]:
    project = project.resolve()
    media = media.resolve()
    if media_type not in MEDIA_TYPES:
        raise SuppliedMediaError(f"media type must be one of {sorted(MEDIA_TYPES)}")
    if not is_below(media, project) or not media.is_file():
        raise SuppliedMediaError(f"media must be an existing file inside the project: {media}")
    if not (maker or supplied_by):
        raise SuppliedMediaError("name the maker or who supplied it (--maker or --supplied-by)")
    if not rights_basis or not rights_basis.strip():
        raise SuppliedMediaError("state the rights basis this file is used under (--rights-basis)")
    if provider_upload not in PROVIDER_UPLOAD:
        raise SuppliedMediaError(f"--provider-upload must be one of {list(PROVIDER_UPLOAD)}")
    facts = probe_fn(media)
    if media_type in AUDIO_TYPES and not facts["audioStreams"]:
        raise SuppliedMediaError(f"a {media_type} needs an audio stream: {media}")
    if media_type in VIDEO_TYPES and not facts["videoStreams"]:
        raise SuppliedMediaError(f"{media_type} needs a video stream: {media}")
    if media_type == "voice-take" and (not language or not performed_text or not performed_text.strip()):
        raise SuppliedMediaError("a voice take records its language and the exact text performed")
    rights: dict[str, Any] = {"basis": rights_basis.strip(), "providerUpload": provider_upload}
    if rights_evidence is not None:
        rights["evidence"] = _bound(project, rights_evidence, "rights evidence")
    record: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION, "kind": KIND, "mediaType": media_type,
        "file": media.relative_to(project).as_posix(), "sha256": sha256_file(media),
        "bytes": media.stat().st_size, "probe": facts, "rights": rights,
        "registeredAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "declaration": ("Recorded as stated by the operator; maker and rights are a "
                        "declaration, not verified evidence"),
    }
    optional = {"maker": maker, "suppliedBy": supplied_by, "performer": performer,
                "composer": composer, "digitalSourceType": digital_source_type,
                "language": language, "performedText": performed_text, "note": note}
    record.update({key: value for key, value in optional.items() if value})
    if derived_from is not None:
        record["derivedFrom"] = _bound(project, derived_from, "derivedFrom")
    return record


def register(project: Path, media: Path, **fields) -> Path:
    record = build_record(project, media, **fields)
    sidecar = sidecar_for(media.resolve())
    raw = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        atomic_new_bytes(sidecar, raw)
    except FileExistsError as exc:
        raise SuppliedMediaError(f"refusing to overwrite an existing registration: {sidecar}") from exc
    return sidecar


def validate_supplied(project: Path, media: Path, sidecar_sha256: str | None = None,
                      media_type: str | None = None) -> dict[str, Any]:
    """The registration for ``media``, refused if it no longer describes these bytes."""
    project, media = project.resolve(), media.resolve()
    sidecar = sidecar_for(media)
    if not sidecar.is_file():
        raise SuppliedMediaError(f"no supplied-media registration beside {media}")
    if sidecar_sha256 is not None and sha256_file(sidecar) != sidecar_sha256:
        raise SuppliedMediaError(f"supplied-media registration changed: {sidecar}")
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuppliedMediaError(f"cannot read {sidecar}: {exc}") from exc
    if not isinstance(record, dict) or record.get("kind") != KIND \
            or record.get("schemaVersion") != SCHEMA_VERSION:
        raise SuppliedMediaError(f"not a version {SCHEMA_VERSION} supplied-media sidecar: {sidecar}")
    if record.get("file") != media.relative_to(project).as_posix():
        raise SuppliedMediaError(f"{sidecar} names another file: {record.get('file')}")
    if record.get("sha256") != sha256_file(media):
        raise SuppliedMediaError(f"{media} changed after it was registered")
    if media_type is not None and record.get("mediaType") != media_type:
        raise SuppliedMediaError(f"{media} is registered as {record.get('mediaType')}, not {media_type}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("media", type=Path)
    parser.add_argument("--project", type=Path, help="project root (or AGENTIC_PROJECT_ROOT)")
    parser.add_argument("--type", dest="media_type", required=True, choices=sorted(MEDIA_TYPES))
    parser.add_argument("--maker", help="who made it")
    parser.add_argument("--supplied-by", help="who supplied it, when the maker is someone else")
    parser.add_argument("--performer")
    parser.add_argument("--composer")
    parser.add_argument("--rights-basis", required=True,
                        help="the basis this file is used under, in the operator's words")
    parser.add_argument("--provider-upload", required=True, choices=PROVIDER_UPLOAD,
                        help="whether this file may be sent to a generation provider")
    parser.add_argument("--rights-evidence", type=Path, help="optional evidence file to bind by hash")
    parser.add_argument("--digital-source-type", help="optional IPTC digitalSourceType URI")
    parser.add_argument("--language", help="language of a voice take")
    parser.add_argument("--performed-text-file", type=Path,
                        help="UTF-8 file holding the exact text a voice take performs")
    parser.add_argument("--derived-from", type=Path, help="the project file this was made from")
    parser.add_argument("--note")
    args = parser.parse_args(argv)
    try:
        text = (args.performed_text_file.read_text(encoding="utf-8").strip()
                if args.performed_text_file else None)
        sidecar = register(
            project_root(args.project), args.media, media_type=args.media_type,
            maker=args.maker, supplied_by=args.supplied_by, rights_basis=args.rights_basis,
            provider_upload=args.provider_upload, performer=args.performer,
            composer=args.composer, rights_evidence=args.rights_evidence,
            digital_source_type=args.digital_source_type, language=args.language,
            performed_text=text, derived_from=args.derived_from, note=args.note)
    except (SuppliedMediaError, OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"registered": str(sidecar)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
