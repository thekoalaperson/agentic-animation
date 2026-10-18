"""Supplied-media registration: declared provenance, rights and upload refusal."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from pipeline import _common
from pipeline import asset_registry
from pipeline import register_supplied_media as supplied


def tone(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=220:duration={seconds}", str(path)], check=True)
    return path


def voice_fields(**changes):
    fields = dict(media_type="voice-take", maker="demo-performer", supplied_by=None,
                  rights_basis="recorded for this demo project", provider_upload="forbidden",
                  language="hi", performed_text="एक पंक्ति।")
    fields.update(changes)
    return fields


class RegisterTest(unittest.TestCase):
    def test_a_voice_take_is_registered_once_with_its_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            take = tone(project / "audio/supplied/demo-s004.wav")
            raw = tone(project / "audio/supplied/demo-s004-raw.wav")
            sidecar = supplied.register(project, take, derived_from=raw, **voice_fields())
            self.assertEqual(sidecar.name, "demo-s004.wav.supplied.json")
            record = json.loads(sidecar.read_text())
            self.assertEqual(record["file"], "audio/supplied/demo-s004.wav")
            self.assertEqual(record["rights"], {"basis": "recorded for this demo project",
                                                "providerUpload": "forbidden"})
            self.assertEqual(record["probe"]["audioStreams"], 1)
            self.assertEqual(record["derivedFrom"]["path"], "audio/supplied/demo-s004-raw.wav")
            self.assertNotIn("suppliedBy", record)
            with self.assertRaisesRegex(supplied.SuppliedMediaError, "refusing to overwrite"):
                supplied.register(project, take, **voice_fields())
            self.assertEqual(supplied.validate_supplied(project, take, media_type="voice-take"), record)
            with self.assertRaisesRegex(supplied.SuppliedMediaError, "not picture"):
                supplied.validate_supplied(project, take, media_type="picture")
            with self.assertRaisesRegex(supplied.SuppliedMediaError, "registration changed"):
                supplied.validate_supplied(project, take, sidecar_sha256="0" * 64)
            with take.open("ab") as handle:
                handle.write(b"\0")
            with self.assertRaisesRegex(supplied.SuppliedMediaError, "changed after"):
                supplied.validate_supplied(project, take)
            registry = asset_registry.load_registry(project)
            self.assertEqual(registry["audio/supplied/demo-s004.wav"]["refs"],
                             ["audio/supplied/demo-s004-raw.wav"])

    def test_missing_declarations_are_refused_without_writing(self):
        cases = [
            (voice_fields(maker=None), "maker or who supplied"),
            (voice_fields(rights_basis=" "), "rights basis"),
            (voice_fields(provider_upload="maybe"), "--provider-upload"),
            (voice_fields(performed_text=None), "exact text performed"),
            (voice_fields(media_type="picture"), "needs a video stream"),
        ]
        for fields, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp).resolve()
                take = tone(project / "audio/take.wav")
                with self.assertRaisesRegex(supplied.SuppliedMediaError, message):
                    supplied.register(project, take, **fields)
                self.assertFalse(supplied.sidecar_for(take).exists())
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            outside = tone(Path(other) / "take.wav")
            with self.assertRaisesRegex(supplied.SuppliedMediaError, "inside the project"):
                supplied.register(Path(tmp), outside, **voice_fields())

    def test_command_line_registers_a_cue(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            cue = tone(project / "audio/demo-cue.wav")
            code = supplied.main([str(cue), "--project", str(project), "--type", "music-cue",
                                  "--composer", "demo-composer", "--supplied-by", "demo-studio",
                                  "--rights-basis", "commissioned", "--provider-upload", "allowed"])
            self.assertEqual(code, 0)
            record = json.loads(supplied.sidecar_for(cue).read_text())
            self.assertEqual((record["mediaType"], record["composer"]), ("music-cue", "demo-composer"))


class UploadRefusalTest(unittest.TestCase):
    def test_forbidden_upload_never_reaches_the_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            take = tone(project / "audio/take.wav")
            supplied.register(project, take, **voice_fields())
            with mock.patch.object(_common, "http_json") as post, \
                 mock.patch.object(_common, "env_key", return_value="offline"):
                with self.assertRaisesRegex(ValueError, "providerUpload is forbidden"):
                    _common.upload_ref(take, project / "work/upload-log.jsonl")
            post.assert_not_called()
            self.assertFalse((project / "work/upload-log.jsonl").exists())

    def test_allowed_or_unregistered_files_upload_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            take = tone(project / "audio/take.wav")
            plain = tone(project / "audio/plain.wav")
            supplied.register(project, take, **voice_fields(provider_upload="allowed"))
            reply = {"data": {"downloadUrl": "https://files.invalid/x.wav"}}
            with mock.patch.object(_common, "http_json", return_value=reply) as post, \
                 mock.patch.object(_common, "env_key", return_value="offline"):
                for path in (take, plain):
                    self.assertEqual(_common.upload_ref(path, project / "work/log.jsonl"),
                                     "https://files.invalid/x.wav")
            self.assertEqual(post.call_count, 2)

    def test_a_still_sidecar_that_forbids_upload_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            still = Path(tmp) / "demo-s001.png"
            still.write_bytes(b"demo still")
            still.with_suffix(".imagegen.json").write_text(json.dumps(
                {"rights": {"basis": "own painting", "providerUpload": "forbidden"}}))
            self.assertEqual(_common.provider_upload_forbidden(still),
                             still.with_suffix(".imagegen.json"))


if __name__ == "__main__":
    unittest.main()
