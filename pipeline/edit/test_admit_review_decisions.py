from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

import test_build_episode_timeline as base

timeline = base.timeline
import admit_review_decisions as admit

AT = "2026-01-02T03:04:05Z"


class AdmitReviewDecisionsTest(unittest.TestCase):
    setUp = base.TimelineBuilderTest.setUp
    tearDown = base.TimelineBuilderTest.tearDown
    row = staticmethod(base.TimelineBuilderTest.row)
    write_json = staticmethod(base.TimelineBuilderTest.write_json)
    sha = staticmethod(base.TimelineBuilderTest.sha)
    canonical_sha = staticmethod(base.TimelineBuilderTest.canonical_sha)
    refresh_provenance = base.TimelineBuilderTest.refresh_provenance
    fake_probe = base.TimelineBuilderTest.fake_probe

    def read(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def unadmit_dialogue(self):
        """The fixture's take as a runner leaves it: measured, not yet reviewed."""
        config = self.read(self.config)
        for key in ("reviewExport", "reviewExportSha256", "reviewFingerprint", "verdict", "speechWindow"):
            config["dialogueOutputs"]["demo-s003"].pop(key)
        (self.ep / "work/review/dialogue-review.json").unlink()
        self.write_json(self.config, config)
        return config

    def decide(self, rows, name="decisions.json", **top):
        path = self.ep / "work/review" / name
        self.write_json(path, {"reviewedBy": "agent:demo-reviewer", **top, "decisions": rows})
        return path

    def admit(self, lane, decisions, **kwargs):
        return admit.Admission(self.ep, self.config, decisions, lane, **kwargs).run()

    def take_sha(self):
        return self.sha(self.ep / "work/grok/demo-s003.mp4")

    def test_dialogue_decision_round_trips_into_the_timeline(self):
        before = self.unadmit_dialogue()
        old_bytes = self.config.read_bytes()
        decisions = self.decide([{"itemId": "demo-s003", "decision": "approved", "reviewedAt": AT,
                                  "sha256": self.take_sha(), "method": "full playback with sound",
                                  "speechWindow": {"start": 1.0, "end": 3.5}, "note": "keeps the pause"}])
        result = self.admit("dialogue", decisions)
        self.assertEqual(result["admitted"], ["demo-s003"])
        export_path = self.ep / "work/review/dialogue-review-v1.json"
        export = self.read(export_path)
        self.assertEqual(export["schemaVersion"], 2)
        self.assertEqual(export["reviewFingerprint"], self.canonical_sha(
            {k: export[k] for k in ("reviewType", "sourceBindings", "reviews")}))
        review = export["reviews"][0]
        self.assertEqual((review["reviewedBy"], review["reviewer"], review["note"]),
                         ("agent:demo-reviewer", {"kind": "agent", "name": "demo-reviewer"}, "keeps the pause"))
        # The previous inputs stay beside the new ones, byte for byte.
        self.assertEqual((self.ep / "work/assembly-v1/timeline-inputs-v1.json").read_bytes(), old_bytes)
        after = self.read(self.config)
        self.assertEqual({k: v for k, v in after.items() if k != "dialogueOutputs"},
                         {k: v for k, v in before.items() if k != "dialogueOutputs"})
        self.assertEqual(after["dialogueOutputs"]["demo-s003"]["speechWindow"], {"start": 1.0, "end": 3.5})

        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready["errors"])
        event = next(e for e in plan["events"] if e["kind"] == "dialogue")
        self.assertEqual(event["admission"]["reviewExport"], "work/review/dialogue-review-v1.json")
        self.assertEqual(event["subtitleAudioAt"], 0.708333)  # the decided speech window was read
        summary = ready["reviewSummary"]["dialogue"]
        self.assertEqual(summary["line"], "dialogue: 1 admitted, 1 agent-reviewed, 0 reviewed by a person")
        self.assertEqual(summary["methods"], {"full playback with sound": 1})

    def test_a_take_changed_after_its_decision_is_inadmissible(self):
        self.unadmit_dialogue()
        self.admit("dialogue", self.decide([{"itemId": "demo-s003", "decision": "approved", "reviewedAt": AT}]))
        (self.ep / "work/grok/demo-s003.mp4").write_bytes(b"another take")
        ready, plan = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertIsNone(plan)
        self.assertTrue(any(e["type"] == "dialogue-admission" for e in ready["errors"]))

    def test_a_stale_bound_document_is_inadmissible(self):
        self.unadmit_dialogue()
        self.admit("dialogue", self.decide([{"itemId": "demo-s003", "decision": "approved", "reviewedAt": AT}]))
        manifest = self.read(self.dialogue_manifest)
        manifest["note"] = "edited after review"
        self.write_json(self.dialogue_manifest, manifest)
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertFalse(ready["ready"])

    def test_refusals_write_nothing(self):
        self.unadmit_dialogue()
        good = {"itemId": "demo-s003", "decision": "approved", "reviewedAt": AT}
        cases = {
            "reviewedBy": ([good], {"reviewedBy": ""}),
            "never read as one": ([{"itemId": "demo-s003", "reviewedAt": AT, "note": "looks fine"}], {}),
            "UTC": ([dict(good, reviewedAt="2026-01-02T03:04:05+05:30")], {}),
            "is not the current file": ([dict(good, sha256="0" * 64)], {}),
            "decided twice": ([good, good], {}),
            "reviewer must be": ([dict(good, reviewer={"kind": "intern", "name": "x"})], {}),
            "no job": ([dict(good, itemId="demo-s099")], {}),
        }
        config_bytes = self.config.read_bytes()
        for message, (rows, top) in cases.items():
            with self.subTest(message), self.assertRaisesRegex(admit.AdmissionError, message):
                self.admit("dialogue", self.decide(rows, **top))
        self.assertEqual(self.config.read_bytes(), config_bytes)
        self.assertEqual(sorted(p.name for p in (self.ep / "work/review").iterdir()), ["decisions.json"])

    def test_a_take_whose_bytes_left_its_sidecar_is_refused(self):
        self.unadmit_dialogue()
        (self.ep / "work/grok/demo-s003.mp4").write_bytes(b"swapped before review")
        with self.assertRaisesRegex(admit.AdmissionError, "changed since its sidecar"):
            self.admit("dialogue", self.decide([{"itemId": "demo-s003", "decision": "approved", "reviewedAt": AT}]))

    def test_a_rejection_is_recorded_and_admits_nothing(self):
        self.unadmit_dialogue()
        result = self.admit("dialogue", self.decide(
            [{"itemId": "demo-s003", "decision": "rejected", "reviewedAt": AT}]))
        self.assertEqual(result["notAdmitted"], ["demo-s003"])
        ready, _ = timeline.build_timeline(self.ep, self.config, dry_run=True, probe=self.fake_probe)
        self.assertIn({"type": "approved-dialogue-verdict", "sceneId": "demo-s003", "actual": "rejected"},
                      ready["missing"])

    def test_rejecting_another_take_keeps_the_admitted_one(self):
        before = self.read(self.config)["dialogueOutputs"]["demo-s003"]
        correction_dir = self.ep / "work/grok-dialogue-corrections-v1"
        correction_dir.mkdir()
        manifest = correction_dir / "execution-manifest.json"
        doc = self.read(self.ep / "work/grok-dialogue-v1/execution-manifest.json")
        doc["jobs"][0]["outputStem"] = "work/grok-dialogue-corrections-v1/demo-s003"
        self.write_json(manifest, doc)
        take = correction_dir / "demo-s003.mp4"
        take.write_bytes(b"a correction take")
        self.write_json(correction_dir / "demo-s003.vid.json", {
            "sha256": self.sha(take), "outputSha256": self.sha(take),
            "executionManifestSha256": self.sha(manifest)})
        result = self.admit("dialogue", self.decide([{
            "itemId": "demo-s003", "decision": "rejected", "reviewedAt": AT,
            "executionManifest": "work/grok-dialogue-corrections-v1/execution-manifest.json"}]))
        self.assertEqual((result["notAdmitted"], result["currentKept"]), (["demo-s003"], ["demo-s003"]))
        self.assertEqual(self.read(self.config)["dialogueOutputs"]["demo-s003"], before)
        export = self.read(self.ep / "work/review/dialogue-review-v1.json")
        self.assertEqual(export["reviews"][0]["source"]["path"], "work/grok-dialogue-corrections-v1/demo-s003.mp4")
        ready, _ = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready["errors"])

    def test_check_mode_writes_nothing(self):
        self.unadmit_dialogue()
        before = self.config.read_bytes()
        decisions = self.decide([{"itemId": "demo-s003", "decision": "approved", "reviewedAt": AT}])
        result = admit.Admission(self.ep, self.config, decisions, "dialogue").run(write=False)
        self.assertFalse(result["written"])
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse((self.ep / "work/review/dialogue-review-v1.json").exists())

    def make_score(self):
        score_root = self.ep / "work/music-score-v1"
        sources = score_root / "sources"
        sources.mkdir(parents=True)
        draft = score_root / "creative-draft.json"
        self.write_json(draft, {"fixture": True})
        payload = {"duration": 10, "style": "offline"}
        manifest = score_root / "execution-manifest.json"
        self.write_json(manifest, {"sourceDraftSha256": self.sha(draft),
                                   "cues": [{"id": "m1-test", "payload": payload}]})
        track = sources / "m1-test-take-1.mp3"
        track.write_bytes(b"score-audio")
        sidecar = sources / "m1-test-take-1.source.json"
        self.write_json(sidecar, {
            "cueId": "m1-test", "candidateIndex": 1, "executionManifestSha256": self.sha(manifest),
            "sourceDraftSha256": self.sha(draft), "requestSha256": self.canonical_sha(payload),
            "sha256": self.sha(track)})
        self.write_json(manifest.with_suffix(".state.json"), {
            "executionManifestSha256": self.sha(manifest),
            "items": {"m1-test": {"status": "DOWNLOADED", "candidates": [{
                "index": 1, "status": "COMPLETE", "file": str(track), "sha256": self.sha(track),
                "sidecar": str(sidecar), "sidecarSha256": self.sha(sidecar)}]}}})
        config = self.read(self.config)
        config["scoreManifest"] = str(manifest.relative_to(self.ep))
        self.write_json(self.config, config)
        return track

    def test_score_decision_admits_a_take_and_adds_its_placement(self):
        track = self.make_score()
        result = self.admit("score", self.decide([{
            "itemId": "m1-test", "take": 1, "decision": "approved", "reviewedAt": AT,
            "sha256": self.sha(track), "reviewedBy": "a named listener",
            "reviewer": {"kind": "human", "name": "a named listener"},
            "timelineStart": 0.0, "timelineDuration": 10.0}], name="score-decisions.json"))
        self.assertEqual(result["admitted"], ["m1-test take 1"])
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready["errors"])
        placement = plan["scorePlacements"][0]
        self.assertEqual(placement["reviewExport"], "work/review/score-review-v1.json")
        self.assertEqual(placement["reviewer"], {"kind": "human", "name": "a named listener"})
        self.assertEqual(ready["reviewSummary"]["score"]["line"],
                         "score: 1 admitted, 0 agent-reviewed, 1 reviewed by a person")

    def test_score_decision_without_a_placement_is_reported_unplaced(self):
        self.make_score()
        result = self.admit("score", self.decide([{
            "itemId": "m1-test", "take": 1, "decision": "approved", "reviewedAt": AT}]))
        self.assertEqual(result["unplaced"], ["m1-test take 1"])
        self.assertEqual(self.read(self.config).get("scorePlacements", []), [])

    def test_a_rejected_score_take_adds_no_placement_and_leaves_the_cut(self):
        self.make_score()
        placed = {"itemId": "m1-test", "take": 1, "reviewedAt": AT,
                  "timelineStart": 0.0, "timelineDuration": 10.0}
        result = self.admit("score", self.decide([dict(placed, decision="rejected")]))
        self.assertEqual(result["notAdmitted"], ["m1-test take 1"])
        self.assertEqual(self.read(self.config)["scorePlacements"], [])
        self.admit("score", self.decide([dict(placed, decision="approved")], name="approve.json"))
        self.assertEqual([p["id"] for p in self.read(self.config)["scorePlacements"]], ["m1-test-take-1"])
        result = self.admit("score", self.decide([dict(placed, decision="rejected")], name="reject.json"))
        self.assertEqual(result["removedPlacements"], ["m1-test-take-1"])
        self.assertEqual(self.read(self.config)["scorePlacements"], [])
        archived = self.read(self.ep / "work/assembly-v1/timeline-inputs-v3.json")
        self.assertEqual(len(archived["scorePlacements"]), 1)
        ready, _ = timeline.build_timeline(self.ep, self.config, probe=self.fake_probe)
        self.assertTrue(ready["ready"], ready["errors"])

    def make_h3_audit(self, scenes, receipt=True):
        """A local audit as review_finite.py leaves it after a grouped ingest."""
        root = self.ep / "work/h3-run"
        (root / "silent").mkdir(parents=True)
        rows, items = [], []
        for scene in scenes:
            silent = root / "silent" / f"{scene}-job.mp4"
            silent.write_bytes(f"silent {scene}".encode())
            items.append({"id": f"{scene}-job", "silent": silent.name, "silentSha256": self.sha(silent)})
        receipt_path = root / "silent/ingest-receipt.json"
        self.write_json(receipt_path, {"items": items})
        for scene in scenes:
            silent = root / "silent" / f"{scene}-job.mp4"
            rows.append({"id": f"{scene}-job", "status": "technical-pass", "silent": {
                "file": f"/worker/elsewhere/silent/{silent.name}", "sha256": self.sha(silent), "frames": 72,
                **({"ingestReceiptSha256": self.sha(receipt_path)} if receipt else {})}})
        audit = root / "local-audit.json"
        self.write_json(audit, {"outputs": rows})
        return audit

    def probe_with_silent(self, path: Path):
        if path.parent.name == "silent":
            return timeline.MediaProbe(3.0, 3.0, None, 72, 24.0, 1, 0)
        return self.fake_probe(path)

    def test_h3_decisions_write_a_new_version_two_ledger_with_origins(self):
        (self.ep / "work/h3-review-ledger-v1.json").unlink()
        scenes = ("demo-s001", "demo-s002", "demo-s004")
        audit = self.make_h3_audit(scenes)
        rows = [{"itemId": f"{scene}-job", "sceneId": scene, "decision": "approved", "reviewedAt": AT,
                 "method": "frame sheet"} for scene in scenes]
        self.admit("h3", self.decide(rows), audit_path=audit)
        ledger_path = self.ep / "work/review/h3-review-ledger-v1.json"
        ledger = self.read(ledger_path)
        self.assertEqual(ledger["schemaVersion"], 2)
        self.assertEqual(ledger["entries"]["demo-s003"], {"pictureRoute": "grok-dialogue", "segments": []})
        self.assertEqual(ledger["routesFromStoryboard"], ["demo-s003"])
        segment = ledger["entries"]["demo-s002"]["segments"][0]
        self.assertEqual(segment["origin"]["kind"], "h3-ingest")
        self.assertEqual(self.read(self.config)["h3ReviewLedger"], "work/review/h3-review-ledger-v1.json")
        ready, plan = timeline.build_timeline(self.ep, self.config, probe=self.probe_with_silent)
        self.assertTrue(ready["ready"], ready["errors"])
        self.assertEqual(ready["reviewSummary"]["h3"]["admitted"], 3)
        self.assertEqual(ready["reviewSummary"]["h3"]["methods"], {"frame sheet": 3})

    def test_h3_re_review_of_the_same_bytes_keeps_the_authored_trim(self):
        (self.ep / "work/h3-review-ledger-v1.json").unlink()
        audit = self.make_h3_audit(("demo-s001", "demo-s002", "demo-s004"))
        rows = [{"itemId": f"{scene}-job", "sceneId": scene, "decision": "approved", "reviewedAt": AT}
                for scene in ("demo-s001", "demo-s002", "demo-s004")]
        rows[1].update({"in": 0.5, "out": 2.5, "speed": 1.0, "role": "demo-role"})
        self.admit("h3", self.decide(rows), audit_path=audit)
        self.admit("h3", self.decide([dict(rows[1], method="full playback", **{
            k: None for k in ("in", "out", "speed", "role")})], name="again.json"), audit_path=audit)
        segments = self.read(self.ep / "work/review/h3-review-ledger-v2.json")["entries"]["demo-s002"]["segments"]
        self.assertEqual(len(segments), 1)
        self.assertEqual({k: segments[0][k] for k in ("in", "out", "speed", "role", "method")},
                         {"in": 0.5, "out": 2.5, "speed": 1.0, "role": "demo-role", "method": "full playback"})
        ready, _ = timeline.build_timeline(self.ep, self.config, probe=self.probe_with_silent)
        self.assertTrue(ready["ready"], ready["errors"])

    def test_h3_refuses_to_guess_the_row_or_drop_an_origin(self):
        (self.ep / "work/h3-review-ledger-v1.json").unlink()
        audit = self.make_h3_audit(("demo-s002",), receipt=False)
        row = {"itemId": "demo-s002-job", "decision": "approved", "reviewedAt": AT}
        with self.assertRaisesRegex(admit.AdmissionError, "name the storyboard row"):
            self.admit("h3", self.decide([row]), audit_path=audit)
        with self.assertRaisesRegex(admit.AdmissionError, "only with its origin"):
            self.admit("h3", self.decide([dict(row, sceneId="demo-s002")]), audit_path=audit)

    def test_h3_decision_joins_an_existing_version_one_ledger(self):
        audit = self.make_h3_audit(("demo-s002",), receipt=False)
        old = self.read(self.ep / "work/h3-review-ledger-v1.json")
        self.admit("h3", self.decide([{"itemId": "demo-s002-job", "sceneId": "demo-s002", "role": "second-look",
                                       "decision": "approved", "reviewedAt": AT}]), audit_path=audit)
        ledger = self.read(self.ep / "work/review/h3-review-ledger-v1.json")
        self.assertEqual(ledger["schemaVersion"], 1)
        segments = ledger["entries"]["demo-s002"]["segments"]
        self.assertEqual(segments[0], old["entries"]["demo-s002"]["segments"][0])
        self.assertEqual(segments[1]["role"], "second-look")
        ready, _ = timeline.build_timeline(self.ep, self.config, probe=self.probe_with_silent)
        self.assertTrue(ready["ready"], ready["errors"])

    def test_help_runs_without_a_project(self):
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stdout(io.StringIO()):
            admit.main(["--help"])
        self.assertEqual(stop.exception.code, 0)


class ScoreBenchTemplateTest(unittest.TestCase):
    def test_bench_template_names_the_decisions_fields(self):
        # The shape itself is proved by the score admission tests above.
        page = (Path(__file__).resolve().parents[1] / "score_candidate_review_template.html").read_text()
        for field in ("lane:'score'", "reviewedBy", "itemId", "take:", "decision", "reviewedAt"):
            self.assertIn(field, page)


if __name__ == "__main__":
    unittest.main()
