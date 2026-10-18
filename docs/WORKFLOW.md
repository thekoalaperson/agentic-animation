# A toolbox for making films

Start with the user's intent, available material, and the decisions that matter
now. There is no required sequence of documents, approvals, shot types, or model
calls. A small scene may need only references, one prompt, and an edit. A long
film benefits from shared canon, measured timing, and independently resumable
sequences. Choose the amount of process that helps the work.

| Need | Useful tools |
|---|---|
| Register, derive and trace stills | `register_builtin_imagegen.py`, `derive_still.py`, `make_reference_pack.py`, `asset_registry.py` |
| Generate hosted motion/dialogue | `kie_video.py`, `kie_dialogue_finite.py` |
| Generate narration or score | `eleven_batch_finite.py`, `kie_music_finite.py` |
| Run H3 jobs and ingest picture | `h3/stage_finite.py`, `h3/box/finite_client.py`, `h3/ingest_finite.py` |
| Compile paired text or inventory a project | `edit/compile_production_script.py`, `edit/prepare_episode_production.py` |
| Look and listen before paying, on any host with ffmpeg | `edit/render_timed_animatic.py --preview`, `edit/prepare_episode_production.py --sheets --preview`, `prepare_dialogue_review.py --preview`, `h3/review_finite.py --preview-sheets`, `--audition` on `music_candidate_ranker.py`, `edit/render_working_music.py` and `edit/build_working_audio.py` |
| Assemble a measured timeline | `edit/build_episode_timeline.py`, `edit/assemble_longform.py` |
| Design and revise sound | `edit/render_designed_sfx.py`, `edit/revise_master_sound.py` |
| Prepare subtitles | `edit/prepare_subtitle_manuscript.py`, `edit/export_bilingual_subtitles.py` |
| Reuse HD picture or change its audio | `open_media/build_hd_range_plan.py`, `open_media/hd_audio_remux.py` |
| Package selected deliverables | `edit/package_hindi_delivery.py`, `edit/package_hindi_longform.py` |
| Draw an optional push from outside the model | `draw_provocations.py` (see [creative notes](CREATIVE.md)) |

These commands expose specific input contracts through `--help`; they are not
creative instructions. Use an external project directory. Source stills can come
from any hand: an image model, the filmmaker's own painting or scan, or a
third party's work. The registrar records which, and does not grant account
access or provide an image generator.

One worked example uses H3 picture, Grok native Hindi dialogue, ElevenLabs
narration, Suno/Kie score, and self-hosted effects. The current packaging/subtitle
preset supports Hindi performance and Hindi/English subtitles. That describes
what has been implemented and exercised, not what every film should be.
[Video models](VIDEO-MODELS.md) separates what each motion runner sends and
enforces from what its provider publishes.

Research, source grounding, animatics, auditions, and reviews are useful when
they resolve a real uncertainty. They do not need to become rituals. Make an
initial mix to hear it; refine what matters. Preserve useful source-bound
observations so unchanged work does not need to be reviewed repeatedly.

Previews and auditions run on the coordinator and are labelled
`-preview-not-final` or `-audition-not-final`; nothing downstream admits them.
The timed animatic can be written from the storyboard (`--from-storyboard`,
using measured narration or an explicit `estimatedSeconds`), can hold a
wordless beat for authored frames, and can carry one temporary score stem. It
can therefore preview a cut the measured timeline cannot assemble yet: today
the timeline clocks every row by its speech.

Check current provider availability and credits before a paid run. Save concrete
job state so interruption does not create duplicate work. Delivery and publication
follow the user's actual request; neither a model response nor a passing test
supplies creative approval.

## Looking at the work

Each kind of look costs something different and catches something different.
They come in no fixed order, and any of them may be skipped.

- A cold read of the plan, against its own direction and against the provider
  facts, costs minutes. It catches a plan that contradicts itself, a request no
  runner sends, and major choices that were inherited rather than decided.
- The stills sheet in story order costs nothing beyond the stills. It catches
  identity, state and continuity errors while they are cheapest to fix.
- The timed animatic costs a coordinator render. It catches pacing, held
  silence, late or early music and a reordered memory before any motion spend.
- A pilot of the riskiest shots costs a few paid clips. It catches route
  behaviour that no page states.
- Watching the main batch's takes costs the batch. It catches the performances
  and the drift that a pilot is too small to show.
- One continuous watch of the working cut, with sound, costs its runtime. It
  catches what only accumulates: rhythm, repetition and fatigue.

Review each take against this film's own direction and the technical facts, not
against any look these docs mention.

## Files and keys

Recurring files, by the role they play:

- `story/storyboard_plan.json`: the rows, in story order. The timeline binds it
  as a whole file, so free-text notes go on `generation_manifest.json` rows.
- `story/generation_manifest.json` and a still policy: the stills, their
  references and continuity facts.
- `story/GOAL.md`: the dialogue runner's default authority file. Its hash is
  frozen into each dialogue manifest, so an edit means a new manifest.
- A frozen execution manifest per paid batch (for example
  `work/grok-dialogue-v1/execution-manifest.json`), its draft beside it, and a
  separate approval record bound to the manifest's bytes.
- `.pending.json`, `.vid.json` and other sidecars: persisted provider task IDs
  and the provenance of landed media.
- `work/h3-review-ledger-v1.json`: the accepted H3 segments and their verdicts.
- Review exports: the source-bound decisions that admit a dialogue clip or a
  score take to the timeline.
- `work/assembly-v1/timeline-inputs.json`: the timeline's configuration. Its
  outputs are `edit-plan.json`, the subtitle timings, `timeline-audit.json`
  and, from `--dry-run`, `timeline-readiness.json`.

No tool in this checkout writes the frozen dialogue manifest, the approval
record or the review exports; each runner's `--help` and validator define what
they must hold.

Keys `build_episode_timeline.py` reads from `timeline-inputs.json`, grouped by
role:

- Documents: `storyboard`, `narrationManifest`, `dialogueManifest`,
  `h3ReviewLedger`, `productionOverrides`, `subtitleManuscript`, `scoreManifest`,
  `sfxManifest`, `outputDirectory`, and `narrationStateEpisodeRoot` and
  `scoreStateEpisodeRoot` for state recorded under another root.
- Admission: `dialogueOutputs` (per scene, including its review and an optional
  `pictureCrop`, `pictureCropFrom` and `pictureCropReason`),
  `requireApprovedDialogue`, `requireApprovedScore`, `reviewScope`.
- Rhythm: `dialogueHandles` (`preSeconds`, `postSeconds`),
  `preserveCompleteDialogueClips`, `minimumPictureSegmentSeconds`,
  `microCutRoleWords`, `allowCausalBorrowing`, `narrationBreathing`,
  `narrationBreathOverrides`, and the ledger's per-segment in, out, speed and
  role.
- Sound and text: `scorePlacements`, `musicBreaths`, `dialogueSubtitleMode`.

What the tools enforce, as facts rather than a sequence to follow:

- A reference is registered before any still that cites it.
- A dialogue manifest and an H3 plan bind each still's hash, so a still
  retaken afterwards refuses them until they are remade.
- A paid runner submits only against a frozen manifest and its declared limits;
  a changed prompt or authority file refuses the manifest as well.
- The timeline admits a dialogue clip or a score take only through a review
  export bound to its source and sidecar hashes, and H3 picture only through an
  accepted ledger segment.
- The assembler, mixer and subtitle exporter read the timeline's outputs and
  refuse when what they bind has changed.
- Packaging binds its approvals to the delivered files' hashes.

## Stills and canon

A still is a row in the project's `story/generation_manifest.json` (a small
demo sits in `examples/project/story/`). The fields the tools read are
`still_id`; `status` (`generation_ready`, `failed` or `needs_changes` can be
registered); `target_output_path`, a `.png` inside the project; `prompt` or
`imagegen_prompt`; an optional `size`, the requested generation size; and
`refs.all` or `imagegen_refs`, the references the still was made from. A row
may also carry free-text `staging` notes (screen side, gaze, attitude, who is
reacting, and whether the line is kept or crossed and why). The stills sheet
draws the speaker and a one-line summary of them under each thumbnail, and
nothing checks them against any rule. They go on manifest rows, not on
storyboard rows, because the storyboard is hash-bound as a whole file.

Register a reference before any still that uses it. A registration that names
unregistered or missing references lists all of them in one refusal.

An explicit `--policy` file holds `grounding_guard.identity_authorities`
(`name`, `aliases`, `sources`) and `grounding_guard.continuity_facts` (`id`,
`value`, and optionally `still_ids`, `episode`, `line_range`, or `from`/`to`
still IDs). A range reads the number in a `PROJECT-sNUMBER` ID, including split
and alternate IDs such as `demo-s010b` and `demo-s010-alt`, so range and
`episode` facts apply to those IDs as well. A still without a number, such as a
plate or a canon sheet, skips range facts, and its sidecar records each skip. `from`/`to` resolve by the order of an explicit
`--storyboard`. `asset_registry.py --facts STILL_ID --policy ...` prints the
facts active at a still. `asset_registry.py --board` with explicit `--policy`,
`--storyboard` and `--manifest` writes a read-only HTML page under `work/`. For
each identity it shows the sheets, then every still that claims or references
that identity, in story order, with the facts active there.

`register_builtin_imagegen.py --origin` records who made the image:

- `imagegen` (the default) accepts only PNGs beneath the ImageGen output root
  and writes the historical `.imagegen.json` sidecar.
- `human-made`, `third-party` and `other-tool:NAME` write a `.still.json`
  sidecar of kind `supplied_still_registration`, naming that provider. They
  need `--maker` or `--supplied-by`, `--rights-basis`, and `--provider-upload
  allowed|forbidden`. A rights evidence file, a creation date and an IPTC
  `digitalSourceType` URI are optional. The sidecar also records the row's
  requested size and the hash of every reference.

The source must sit under the origin's root (`--source-root`, or the profile's
`stills.sourceRoots.<provider>`; ImageGen also reads `imagegen.outputRoot`). A
supplied still with no declared root must be inside the project. The registered
target is always a PNG. If the image started as a JPEG, WebP, TIFF or other
file, convert it and pass the unconverted file as `--original`, which records
its hash and header dimensions and whether they match the PNG's. An optional
`--conversion` note says how the step was made (the tool, its settings, any crop
or resample). The original follows the same rule as the source: under the
declared root, or else inside the project. The grounding guard runs the same
way for every origin.

`--retake` moves a registered still and its sidecar to
`art/retired/<still_id>/<short-sha>/` and deletes nothing. It returns the row to
`needs_changes` with a history entry and prints the assets downstream of it.
Reference packs and motion manifests bind its hash and refuse until remade.
Stills registered against it hold its path; supplied and derived sidecars also
record the hash it had, but nothing re-checks that, so the downstream list in
the history entry is the record of which stills to remake or re-point. A
retired still is refused as a reference, a pack source or a derive parent. A
retake also refuses while the registry is inconsistent, because its impact
cannot be read. `--batch FILE` applies the same refusals to a list of
registrations and stops at the first refusal; per-row options go on the rows,
and a per-row flag on the command line is refused rather than ignored. `make_reference_pack.py` composes registered images into one
labelled contact image and writes the `.refpack.json` that the registrar accepts
as packed coverage. Labels come from the policy's identity authorities.

`derive_still.py` writes a still derived from a registered image or an accepted
clip into a manifest row. The parent must be registered, unretired and
unchanged since its registration. The `.derive.json` sidecar records the parent
path and hash, the operation, the tool hash and the output hash, and carries the
parent's origin, provider, maker or supplier and rights, so a crop of a still
that may not be uploaded keeps saying so. The grounding guard runs on the row's
references plus the parent, with the registrar's `--policy`,
`--exact-identity`, `--grounding-fact` and `--no-visible-identity` flags.
`exact-frame` decodes one frame by index and runs only on the qualified worker. `crop`, `pad`
and `scale` run anywhere and need an explicit anchor or offset. Production
inventory reads a derived row the same way it reads a registered one.

Series canon lives in its own read-only root (`--canon-root`, or the profile's
`stills.canonRoot`). Episode rows reference it as `canon:<path>`, and the
sidecar records the SHA-256 of each canon reference.
`asset_registry.py --impact canon:<path> --projects P1 P2 ...` lists the
dependents of a canon file across episodes.

Geometry the code checks today: H3 output is 1344x768 or 768x1344; the Grok
dialogue runner sends 720p at 16:9, and after the clip is paid for it refuses
any landed clip that is not 16:9. The dialogue dry run and H3 staging warn,
without refusing, when a still's aspect is more than 0.03 away from the planned
output, and point to `derive_still.py`. Whether Grok output follows the still's
geometry has not been qualified. The timeline readiness report lists any still
that opens more than one dialogue job, and says whether those jobs are adjacent.
That is information for the edit, and a held composition is a legitimate
choice.
