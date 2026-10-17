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
| Assemble a measured timeline | `edit/build_episode_timeline.py`, `edit/assemble_longform.py` |
| Design and revise sound | `edit/render_designed_sfx.py`, `edit/revise_master_sound.py` |
| Prepare subtitles | `edit/prepare_subtitle_manuscript.py`, `edit/export_bilingual_subtitles.py` |
| Reuse HD picture or change its audio | `open_media/build_hd_range_plan.py`, `open_media/hd_audio_remux.py` |
| Package selected deliverables | `edit/package_hindi_delivery.py`, `edit/package_hindi_longform.py` |

These commands expose specific input contracts through `--help`; they are not
creative instructions. Use an external project directory. Source stills can come
from any hand: an image model, the filmmaker's own painting or scan, or a
third party's work. The registrar records which, and does not grant account
access or provide an image generator.

One established route uses H3 picture, Grok native Hindi dialogue, ElevenLabs
narration, Suno/Kie score, and self-hosted effects. The current packaging/subtitle
preset supports Hindi performance and Hindi/English subtitles. That describes
what has been implemented and exercised, not what every film should be.

Research, source grounding, animatics, auditions, and reviews are useful when
they resolve a real uncertainty. They do not need to become rituals. Make an
initial mix to hear it; refine what matters. Preserve useful source-bound
observations so unchanged work does not need to be reviewed repeatedly.

Check current provider availability and credits before a paid run. Save concrete
job state so interruption does not create duplicate work. Delivery and publication
follow the user's actual request; neither a model response nor a passing test
supplies creative approval.

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
and alternate IDs such as `demo-s010b` and `demo-s010-alt`. A still without a
number, such as a plate or a canon sheet, skips range facts, and its sidecar
records each skip. `from`/`to` resolve by the order of an explicit
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
its hash and header dimensions. The grounding guard runs the same way for every
origin.

`--retake` moves a registered still and its sidecar to
`art/retired/<still_id>/<short-sha>/` and deletes nothing. It returns the row to
`needs_changes` with a history entry and prints the assets downstream of it.
Their hash bindings then fail until each one is remade or re-pointed.
`--batch FILE` applies the same refusals to a list of registrations and stops at
the first refusal. `make_reference_pack.py` composes registered images into one
labelled contact image and writes the `.refpack.json` that the registrar accepts
as packed coverage. Labels come from the policy's identity authorities.

`derive_still.py` writes a still derived from a registered image or an accepted
clip into a manifest row. It records the parent path and hash, the operation,
the tool hash and the output hash in a `.derive.json` sidecar. `exact-frame`
decodes one frame by index and runs only on the qualified worker. `crop`, `pad`
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
