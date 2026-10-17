# A toolbox for making films

Start with the user's intent, available material, and the decisions that matter
now. There is no required sequence of documents, approvals, shot types, or model
calls. A small scene may need only references, one prompt, and an edit. A long
film benefits from shared canon, measured timing, and independently resumable
sequences. Choose the amount of process that helps the work.

| Need | Useful tools |
|---|---|
| Register stills and trace dependencies | `register_builtin_imagegen.py`, `asset_registry.py` |
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
from the user's chosen image tool; the ImageGen registrar supports that tool's
provenance but does not grant account access or provide an image generator.

One established route uses H3 picture, Grok native Hindi dialogue, ElevenLabs
narration, Suno/Kie score, and self-hosted effects. The current packaging/subtitle
preset supports Hindi performance and Hindi/English subtitles. That describes
what has been implemented and exercised, not what every film should be.

A row of the film can be spoken on camera, heard over picture, or wordless, and
the pipe script's optional fifth column, `delivery`, says which. `synced` (the
default for a named speaker) routes the line to a lip-sync take. `unsynced`
keeps the speaker's name as `voice` and sends the line through the narration
lane, clocked by its measured file, with subtitles that name that voice;
`Narrator` is shorthand for this. A `wordless` row is written with the speaker
`-` and empty text and becomes a picture event with no voice, no breath and no
subtitle cue. The storyboard row may carry the same `delivery` and `voice`; a
row heard over picture has no `speaker` on the storyboard either, because the
timeline routes any storyboard speaker to a lip-sync take. A
picture row is clocked by its accepted take (`durationRule: take`, the default)
or by an authored hold (`holdSeconds` or `holdFrames` on the storyboard row, a
whole number of frames)
that the timeline fills by trimming accepted, hash-bound segments; a shortfall
is reported like any other coverage deficit. One cost to know before choosing:
an `unsynced` line in the ElevenLabs lane is a second voice for that character
and will not match the same character's on-camera Grok voice. The other way to
put a character's voice over other picture, a Grok take of that speaker used
for its audio only, needs picture windows the timeline does not have yet, and
carries its own identity trade-offs.

Research, source grounding, animatics, auditions, and reviews are useful when
they resolve a real uncertainty. They do not need to become rituals. Make an
initial mix to hear it; refine what matters. Preserve useful source-bound
observations so unchanged work does not need to be reviewed repeatedly.

Check current provider availability and credits before a paid run. Save concrete
job state so interruption does not create duplicate work. Delivery and publication
follow the user's actual request; neither a model response nor a passing test
supplies creative approval.
