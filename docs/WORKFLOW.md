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

Research, source grounding, animatics, auditions, and reviews are useful when
they resolve a real uncertainty. They do not need to become rituals. Make an
initial mix to hear it; refine what matters. Preserve useful source-bound
observations so unchanged work does not need to be reviewed repeatedly.

Check current provider availability and credits before a paid run. Save concrete
job state so interruption does not create duplicate work. Delivery and publication
follow the user's actual request; neither a model response nor a passing test
supplies creative approval.
