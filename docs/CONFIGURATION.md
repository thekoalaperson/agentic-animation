# Configuration and ownership

Keep three locations distinct: this code checkout, a private production project,
and the storage holding its media. None must be a sibling of another repository.

| Setting | Meaning |
|---|---|
| `AGENTIC_PROFILE` | Optional private JSON profile; start from `examples/profile.json` |
| `AGENTIC_PROJECT_ROOT` | Default private project when a tool needs one |
| `AGENTIC_ENV_FILE` | Explicit private env file; never implicitly read checkout secrets |
| `KIE_KEY` | Kie credential, supplied privately |
| `ELEVENLABS_API_KEY` | Direct narration credential, supplied privately |

An explicit CLI project path wins over defaults. `AGENTIC_PROJECT_ROOT`
overrides the profile's `projectRoot`. Channel, authorizer display identity,
worker paths, and optional tool settings can be read through `profile_value()`
in `pipeline/_common.py`. Tool-specific CLI parameters remain authoritative.
Read the relevant command's `--help`; a profile does not override every argument.

`providers.kie.usdPerCredit` is the operator's own reading of the Kie price.
Dry runs and sidecars use it to show dollars beside credits. It has no default,
and without it the tools show credits only. `AGENTIC_UPLOAD_LOG` names the file
where `_common.upload_ref` appends one line per upload, and `upload_ref` refuses
to upload without a log. The paid runners point it at the project's
`work/upload-log.jsonl`. Run directly with the variable unset, `kie_video.py`
writes `upload-log.jsonl` beside its output stem.

Project structure remains small:

```text
project/
  story/         source citations, script, storyboard, approvals
  art/           canon, stills, motion, provenance sidecars
  audio/         narration, native dialogue, score, effects
  work/          versioned manifests, receipts, reviews, caches
  deliverables/  selected shipping files and final manifest
```

Use stable shot and asset IDs across these directories. Cross-project references
must resolve explicitly, with content hashes. Never discover accepted media by
glob: the accepted registry and current source-bound decisions define reuse.
The content corpus is an input, not an implicit local path or dependency.

## Authority

Two different things are bound before a paid run. The approval record says who
may spend, how much, and on which exact manifest bytes. The manifest's
`authority` file says what the film is: the direction the batch was frozen
against. Changing either one means a new freeze or a new record. Neither
substitutes for the other.

A profile's `authorizer` is a label. Paid execution requires a separate approval
record bound to exact manifest bytes and finite limits. A manifest's
`financialApproval` is reported by the dry run but is never sufficient: agents
write manifests, so only the record authorizes a submission. Version 2 records
use an explicit project root, arbitrary authorizer identity, request/item count,
and a usage limit. They do not require Git. New providers, top-ups, and
publication are never inherited from a sample or another production's grant.

| Lane | Runner | `finiteExecution` keys |
|---|---|---|
| `grok-dialogue` | `kie_dialogue_finite.py` | `jobs`, `generatedSeconds`, `credits` (the manifest's `maximumCredits`) |
| `elevenlabs-narration` | `eleven_batch_finite.py` | `items`, `characters` |
| `suno-score` | `kie_music_finite.py` | `requests`, `credits`, `sourceSeconds` |
| `media-observation` | `observe_media_finite.py` | `jobs`, `credits` (the plan's `maxCredits`) |

`paid_approval.py describe` prints the same table from code.
`paid_approval.py draft LANE MANIFEST --out RECORD` writes a record with status
`DRAFT` and the grant fields empty, which every runner refuses.
`paid_approval.py check RECORD MANIFEST LANE` names whatever is stale. The empty
fields record a person's actual grant, written from their own words when they
give it. An agent does not fill them from its own judgement. Draft the record
after the final freeze, because any later edit to the manifest changes the hash
the record binds.

Recovery needs no new record. `kie_music_finite.py --poll` and
`kie_dialogue_finite.py --reconcile-only` poll and land task IDs that are
already recorded, and never post a new generation request.

Old grants and receipts retain their original schema and runner. Migrate paths
with an external relocation map, not by editing recorded source hashes or
approval wording. A changed runner hash is not permission to resume old jobs.
A relocation map is JSON with `originalEpisodeRoot` (the absolute project path
before the move) and/or `paths` (recorded absolute path to its project-relative
path now). The subtitle exporter and asset registry take it as `--relocation-map`.
A map only says where a recorded file now lives; content hashes are still
compared.

Delivery metadata must explicitly supply upload settings, including video
language, visibility, audience, category, and AI-use disclosure. Unknown choices
can remain marked for review; they are never inferred from a historical channel.
Changing settings changes the metadata hash and invalidates its prior approval.

## Moving computers

Clone the code, install the locked coordinator environment, restore the private
project and selected assets, and authenticate accounts on the new machine.
Validate profile paths and worker access before handing over queue ownership.
Do not synchronize live locks, virtual environments, or two active coordinators.
Stop submissions on the old host before enabling them on the new one.

Use an asset catalog recording stable ID, size, SHA-256, storage object identity,
and local restore path. A generation result URL is temporary transport, not a
backup. Verify restored bytes and a representative usable project before
retiring a source copy. Keep archive keys separate from encrypted archives.
