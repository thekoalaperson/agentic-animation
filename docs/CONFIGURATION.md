# Configuration and ownership

Keep three locations distinct: this code checkout, a private production project,
and the storage holding its media. None must be a sibling of another repository.

| Setting | Meaning |
|---|---|
| `AGENTIC_PROFILE` | Optional private JSON profile; start from `examples/profile.json` |
| `AGENTIC_PROJECT_ROOT` | Default private project when a tool needs one |
| `AGENTIC_WORKSPACE_ROOT` | Private root containing projects and shared assets |
| `AGENTIC_ENV_FILE` | Explicit private env file; never implicitly read checkout secrets |
| `KIE_KEY` | Kie credential, supplied privately |
| `ELEVENLABS_API_KEY` | Direct narration credential, supplied privately |

An explicit CLI project path wins over defaults. The two root environment
variables override their profile values. Channel, authorizer display identity,
worker paths, and optional tool settings can be read through `profile_value()`
in `pipeline/_common.py`. Tool-specific CLI parameters remain authoritative.
Read the relevant command's `--help`; a profile does not override every argument.

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

A profile's `authorizer` is a label. Paid execution requires a separate approval
record bound to exact manifest bytes and finite limits. Version 2 records use an
explicit project root, arbitrary authorizer identity, request/item count, and a
usage limit. They do not require Git. New providers, top-ups, and publication are
never inherited from a sample or another production's grant.

Old grants and receipts retain their original schema and runner. Migrate paths
with an external relocation map, not by editing recorded source hashes or
approval wording. A changed runner hash is not permission to resume old jobs.

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
