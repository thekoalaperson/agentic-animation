# Technical integrity

These contracts protect files, money, and synchronization. They do not decide
style, story, shot length, emotional tone, or who must approve creative work.

## Files and provenance

Keep generated sources and identify derivatives by their source hashes and edit
parameters. Resolve accepted/current material from recorded decisions rather
than guessing from filenames. When a source changes, identify dependent outputs
so an old cache does not masquerade as an updated result.

Describe evidence honestly: generated, downloaded, technically checked, sampled,
watched, and approved mean different things. An optional review record describes
what was actually reviewed; its absence need not block making an audition.

Media made by hand or brought in (a recorded take, a composed cue, filmed or
painted picture) can be registered with `register_supplied_media.py`, which writes
`<file>.supplied.json`: the file's hash and probe facts, who made or supplied
it, the rights basis, and `rights.providerUpload`. That sidecar is a
declaration. Its maker and rights fields make a false claim a recorded false
statement; they are not proof and not a bypass of any measured or review check.
`providerUpload: forbidden` makes the shared uploader refuse the file, which
guards against an accidental upload and is not DRM. It checks only the exact
path handed to the uploader: an extracted frame, a conformed or stripped copy,
or any other file derived from it carries no sidecar and is not refused.

## Jobs and spending

Respect the user's actual scope. Freeze a finite job list, persist task IDs, and
reconcile an uncertain submission before retrying it. A queue has one owner and
bounded execution. Credit deltas can overlap; keep provider receipts and session
totals. Profile identity alone is not authorization.

A frozen manifest binds more than its job list. It binds the draft it came
from, the source audit, production overrides, the authority file, every still
and prompt, and the price it was frozen at. A batch resumes against those
frozen bytes, so none of them may change between the freeze and the last
reconciled task. Change a bound file and you are making a new manifest, which
needs its own approval record. Each runner's `--freeze` writes the manifest
with these hashes, and a pilot is simply a smaller frozen manifest. A later
manifest may list it under `siblingManifests` so that no job is paid for twice.

A price is evidence with a source. Record the provider's own page for the exact
model and resolution, its URL, and the UTC time you read it. Once a batch lands,
a receipt's task ID and `creditsConsumed` are the best evidence for the next
manifest's rate. Never use a test fixture's rate. Worker hours usually cost more
than API calls. The dialogue runner stops once landed charges pass the
manifest's `maximumCredits`, and it reports each receipted job's rate in
`price-variance.json`. The score lane posts one cue and waits for it to land
before posting the rest. Its wrapper records the wallet change beside
`creditsPerRequest` as an observation only, because the Suno poll reports no
per-task credits.

A version 2 dialogue manifest sends one still at 720p and 16:9. A version 3
manifest reads its model, resolution and aspect ratio from a dated table in the
runner; `--print-schema` shows the table, with each value marked exercised or
provider-published. Its price names the model and resolution it was read for.
A version 3 job may also send up to six reference images after its first frame.
They are hash-bound in the execution request and recorded as provenance
version 3, and the timeline admits the take only with exactly those hashes, in
order. A job without references keeps its earlier request and records byte
for byte. References stay unqualified until a one-clip pilot records how the
first image and the references behave. Each landed job's technical audit
records how far frame 0 is from its source still, as a number with no
threshold. Every definite provider failure is appended to `refusals.jsonl`
beside the manifest as raw facts: task ID, prompt hash, `failCode`, `failMsg`
and credits, with an empty `operatorResponse`.

An H3 request renders 17n+5 frames. Staging refuses any other `expectedFrames`
before GPU work. The continuation builder fills a length outside its qualified
table with an `unqualified` 17n+5 candidate. It snaps up or down only as the
config's `qualifiedRequest.frameSnap` says; no receipt format exists yet. The two-keyframe body (`lastSource`, sent at frame
index -1) is unexercised at every length until a one-job qualification run is
recorded.

A hash-bound old job resumes with its recorded runner and manifest. Use an
explicit migration record when adapting it; do not change old evidence to make
incompatible files appear to match.

## Media and delivery

Measure returned streams and duration rather than trusting requested settings.
Keep picture and native dialogue on the same time map. A retime cannot restore
missing speech. Subtitle placement follows the current measured timeline.

Cache independent picture ranges with their source/settings identities. Reuse
unchanged ranges and remux sound-only revisions. Preserve actual audio bytes and
timing where matching deliverables require them. Measure encoded loudness/peaks,
not just requested filter settings.

Validate copied deliverables before replacing a selected version. Keep the
final manifest tied to the actual files. Use diagnostic tools proportionally:
unit tests establish behavior, media checks establish technical facts, and
creative assessment establishes whether the result works for the film.
