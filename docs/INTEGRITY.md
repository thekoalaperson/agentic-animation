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
`price-variance.json`.

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
