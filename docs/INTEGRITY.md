# Technical integrity

These contracts protect files, money, and synchronization. They do not decide
style, story, shot length, emotional tone, or who must approve creative work.

## Files and provenance

Keep generated sources and identify derivatives by their source hashes and edit
parameters. Resolve accepted/current material from recorded decisions rather
than guessing from filenames. When a source changes, identify dependent outputs
so an old cache does not masquerade as an updated result.

A picture review ledger at `schemaVersion` 2 admits a segment only when it
declares its `sourceSha256` and an `origin` `{kind, record, recordSha256}`
whose record sits below the episode, matches that hash and names those bytes
under the key its kind uses: an `h3-ingest` receipt in `items[].silentSha256`,
a `kie-video` sidecar in both `sha256` and `outputSha256` beside a `taskId`, a
`derivative` record in `outputSha256` with a different parent `sourceSha256`
and non-empty `parameters`, and a `supplied` record in `sha256` or
`fileSha256`. A derivative's parent hash is checked for shape only, not yet
resolved to an admitted record of its own. A version-1 ledger keeps its
recorded behaviour. A wordless picture row needs a declared `sourceSha256`
under either version.

Describe evidence honestly: generated, downloaded, technically checked, sampled,
watched, and approved mean different things. Auditions, animatics, contact
sheets and model observations need no record. Admission to the measured
timeline does: a dialogue or score take enters only through a source-bound
schema-2 review export, and H3 picture only through a ledger verdict.
`admit_review_decisions.py` writes either from a plain decisions file that
says what review was actually done: who decided (`reviewedBy`, verbatim; an
agent writes `agent:<name>`), when, and optionally how. It refuses a take whose
bytes or sidecar changed, and it never turns a note or an unlisted take into a
decision. The timeline admits the verdict words `accept`, `accepted`,
`approve`, `approved`, `approved-bounded` and `use`; any other word is recorded
and admits nothing. Who reviewed and by which method are reported per lane,
and their absence never refuses admission.

The two admission routes differ on purpose. A review export binds the take, its
sidecar and the lane's manifests, so changing the script or a manifest makes it
stale; a ledger verdict binds the picture bytes and, from version 2, their
origin. Binding H3 verdicts to the manifests would add ceremony without
protecting anything, and loosening the dialogue binding would weaken it.

## Jobs and spending

Respect the user's actual scope. Freeze a finite job list, persist task IDs, and
reconcile an uncertain submission before retrying it. A queue has one owner and
bounded execution. Credit deltas can overlap; keep provider receipts and session
totals. Profile identity alone is not authorization.

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
