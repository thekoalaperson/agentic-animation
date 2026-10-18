# Video models

This sheet keeps two kinds of fact apart. **Runner** facts are read from this
checkout's code: what a runner sends and what it refuses. **Provider** facts are
what a provider's own page published on the date given, with the source named.
A provider ability that no runner sends is not a choice available today; using
it is a new lane with its own frozen manifest and its own approval record.

Evidence labels used below: *code* (read from this checkout), *page* (the
provider's published page, read on the date shown), *unverified* (repeated by
operators or third parties, not confirmed here).

## Frame geometry

| Where | Runner (code) | Provider (page) |
|---|---|---|
| Stills | The registered target is a PNG; the row's `size` is recorded, not enforced. | Depends on the image tool. |
| H3 picture | 1344x768 or 768x1344, short edge 768. Staging warns when a source's aspect is more than 0.03 from the planned output. | Self-hosted; see H3 below. |
| Grok dialogue | Sends `resolution: 720p` and `aspect_ratio: 16:9` with one still. After the clip is paid for, refuses any landed clip narrower than 1200x675 or more than 0.03 from 16:9. The dry run warns when a still is more than 0.03 from 16:9. | `aspect_ratio` "is invalid if it is a single image" (page, 2026-10-18). Whether output follows the still's geometry is not stated and has not been qualified here. |
| Delivery | 1920x1080 long form, 1080x1920 Short. | |

Because the dialogue route sends one image, a still far from 16:9 risks a clip
that is refused after payment. `pipeline/derive_still.py` can crop, pad or
scale a registered still first.

## H3 picture (self-hosted)

Runner (code: `h3/stage_finite.py`, `h3/box/finite_client.py`):

- One non-empty prompt per item, 4 to 15 seconds, short edge 768,
  `steps` of 20, 30 or 50, and an output of 1344x768 or 768x1344.
- `expectedFrames` is at most 400 and within 0.75 s of `seconds` at 24 fps. The
  landed clip must have exactly that many frames. The only qualified request in
  code is 7 seconds at 175 frames (`build_measured_deficit_manifest.py`).
- The client sends `task: fl2va` with the first still as a keyframe at
  `frame_index` 0, `flow_shift` 12.0 and `audio_flow_shift` 3.0. Generated
  audio is stripped later.
- A staged item with `lastSource` is copied as `last_frame_still` and sent as a
  second keyframe at `frame_index` -1. No qualification run of this two-keyframe
  mode is recorded at any length, so treat it as unexercised.
- The client hashes itself, and a staged queue refuses to resume under a changed
  client.

Provider: operators report that valid lengths follow 17n+5 frames (175 = 17x10+5)
and that a wrong count fails only after the GPU has rendered it (*unverified*;
the primary page cited for it was unavailable). Kie also hosts H3 as
`minimax-h3/image-to-video`: 4 to 15 s, `768P` or `2K`, first or last frame,
audio always on, prompt up to 7000 characters (page, 2026-10-18). No runner here
sends to it.

Prompting: the model's official prompt format and a bare sentence describing the
event are both workable starting points; compare them on the configured worker.
Operators have observed that anything named in a prompt tends to appear, and
that shot labels can summon objects (*unverified*). A locked camera and a camera
path ending on an authored endpoint are equals; neither is the default.

## Grok dialogue

Runner (code: `kie_dialogue_finite.py`, `dialogue_provenance.py`):

- Route `grok15` (`grok-imagine-video-1-5-preview`), `720p`, `16:9`, integer
  `durationSeconds` from 1 to 15, one still, no references, no last frame.
  The provenance check requires an empty reference list.
- The prompt file must contain the job's exact Devanagari line and stay within
  4096 characters. The still must be at most 20 MB.
- After landing: the clip must hold picture and audio, decode fully, last 0.8 to
  1.25 times the requested seconds, and pass the geometry check above.

Provider (page, 2026-10-18, `docs.kie.ai/market/grok-imagine/1-5-preview`):
`image_urls` up to 7, only one at 1080p, JPEG/PNG/WebP up to 20 MB; `duration`
1 to 15, default 8; `resolution` 480p (default), 720p or 1080p; `aspect_ratio`
1:1, 16:9, 9:16, 3:2, 2:3 or auto, invalid for a single image; audio always
generated and spoken from dialogue in the prompt; no last-frame input; prompt up
to 4096 characters. The page does not document `@image(n)` prompt syntax for
this model; the older `grok-imagine/image-to-video` page does.

References, 1080p and other aspects would each change what the runner sends and
what the timeline admits, so each needs a paid pilot before a batch relies on it.

What one dialogue job sends, with thin demo content and a placeholder URL:

```json
{
  "model": "grok-imagine-video-1-5-preview",
  "input": {
    "prompt": "demo-s003: a lamp on a table. <the job's exact line>",
    "image_urls": ["https://placeholder.invalid/demo-s003.png"],
    "resolution": "720p",
    "aspect_ratio": "16:9",
    "duration": 8
  }
}
```

## Other routes in `kie_video.py`

Each is callable directly through `kie_video.py`. No finite runner or approval
lane in this checkout uses them, and none is exercised for Hindi. Every line is
*code*; re-verify the provider page before a first paid call.

- `grok`: `grok-imagine/image-to-video`; first frame plus `--ref` images in
  `image_urls`; aspect forced to 16:9 when references are present. Page
  (2026-10-18): duration 6 to 30, up to 7 images.
- `omniflash11`: `google/gemini-omni-flash-1-1`; first frame and optional last
  frame, no free references, duration default 8.
- `geminiomni`: `gemini-omni-video`; images plus at most one video, at most 7
  units with a video counting two; with a video the output length is the
  model's.
- `kling3omni`: `kling-3.0-omni/image-to-video`; first frame plus references, no
  last frame, aspect default `auto`.
- `kling3omniref`: `kling-3.0-omni/reference-to-video`; at most four images and
  one video, each extra image a named element.
- `pixverse6`: `pixverse-v6/image-to-video`; at most two images, optional
  multi-clip.
- `pixverse6transition`: `pixverse-v6/transition`; first and last frame only,
  duration default 10.
- `pixverse6ref`: `pixverse-v6/reference-to-video`; up to seven named subject
  references.
- `pixverse6extend`: `pixverse-v6/extend`; one parent video.
- `seedance`, `seedance2`, `seedance2mini`, `seedance25`: `bytedance/seedance-2-fast`,
  `-2`, `-2-mini`, `-2-5`; either first frame with optional last frame, or
  reference-only images and videos, never both; Seedance 2.5's first-frame mode
  sends `adaptive` aspect. A code comment records duration 4 to 15 s and a 720p
  cap for the fast tier, verified 2026-07-22.
- `veo3fast`: `veo3_fast`; reference-to-video with at most three images, no
  duration sent.

## Failures and what each one means

- **Uncertain POST.** The pending record says `POSTING_UNCERTAIN` with no task
  ID. Recover the task ID from the provider before anything else; a retry could
  pay twice.
- **Pending record mismatch or legacy pending record.** The inputs or options
  changed after a paid POST. Resolve the recorded task before changing anything.
- **Provider failure.** The record becomes `GENERATION_FAILED` with the
  provider's response. The dialogue runner halts the batch rather than posting
  again; a retry is a new paid request.
- **Refused after landing.** Wrong geometry, length or missing audio. The clip
  was paid for; keep it and its sidecar, and change the still or the request
  before another attempt.
- **H3.** An uncertain submit must be reconciled; a persisted GPU failure needs
  explicit server recovery; a job fails after two attempts; a staged still that
  changed is refused before submission.
- **Content refusal.** Information about the route and the request, recorded
  with the provider's own words.

## Re-verifying

Before a first paid call on a route, or when a page date here is old: read the
provider's current page for the exact model and resolution and compare it with
the lines above; check the provider's reported success rate for the model if it
publishes one; then make one cheap clip, measure what landed (frames, geometry,
audio, duration) and keep its receipt.

## Prices

Prices belong to the provider and change; read them on the day, for the exact
model and resolution, from the provider's own model or pricing page, and record
the URL and UTC time. A receipt's `creditsConsumed` for a real task is better
evidence for the next manifest than any page. Self-hosted H3 costs worker hours,
which often exceed what a hosted call costs. [Integrity](INTEGRITY.md) says how
a price becomes evidence in a frozen manifest.
