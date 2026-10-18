# agentic-animation

Tools and agent guidance for making consistent animated films. Bring your own
story, approved references, provider accounts, and GPU workers. Productions live
outside this checkout; this repository contains no films, private scripts, voice
identities, account configuration, or model weights.

One exercised route combines reference-grounded stills, MiniMax H3
non-dialogue picture, Grok native dialogue, ElevenLabs narration, Suno scoring,
self-hosted sound effects, and source-bound editing. The Hindi performance with
Hindi/English subtitles is the qualified preset; other languages require their
own validation. These are explicit tools, not an unattended one-command movie
generator.

## Start

Install Python 3.11+, [uv](https://docs.astral.sh/uv/), and FFmpeg (`ffmpeg` and
`ffprobe` on `PATH`, needed by `make test`), then:

```sh
uv sync --locked
uv run python pipeline/edit/compile_production_script.py --help
```

Copy `examples/project` to a private directory. This small synthetic example can
be compiled without credentials, media, a GPU, or a Git repository:

```sh
cp -R examples/project /tmp/animation-demo
uv run python pipeline/edit/compile_production_script.py \
  /tmp/animation-demo /tmp/animation-demo/story/script.txt \
  /tmp/animation-demo/work/script-v1
```

For real work, copy `examples/profile.json` outside the checkout, configure it,
and set `AGENTIC_PROFILE` to that file. Pass production paths explicitly to the
tools. Read [configuration](docs/CONFIGURATION.md) and [workflow](docs/WORKFLOW.md).
Provider credentials are environment variables or an explicit private env file.
Nothing in a sample profile authorizes spending or publication.

## Layout

| Path | Purpose |
|---|---|
| `pipeline/` | Generation, provenance, and review tools |
| `pipeline/edit/` | Measured timelines, sound, subtitles, and packaging |
| `pipeline/h3/` | Finite H3 jobs and source-bound ingestion |
| `pipeline/open_media/` | GPU sound effects, HD ranges, and audio remux |
| `docs/` | Workflow, configuration, creative guidance, and worker contracts |
| `skills/produce-film/` | Agent entry point for a requested production |
| `examples/` | Synthetic inputs and configuration, never real production data |

Read [AGENTS.md](AGENTS.md) for the operating contract. Each tool exposes its
arguments through `--help`; GPU media commands run on qualified workers, not the
coordinator. The coordinator lockfile does not install a GPU model environment.
See [workers](docs/WORKERS.md) before running inference or rendering.

Historical project data and old runners belong in private recovery storage.
Hash-bound jobs must resume with their recorded runner version; a newer checkout
does not silently migrate an old paid authorization or approval.

## Verification

Run `make test` from the repository root. The retained unit tests use temporary
synthetic inputs and mocked providers; they must not require private episodes,
active credentials, or paid calls. GPU/model qualification and creative review
are separate from unit tests.

Third-party software, fonts, model weights, provider access, and generated media
retain their own terms. No third-party model or font payload is redistributed here.
The reusable code and guidance use the [MIT license](LICENSE); private productions
are not part of that license or this repository.

If this project helps your commercial work, please credit **agentic-animation**
and link to this repository. Public credit is appreciated; the binding license
condition is preservation of the copyright and license notice in software copies.
