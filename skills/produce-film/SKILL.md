---
name: produce-film
description: Make or continue an animated film with this repository's generation, editing, sound, and delivery tools. Use for actual film work; ordinary research or website work does not activate production spending.
---

# Make the film

Start with the user's intent, existing material, and actual authority. Read the
private project's current notes and decisions so you can continue without asking
for settled preferences or repeating paid work.

[Workflow](../../docs/WORKFLOW.md) maps the tools; use only the lanes needed now.
[Video models](../../docs/VIDEO-MODELS.md) separates what each runner sends from
what providers publish.
[Configuration](../../docs/CONFIGURATION.md) separates code, private projects, and
credentials. [Workers](../../docs/WORKERS.md) describes external runtime needs.

[Creative](../../docs/CREATIVE.md) and [sound](../../docs/SOUND.md) notes are
optional experience, not mandatory templates. Choose the story structure, style,
shots, pacing, model route, and amount of review that best serve this project.
Do not impose a fixed ceremony on a small task or constrain ambition to a recipe.

Keep [technical integrity](../../docs/INTEGRITY.md): preserve sources, synchronize
media, persist finite job state, and respect spending/publication scope. Be honest
about what was inspected and approved. Record a compact next-action checkpoint
in the private project when it helps another session continue, and name in it
where the direction note is.

## Keep the film's direction

A resumed session should go on making the same film. Keep a direction note in
the private project, in the operator's own words, that lets it: what this film
is doing and the reasons behind its choices. It might cover, for example, how
the film looks and why, the camera idea, the acting register, the score idea and
the voices. Reasons travel better than adjectives, and a line pasted from these
docs is not a direction.

Keep that living note apart from the file a paid runner binds as authority (the
dialogue runner binds `story/GOAL.md` by default). Stable intent can live in the
bound file, edited rarely, because an edit there means a new manifest version.
Version the note rather than rewriting it, and record departures as dated
entries.

When the owner has given no creative direction for this film, a good first move
is to run `pipeline/draw_provocations.py`, and again with `--axes`, before
forming a look, reading only its output and never its data files. Treat what
it draws as a starting push: keep, combine, override or reject it freely, and
record in the direction note what was drawn and what you did with it. This is a
default, not a gate; nothing validates it, and the owner's direction always
wins.

When the note exists it outranks the creative and sound notes for this film. It
never outranks technical integrity, a runner's contract, or the user's own
latest direction.

After a resume or a compaction, re-read the direction and look at the last
accepted pictures and cues before new shot work.

When delegating shot or prompt writing, pass the direction note whole, because a
worker starts with only what it is handed. Put each shot's intent first, and
repeat identity text only where continuity needs it. If seventy prompts share one
camera sentence, check that it was a decision.
