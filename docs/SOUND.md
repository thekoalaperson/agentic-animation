# Sound notes

Sound can shape the story from the beginning. Keep narration, native dialogue,
score, atmosphere, and effects separately editable. The H3 picture adapter strips
its generated audio; that is the adapter's contract, not a judgment about every
model or every possible production.

## Music

The instrument vocabulary comes from the film's world, and these notes name
none. Choices with real costs, none of them recommended: one instrument or an
ensemble; music that arrives late or a bed that runs through the film; with words
or without; cues grown from one seed take or written independently; one long cue
cut to picture or many short ones; music inside the story's world or outside it;
no score at all. A choice that gives the film continuity usually takes some
contrast away, and a choice that makes an entrance an event leaves the rest of
the film to voices and the world.

Give the score meaningful presence, including speech-free moments, or decide it
has none. Make room for voices deliberately instead of stacking attenuation until
the music disappears. Action peaks and quiet scenes can have different balances.
Judge the whole scene.

Community guidance, as of October 2026, to re-test on the configured model: each
score request is independent, and a request cannot recall another cue's theme
from a description in prose. A plan that asks a later cue to return to an earlier
one gets a new piece. Hits and silences land where the edit puts them, not where
a prompt asks for them.

What this repository can do for continuity today: cut one long take into several
placements with `sourceStart`/`sourceDuration`, which must cover the placement
exactly with no retiming; shape the score's level across shots with the authored
wave that `assemble_longform.py --wave` reads (`shots[].music_energy`); and lower
or silence a stretch with `musicBreaths`. Musical edits often preserve a cue
better than stretching it to an unrelated length. Listen to the returned
composition and use its actual duration. Preserve starts, endings, and exact
prompts when they are useful for later revisions.

Spotting decides where music enters, leaves, meets a picture moment and falls
silent, and continuous scoring is one answer. The fields that carry it are the
`scorePlacements` in `timeline-inputs.json` (`movement`, `take`,
`timelineStart`/`timelineDuration`, an optional `sourceStart`/`sourceDuration`,
`gainDb`, `fadeInSeconds`/`fadeOutSeconds`) and `musicBreaths` (a range of
shots, a level and a reason; level 0 is silence). Today the mixer requires
placements to cover the measured clock from start to end, so silence is made
with a breath inside a placement rather than with a gap. A film with no audible
score therefore still needs one admitted take, held at level 0 by a breath.

The score lane today accepts instrumental requests only, with an empty lyrics
prompt. The music ranker and the effects renderer read eight fixed emotion keys
(`awe`, `dread`, `devotion`, `grief`, `hope`, `scale`, `kinetic`, `intimacy`).
Those are the tools' input names, not a vocabulary a film has to think in.

In a serial: does a theme belong to a person, a place or an idea? Does it return
unchanged, or changed by what the story has done? Do some episodes hear none of
it?

## Effects

How literal the world's sound is, how much the score carries instead, how much
silence the film allows, and whose ears the viewer hears through are decisions
for this film. Strategies that answer them differently each lose something;
invent the one this film needs. Sound from outside the frame, and sound
perspective (near or far, inside a head or out in the room), are tools as well.

A dialogue clip arrives with its own room under the line, and today that room
stays with the line through the edit. Keeping the room, replacing it with an
authored bed, and playing lines dry are all choices, but the last two need the
line separated from its room, which no qualified tool here does yet. When the
room stays, prompt the clip to match the scene and do not ask for the same ambience
twice, once in the clip and again in an effects bed.

Short concrete material and action prompts have produced more variety than a
fixed tail of genre words appended to every request. Try different acoustic
mechanisms. Story reasoning can stay in the brief while the source prompt stays
simple. Vary attack, body, movement, and decay through selective layers,
pitch/speed changes, and edits.

A stable middle section can extend a texture without repeating its attack. Keep
raw originals and treatment parameters. Speech-like artifacts or competing music
may be unwanted in an effect; listen to what the source actually contributes.
Self-hosted text-to-audio is an available route, not a compulsory creative choice.

## Voices

What the lanes do today:

- The narration lane (`eleven_batch_finite.py`) takes one voice, one model and
  language `hi` per manifest, with voice settings fixed at the root for the
  whole batch. It sends no pronunciation dictionary. The numbers in its
  docstring show the format; they are not a performance.
- The dialogue route invents a voice for each clip from its still and prompt.
  Nothing keeps a character's voice the same from clip to clip, and nothing here
  measures the drift yet, so listen across clips.
- A dialogue clip is at most 15 seconds.

Casting is a set of questions. How old does each voice sound, and how near? Which
register of Hindi does each person speak? Is the narrator a storyteller, a
witness, a character, or absent? Auditioning two or three genuinely different
castings before settling usually costs less than recasting later.

A quiet scene can be whispered or spoken at full voice in a quiet place, and a
loud one can be shouted or held back; neither is the scene's default.

Subtitles translate rather than explain. When a speech is split across rows, each
half carries only its own words.

Another school of performance makes a cast or human-performed voice the clock and
drives picture from it. No lane for it exists here, and it is unqualified for
Hindi.

## Audition and refine

Make a mix to hear it. Compare raw sounds and a few labelled scene versions when
that answers a real question; do not build an approval ceremony around every cue.
Waveforms and loudness measurements help diagnosis but cannot establish taste.
Record listening limits honestly and preserve useful observations for reuse.

Inspect the combined gain of all processing stages and measure the encoded
result. Preserve complete speech and native sync. A sound-only revision can
reuse accepted picture and HD caches rather than regenerate the film.
