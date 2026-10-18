# Creative notes

Use these as useful experience, not a style law or checklist. The story, the
user's taste, and the actual result outrank a recipe. Explore freely; retain the
ideas that make this particular film better. Nothing on this page is meant to be
copied into a prompt.

## Finding this film's look

The look is decided in the stills, because every motion route starts from them.
Choose it early and on purpose, from a reading of this story rather than from a
previous film or from these notes.

A look is a set of decisions, not an adjective. Its axes include the medium the
images seem made in, how far faces and bodies are simplified, line and edge,
surface texture, the logic of colour, where light comes from and how much dark
the frame may hold, and what is allowed to look strange. None is ranked, the
list is not complete, and a film may decide an axis that no list names.

The first look that comes to mind is probably the one everyone else would reach
for too. If a look could move to another story unchanged, it has not yet been
chosen for this one. Every look gains something and gives something up; know
what this one gives up.

Describe the look in terms a renderer can act on rather than in praise words.
Where a prompt carries a style passage at all, it can come after the subject and
the action and stay short. An image often carries a look better than words: a
registered style frame placed in each canon sheet's references lets the sheets
inherit it, and the stills inherit the sheets.

Identity can be held in several ways, alone or combined: a turnaround, a hero
portrait with expressions, one sheet per visual state the story needs, location
plates at the times of day the film uses (from as many angles as the director
wants, including none), props on a sheet of their own, or the first registered
shot of a figure standing as its authority.

Colour can follow the emotional movement, or hold still while everything else
moves. When two looks both seem right, a few test frames of one key moment made
in each can settle it more cheaply than argument.

For sacred material, reverence lives in how figures are treated, not in a
shading technique. The question is which traditions of telling and of picturing
the film stands in, or stands against.

## Consistency without sameness

Shared references help recurring identities, costumes, props, and locations
survive different framings. Story-time state matters: an earned wound or trophy
should make sense at the moment shown. In researched adaptations, preserve
source distinctions instead of blending incompatible versions accidentally.

One clear instant often gives a motion model a stronger starting point than a
still containing both cause and consequence. Separate state images can help
complex events. Visible contact, gaze, and geography often communicate an action
better than more directional adjectives. These are options, not required forms.

## Adapting a finished screenplay

When the user's source fixes the words, that rule binds the words, not the
structure. Action lines say what happens, not how it is shot. A time stamp in a
script is story time, not a grade. Reading the page straight through in order is
also a decision, and it can be made on purpose.

Some questions to answer before shots: whose experience is this; what does the
first image argue; what plays without words; what is remembered rather than
shown; where does escalation give way to stillness.

## Staging and performance

On image-to-video routes the first frame fixes who is where, which way they face
and how they hold themselves. When the cut depends on that, decide it in the
still. The dialogue route animates whatever still it is given, so framing is
chosen in the still, not by the route.

Staging is a set of decisions with costs: where the viewer stands relative to
the people, whether a conversation keeps one side or crosses it, distance and
height as a statement of the relationship, how long a view is held, and whose
experience the viewer borrows. Acting has its own axes: how much a body moves,
how codified a gesture is, how much is carried by stillness. How much violence
is shown, and through what, is a decision too. None of these has a default here.

A provider refusal is information about the route. Rewording is one response;
changing what the shot shows, or how it shows it, is another.

## Directing time inside a clip

A lip-sync clip has the time before the line, the line, and the time after, and
each can be directed. Say what the listener does as an action rather than what
they avoid. Some operators write time-coded segments into a prompt; that is
dated third-party guidance about how models read prompt text, so confirm it by
audition on the configured model before relying on it. One main action per
lip-sync clip is a default a director may override.

## Directing motion

Different shots can reveal geography, intention, reaction, detail, or a changing
relationship. A moving camera and a still camera can both be powerful. Phase
complex action when that helps a model understand it; use the chosen endpoint's
actual controls and reference semantics. [Video models](VIDEO-MODELS.md) lists
what each runner sends.

Prompts benefit from a clear source, an understandable event, a camera intention,
and somewhere to end. These are questions to consider, not five mandatory blocks.
Write as little or as much as the model and scene need. Check the exact submitted
payload before blaming a model.

Unused accepted coverage, alternate angles, retiming, and true continuation offer
different ways to improve a sequence. Choose by meaning and temporal quality,
not a universal duration grid. Preserve the source so experiments stay reversible.

## Cutting

Cutting decisions come in pairs, and neither side is the default: one take for a
line or several; cut on the question or on the answer; hold the speaker or the
listener; sound leading picture or picture leading sound; a hard cut or a
dissolve; a scene that stays put or one that leaves and returns. The timeline
today joins picture with hard cuts and plays each line over its own clip, so a
dissolve, a line heard over its listener and sound that leads its picture cannot
be assembled yet.

An action line describes an event, not a shot. Camera and cutting can be one
idea held across the whole film. Notice the patterns that arrived without being
chosen, and keep the ones you would choose on purpose.

## Episodes and serials

How much direction the episodes share, and how much each one writes for itself,
is a decision for the serial. Episodes handed out without a written direction
tend to converge rather than diverge, because each is made from the same
defaults. How an episode opens and ends, and whether it recaps, are choices
with costs in runtime and in the viewer's orientation. What does each episode
hand to the next?

## Sources and tellings

Name the text, and the recension or edition, that each sequence follows. Keep
variants attributed rather than merged. Note departures in the project's own
source notes with a tag the project chooses. This is a convention, and nothing
validates it. A profile may point an agent at the project's sources through the
optional `researchCorpus` key described in [configuration](CONFIGURATION.md).

## Shaping the film

Open with a reason to care. Let picture, sound, and performance carry information;
use text, silence, stillness, and spectacle deliberately. A long film can sustain
ambition when its sequences introduce new meaning and share coherent world state.
There is no mandatory act count, runtime, shot formula, or reveal pattern.

Some finishing questions: how does the film open and close; which beat, if any,
becomes the Short; is there a textless master for the thumbnail; do chapters
follow the episode's own structure; what does the AI-use disclosure say, as an
explicit choice recorded in the delivery settings?

Compact reviews in story order help decisions. Thumbnails benefit from one
legible event or tension; strong copy stays truthful. A Short works best when it
resolves a worthwhile beat. These are editorial judgments, not validator rules.

## When a take goes wrong

A strong first pass is often enough. Spend iteration on consequential problems
such as wrong identity, broken contact, or a missing story beat. A local edit can
preserve a good performance better than replacing everything for a small defect.

Before changing anything, name the one variable you think moved, then change
only that. A refusal, a drifting identity and a degraded take usually point at
different variables; which one is a reading of this take, not a rule.

## If you want a push

Optional. `pipeline/draw_provocations.py` draws two story-neutral directorial
constraints at random from a deck kept beside it. Run it rather than reading the
deck: a model that reads every card picks the most salient ones, and operators on
the same model then pick the same ones. Reject, combine or override what it
draws. Nothing records or validates a draw, and the user's direction always wins.

Beside it, an offer and not a step: it can help to name to yourself the
competent default adaptation of this story, and then decide where this film
differs from it.
