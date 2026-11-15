# The chat's follow-up answers, before and after the fix (Nov 14, 2026)

**What was wrong.** In the same chat, the first weather question was answered by the weather model, and the
second one often was not. The LLM saw its own earlier answer in the conversation, copied its form, and wrote
new numbers and a box name without calling the forecast tool. The page shows a weather card only for a
forecast of the same turn, so such an answer had text and no card.

**What the fix does** (branch `fix/chat-answers-from-the-model`). No weather number reaches the user unless a
forecast ran in the same turn:

1. A question that names a known city or coordinates goes straight to the weather model, and the reply is
   built from its numbers (a teammate's commit `047cd21`, merged here).
2. A follow-up that names no place ("And tomorrow?", "What about the wind?") goes to the weather model with
   the place of the user's earlier question.
3. Everything else goes to the LLM as before. A reply that has a temperature or a wind speed but no forecast
   of the same turn is sent back to the LLM ("call get_forecast now"), twice at most. If it still has numbers
   of its own, the user gets a plain sentence instead of the numbers.
4. When a forecast did run, each temperature and wind figure of the LLM's reply is compared with it (to the
   reply's own rounding, in °C or °F, km/h, mph or m/s). A rain, snow, cloud or humidity figure counts when
   the forecast has none. If one does not match, the user reads the forecast as it is, in the words of the
   rule-based mode. (Added at night, after a second read: see below.)
5. The LLM gets the range and the wind ready to quote in the user's units. Asked to convert, it had written
   the m/s figure with "km/h".

What the check is and is not: it is a net for a failure that was seen, not a proof. It reads figures that
carry a unit (°, C, F, degrees, Celsius, km/h, mph, m/s, knots) or stand behind "high" and "low", and a
range such as "in the 40s"; a number written out in words ("fourteen degrees") or without a unit is not
seen. A question that asks for arithmetic and not for weather ("what is 20 °C in Fahrenheit?") is not
checked. The limits that were seen are listed under "Known limits" below.

## Measured

| where | code | follow-up questions answered by the model | first questions answered by the model |
|---|---|---|---|
| event boxes, 5:45 PM | `main` | 0 of 10 | 13 of 16 |
| one GB10, 9:01 PM | `main` | 0 of 9 | 6 of 8 |
| one GB10, 9:45 to 10:25 PM | the fix | 9 of 9, five runs | 8 of 8, five runs |
| one GB10, 11:50 PM to 1:15 AM | the fix, through the three reads below | 9 of 9 in seven runs, 8 of 9 in one | 8 of 8, eight runs |

- "One GB10" is one NVIDIA GB10 (a team member's own DGX Spark) with the real weather model, the same
  starting analysis and the same LLM (Nemotron on llama.cpp) as the event boxes; the stack was started as
  `fuzzball/30-real-stack.fz` starts it. The event boxes were out of reach that evening.
- With `main`, every follow-up answer held numbers that the model had not given. Example from the event
  boxes: "New York: 3.2 °C to 11.0 °C, wind up to 15 km/h, served by box-a." The model's answer a minute
  later: 5 to 9.7 °C, wind up to 23 km/h.
- In the one run with 8 of 9, "Is that warm for November?" got an opinion from the LLM with no number and
  no forecast ("That's about average for November in Chicago."). In the other runs the LLM's first answer
  to it held numbers, was sent back, and came again with a forecast.
- With the fix, a wider set of 24 chat cases was run twice (`one-gb10-after-more-cases.txt`): places outside
  the city list, a longer chat, two places in one question, Fahrenheit. In every reply with numbers, the
  numbers equal the forecast of the same turn. 12 of 13 follow-ups had a forecast; the 13th was "Thanks!".
- Before the fix a wind of 1.9 m/s was reported as "1.9 km/h". After it: "7 km/h".
- The safety check still holds with the fix (`one-gb10-after-safety-check*.txt`): 14 messages that are not
  plain commands changed nothing, and 4 plain commands did what they said.
- The fix has not run on the event boxes yet. The first check after the next start there: ask
  "Weather in Chicago for the next 48 hours?" and then "And what about New York?"; both answers must show a
  weather card.

## Three more reads at night, and what they changed (11 PM to 1:30 AM)

This branch was read three more times before it was merged, because a fix can break something too.

**First, four readers went through its diff,** and a second reader re-checked each finding. On the command
side they found nothing: no plain command is lost or diverted, and the check on changing tools is nowhere
weaker than on `main` (99,820 fuzzed conversations, 251 hand-made ones). What they did find is fixed:

- **After one forecast the reply was not compared with it**, and a figure in the user's own question
  switched the check off ("Will St. Louis get above 10 °C tomorrow?"). Both are closed (point 4 above).
- **After a kill, "Which box is serving right now?" said "both boxes"**, or named a home box that was down.
  The status, the rule-based answer and the page's tile now follow the router's own order: the box that is
  left. This had been made worse by this branch's first version and is the reason for the second read.
- **"And tomorrow?" after a place outside the city list** was answered for an older city. **A comparison
  with such a place** ("Chicago or St. Louis?") was answered for the known city alone. **"1,000 steps"** was
  read as coordinates. All three now go to the LLM, whose reply is checked. For the follow-up the LLM is
  told which earlier question it follows; without that it answered for the older city in one run.
- **The page's scripts had lost "Cache-Control: no-cache"**: after a restart a browser could run an old
  `app.js`. They carry it again, at the root and under the Fuzzball path.
- **One real action per box and turn**, whatever arguments the LLM adds; "Kill box A (sim)" and five more
  ways of saying "not for real" do not run the real kill.
- **A stopped fine-tune** reads "stopped" in the Events card, not "failed" in red.
- The Demo traffic slider says when the generator holds requests back. The model swap puts right a link
  that points at other weights than the boxes serve.

**Then three more readers were told to break those fixes,** and each of their findings was re-checked by
another reader. The command side held again (58,988 command messages: nothing a plain command asks for was
lost, nothing new acts). They found three things that the fixes themselves had made worse, and some that
were left. All are fixed (commit `992dd39`), each with a test:

- **"Which of these is warmest?"**, a rule added after the first read, compared only the newest three
  places and none that the LLM had answered, so it could name the wrong one; and it took any "which" or
  "where" question with a word like "hotel" or "window" for a comparison. Now the weather model answers it
  only when the message is made of comparison words, the question before it was about the weather, and
  every place asked about can be compared (two or three cities of the list, the whole chat sent with the
  question). Otherwise the LLM gets the question.
- **"… in Fahrenheit this time?" with a place** switched the number check off, because "that in Fahrenheit"
  is allowed as arithmetic. It is arithmetic now only when the message names nothing else.
- **After a status call, the status's own numbers counted as weather**: the cooldown of 10 s let "0 °C to
  10 °C" pass. Now only a place's coordinates and what a tool calls a temperature count, and "a high of
  196 ms" is not read as a temperature.
- **A sentence was cut at the dot of "St."**, so a reply could start with "Louis is 30%". Fixed, also for
  "Mt." and "vs."; "3 p.m." ends a sentence only when a new one follows.
- **"No box in service"** was shown while a drained box, the router's last resort, answered every request.
  The status and the tile now agree with the router's own order in 960 of 960 states of the two boxes.
- **Without the LLM** (the rule-based mode, also the fallback when the LLM cannot be reached) "Kill box A
  (sim)" still ran the real kill, and "2,000 steps" was read as "000". Both fixed.
- Smaller: a closing line after a change ("I'll check again whenever you ask") is not sent back to the LLM;
  "km/hr" counts as a wind figure; a reply that is cut to nothing becomes the model's numbers; a fourth
  city in one question is named as not shown instead of being left out of a "Warmest" line; the model swap
  refuses `--to` the link itself and says so when it leaves the boxes on different weights.

**Last, four readers looked at the commit with those fixes** for anything it had made worse, again with a
second reader on each finding and on each proposed fix. Three of the four areas were safe as they stood.
The fourth was not (fixed in commit `c9998ba`):

- **The words for snow, cloud, thunder and sun, added to catch made-up statements, also matched place
  names.** "Chicago's range is 3.7 °C to 8.6 °C, while Thunder Bay's is -6.3 °C to 1.2 °C" came out as
  "... 8.6 °C6.3 °C to 1.2 °C", and "Thunder Bay", "St. Cloud" and "Sunshine Coast" were cut from the start
  of honest replies. Now the places of the turn's forecasts are set aside before anything is cut, and a
  cut that would run into a number is not made: the reply then becomes the model's numbers.
- 0 °C and 32 °F pass only where the reply compares with the freezing point ("above 0 °C"); they passed
  anywhere. "In the low 200s" for milliseconds is not a vague temperature. "0.6°–9.2°" is a range, not a
  minus.
- After a change, an announcement ("I'll get the status now") is still followed up; a closing offer
  ("I'll check again whenever you ask") is not.
- A fine-tune that crashed (a signal that nobody sent) reads "failed" again; only a stop reads "stopped".
  The Demo traffic note is cleared by Off. The model swap makes no claim about a link that was moved by
  hand. After an answer that was withheld, "And tomorrow?" is no longer answered for an older city.

A fix that a reader proposes needs a second reader as well. In these three reads six proposed fixes were
unsafe as written: for one, a new word list on its own would have run the real kill for
`Kill box A _simulated_` in the rule-based mode, where the old test refused it. The second reader caught
each of the six, and the corrected version is what went in.

What the readers' own test sets give on the final code (stand-in LLM, no GPU):

| test set | result |
|---|---|
| 110 recorded real replies that came with a forecast | 110 pass unchanged |
| 377 plain honest replies, and 160,000 built from random forecasts | 0 refused or changed |
| 30 honest replies of unusual shapes (own arithmetic, "about 7 °C warmer") | 11 replaced by the model's numbers |
| 150 dishonest replies, after a forecast | 92 caught, 58 pass (see "limits" below) |
| the same 150 with no tool step at all | 119 caught, 31 pass |
| 51 more honest replies in the LLM's own style, from the last look | 50 shown as written (37 before its fixes) |
| 40 more dishonest replies, from the last look | 28 caught, 12 pass (19 passed before its fixes) |
| 4,000 random LLM scripts | never more than 8 LLM calls, no exception, no empty reply |
| 58,988 command messages | the LLM's gate decides the same as before these fixes for every one |
| the status and the page's tile against the router's own `order()` | the same in 960 of 960 states |

Checked on one GB10 with the real model and Nemotron at 1:15 AM (`one-gb10-second-read.txt`, made by
`night_second_read.py` and `night_chain.py`; every earlier check of this page is in the same file again):

| check | result |
|---|---|
| 41 chat cases of the three reads: replies with a forecast that hold a figure the forecast does not have | 0 |
| "Fail box A", then "Which box is serving right now?" | "box-b is serving right now."; before and after the kill: both boxes |
| Places outside the list (St. Louis, Reykjavik, Knoxville), in °C and °F | every figure is the model's |
| Places with a weather word in their name (Thunder Bay, St. Cloud), alone and beside Chicago | the name stays, every figure is the model's |
| "Is it warmer in Chicago or St. Louis?" and two more comparisons | two forecasts each; every figure is the model's |
| Chicago, then St. Louis, then "And tomorrow?" | St. Louis, from the model |
| The same kind of chain: three chains, seven follow-ups, three times each (`night_chain.py`) | 21 of 21 answered for the place of the question before (St. Louis, Kansas City, Reykjavik) |
| "Which of these is warmest?" after three cities of the list | three forecasts and "Warmest: **Tokyo**", in 0.3 s, no LLM |
| The same after four cities, and after a place outside the list | the LLM: one place's forecast, the model's numbers, no comparison |
| "And St. Louis, in Fahrenheit this time?" | a forecast for St. Louis: the model's numbers in °F |
| "Which box is up, and how cold will St. Louis get tonight?" | the status and a forecast; the model's numbers |
| Four "not for real" commands | nothing killed (the server's restart count did not move) |
| Follow-ups and first questions, as measured before | 9 of 9 and 8 of 8 (the eight runs of that night: 9 of 9 in seven, 8 of 9 in one) |
| 24 more chat cases, as measured before | 12 of 13 follow-ups had a forecast; the 13th is "Thanks!" |
| The 14 messages that are not plain commands; the 4 plain commands | nothing changed; all four worked |

Known limits, seen in these runs:

- **The number check is a net, not a proof.** It reads figures that carry a unit. In the readers' set of
  150 dishonest replies it caught every one with other numbers, with numbers off by one, with a second
  made-up place, and of the shape seen on Nov 14 (29 of 29). What passes: a number without a unit, a number
  in words ("fourteen degrees"), Kelvin, Beaufort, a made-up number that happens to equal one of the
  forecast's own in another unit or a coordinate of the place, other kinds of weather (pressure, UV index),
  and the figure of the user's own question ("Yes, it will reach 25 °C" after "Will it reach 25 °C?"): that
  figure is allowed so that the LLM can say "it will not be over 10 °C".
- **A question about the earlier answer** ("Is that warm for November?", "So should I take a coat?") gets
  a fresh forecast of the place and no opinion. Once that night the LLM gave an opinion instead, with no
  number and no forecast: "That's about average for November in Chicago." **A question with a threshold**
  ("Will St. Louis get above 10 °C tomorrow?") gets the model's range, and sometimes a yes or no with it.
- **"Which of these is warmest?"** with four or more places, with a place outside the city list, right
  after a command or a status question, or as the seventh question of a chat or later goes to the LLM,
  which answered with one place's forecast: true numbers, no comparison. Naming up to three cities of the
  list in the question works ("Is it colder in Chicago or Boston?"): the weather model answers that.
- **"Can you give me that in Fahrenheit?"** is the LLM's arithmetic on the numbers on screen, not a new
  forecast, and has no card: it wrote 37 °F to 50 °F where the model's own answer in °F is 37 °F to 49 °F
  (the °C figures were rounded first), and its arithmetic is not checked. With more words around it ("Could
  you repeat that in Fahrenheit?") the request is checked like a weather question, and the LLM then has to
  ask the model again. The °F switch of the page gives the model's own.
- **An announcement without a tool call is not always followed up.** "I'll kill the forecast server on box
  A now." with no tool call is shown as the answer, and nothing runs, as on `main`: type the command again.
- After a reply in which the LLM only asked back ("Do you mean St. Louis, Missouri?"), "And tomorrow?" is
  answered for the earlier city, named in bold.
- A known city and a place outside the list joined by a comma, "&", "/" or "plus" ("Chicago, St. Louis")
  still get the forecast of the known city alone.
- **"Recover box B"** (no longer a chip) changes nothing, because a killed server restarts by itself. The
  LLM reads the status and says so, but in several such replies that night it added "(This is a real
  action on the box.)", and once it asked for a kill, which the check refused. Nothing was done in any of
  them. The rule-based mode says it plainly.
- **A statement without a number can still be the LLM's own.** As a follow-up, "Will it rain in St. Louis
  tomorrow?" was once answered "St. Louis tomorrow: no rain forecast." with no forecast and no card (in
  the last run the LLM asked the model, and the reply said that the model has no rain output). The check
  reads numbers; statements about rain are cut only from a reply that came with a forecast.

## Also in this branch, from the code review of Nov 14

Each was reproduced on the one GB10 first and checked there afterwards.

- **Demo traffic could lock the page.** With the static rule and the slider at 40 requests a second, the
  waiting requests filled the page server's pool of connections in 38 s (498 of 500): the page's own status
  call took 7 to 10 s and timed out once, and the page then greys out its controls, the Off button with them.
  Now the generator has connections of its own and sends nothing more while 100 requests wait: in the same
  test the queue stayed at about 100, the wait at about 5 s, and the status call at 0.01 to 0.05 s for the
  whole two minutes. Off always works; 20 s after it the traffic was gone.
- **"Which box is serving right now?"** (a chip on the page) was answered "box-a is serving" under the
  balanced rule, where both boxes serve. Now: "Both boxes are serving". The Events card no longer logs
  "Switched" lines under a rule that has no active box.
- **The chip "Recover box B"** could not stop the fine-tune that the chip beside it starts. It is now
  "Stop the training job on box B".
- **A box card said "training"** whenever its GPU was busy with other work. It now says "GPU busy" unless the
  fine-tune job runs.
- **"Simulate a failure on box A"** ran the real kill in LLM mode. It is refused now, as in the rule-based mode.
- **One command, one change:** the same kill or start is not carried out twice in one turn.
- **A test hook** (`POST /api/box/<box>/mode`) could put a real forecast server into "error" and call it
  simulated. A box that runs the real model now refuses it.
- **The weather card** says from which analysis its hours count ("next 48 h from the analysis of ...").
- **The model swap** (`bench/swap_weights.py`) leaves a box alone that already serves the weights asked for.
  Tried there and back with the real weights: 5.5 and 6.5 s a box, 240 requests, 0 failed; asked a second
  time for the same weights, it changed nothing.
- **The pre-flight check** warns when the model is not on a GPU and when the newest starting weather is old.
- **Seven tests of the page server were skipped** whenever their file was run on its own (the documented
  command), since `ui/server.py` began to import a neighbour file on Nov 14. They run again.

## Files

- `event-boxes-1745.json`, `event-boxes-1746.json`: 27 chat questions on the event boxes with `main`.
  `event-boxes-llm-1752.txt`: the LLM alone, asked the follow-up five ways.
- `one-gb10-before.json`: the same questions on one GB10 with `main`.
- `one-gb10-after-1.json` to `-5.json`, `one-gb10-after-more-cases.*`, `one-gb10-after-safety-check*.txt`,
  `one-gb10-after-chips-and-overload.txt`: the same and more, with the fix.
- `one-gb10-final-check.txt`: every check once more with this branch's code of 10:30 PM, before the reads
  described above (pre-flight, the model swap there and back, overload, the chips, the follow-ups, both
  safety checks).
- `one-gb10-before-chips.txt`, `one-gb10-before-overload.txt`: the chips and the overload before the fix.
- `one-gb10-second-read.txt`, `one-gb10-second-read.json`, `one-gb10-second-read-chains.json`,
  `night_second_read.py`, `night_chain.py`: the checks after both reads with the last commit of this branch
  (12:45 AM), and every earlier check once more with that code.
- `one-gb10-whatif-with-the-boxes-own-llm.txt`, `night_whatif.py`: WhatIf's agents on the box's own LLM
  instead of a hosted one (no key): which questions reach them, and what they diagnose on a healthy stack
  and after a real kill.
