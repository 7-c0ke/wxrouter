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
4. The LLM gets the range and the wind ready to quote in the user's units. Asked to convert, it had written
   the m/s figure with "km/h".

## Measured

| where | code | follow-up questions answered by the model | first questions answered by the model |
|---|---|---|---|
| event boxes, 5:45 PM | `main` | 0 of 10 | 13 of 16 |
| one GB10, 9:01 PM | `main` | 0 of 9 | 6 of 8 |
| one GB10, 9:45 to 10:10 PM | the fix | 9 of 9, four runs | 8 of 8, four runs |

- "One GB10" is one NVIDIA GB10 (a team member's own DGX Spark) with the real weather model, the same
  starting analysis and the same LLM (Nemotron on llama.cpp) as the event boxes; the stack was started as
  `fuzzball/30-real-stack.fz` starts it. The event boxes were out of reach that evening.
- With `main`, every follow-up answer held numbers that the model had not given. Example from the event
  boxes: "New York: 3.2 °C to 11.0 °C, wind up to 15 km/h, served by box-a." The model's answer a minute
  later: 5 to 9.7 °C, wind up to 23 km/h.
- With the fix, a wider set of 24 chat cases was run twice (`one-gb10-after-more-cases.txt`): places outside
  the city list, a longer chat, two places in one question, Fahrenheit. In every reply with numbers, the
  numbers equal the forecast of the same turn. 12 of 13 follow-ups had a forecast; the 13th was "Thanks!".
- Before the fix a wind of 1.9 m/s was reported as "1.9 km/h". After it: "7 km/h".
- The safety check still holds with the fix (`one-gb10-after-safety-check*.txt`): 14 messages that are not
  plain commands changed nothing, and 4 plain commands did what they said.
- The fix has not run on the event boxes yet. The first check after the next start there: ask
  "Weather in Chicago for the next 48 hours?" and then "And what about New York?"; both answers must show a
  weather card.

## Also in this branch, from the code review of Nov 14

Each was reproduced on the one GB10 first and checked there afterwards.

- **Demo traffic could lock the page.** With the static rule and the slider at 40 requests a second, the
  waiting requests filled the page server's pool of connections in 38 s: the page's own status call took 8 to
  10 s and timed out once, and the page then greys out its controls, the Off button with them. Now the
  generator has connections of its own and stops sending when 100 requests wait: the status call stayed at
  0.01 to 0.03 s for the whole 150 s, and Off always works.
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
- **The pre-flight check** warns when the model is not on a GPU and when the newest starting weather is old.
- **Seven tests of the page server had been skipped** without notice since an import was added on Nov 14;
  they run again.

## Files

- `event-boxes-1745.json`, `event-boxes-1746.json`: 27 chat questions on the event boxes with `main`.
  `event-boxes-llm-1752.txt`: the LLM alone, asked the follow-up five ways.
- `one-gb10-before.json`: the same questions on one GB10 with `main`.
- `one-gb10-after-1.json` to `-4.json`, `one-gb10-after-more-cases.*`, `one-gb10-after-safety-check*.txt`,
  `one-gb10-after-chips-and-overload.txt`: the same and more, with the fix.
