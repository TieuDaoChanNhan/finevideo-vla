# full_chain_from_scratch, greedy, max_new_tokens=4000 (2x the original 2000-token run)

## Language description fed to the model (the ONLY input)

```
### Title: Home cooking tutorial
### Context: A person chops vegetables on a cutting board in a kitchen.
### Keywords: cooking, kitchen, food preparation
```

## Headline result: agent/pose is STILL never reached, even at 2x budget

The original 2000-token run's working theory was "cosmos ate the budget before agent got a turn" (787/2000
tokens on cosmos). This run doubles the budget to 4000 -- the maximum this checkpoint supports at all
(`config.json`'s `max_position_embeddings` is a hard 4096, this prompt is ~40-50 tokens, so 4000 is close to
the ceiling). Generation used the full budget (cut off mid-`<cosmos>` block at the very end, not an early
`</s>`/EOS) and still produced **zero** `<fps_N>` tokens -- zero agent content of any kind.

This rules out "just needed more room" as the explanation. What actually happened with the extra budget:
the model looped through the same **8-way modality cycle** (caption -> seed2 -> snac -> speech -> cosmos ->
snac -> ...) roughly twice as many times (8 cosmos blocks now vs 3 before, 150 snac tokens vs 75, 160 seed2
tokens vs 96) -- agent was never among the modalities greedy chose to visit, at any point in either run.
This points to a **generation-time (not budget) gap for this specific prompt-context**: whatever the model
learned about when to transition into `<agent>`, greedy's fixed highest-probability path from this exact
"home cooking tutorial" scene doesn't pass through it, no matter how long it's given.

## seed2 blocks: 5 found, still 100% identical

Same repetition pattern as the 2000-token run, now with 3 more repeats: **5 `<seed2>...</seed2>` blocks,
all 5 byte-identical** (same 32 token ids every time). Doubling the budget didn't introduce any variation --
consistent with this being a genuine greedy fixed-point (from the same preceding context, argmax always
selects the same 32-token path), not something that resolves with more tokens.

## Model's own text output (unchanged from the 2000-token run)

- `<caption>`: "The person is cutting vegetables on a cutting board."
- `<speech>`: "1.5g of fresh vegetables."

## Decoded media

- seed2 -> image: `decoded_seed2_0.png` .. `decoded_seed2_4.png` (all 5 identical)
- cosmos -> video: `decoded_cosmos_chunk0.mp4` .. `decoded_cosmos_chunk7.mp4` (8 full 200-token chunks, all decoded successfully)
- snac -> audio: `decoded_snac.wav`
- agent -> pose: not applicable, zero tokens generated

## What this suggests for next steps

- **Sampling** (not greedy) is the more informative next test for reaching agent -- REPORT.md #31 already
  showed sampling escapes greedy's repetition loops at the token level; this result suggests it may also be
  needed to escape a modality-level loop, not just a token-level one.
- A prompt more textually similar to real FineVideo agent-bearing training examples (e.g. describing a
  specific body motion like the existing `agent_continuation`/`agent_from_scratch` tests, rather than a
  generic scene description) would be a fairer test of whether the model *can* reach agent unaided, versus
  whether *this particular* prompt's greedy path happens to avoid it.
