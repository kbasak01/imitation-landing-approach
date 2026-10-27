---
name: claim-trace
description: Rules for any text that states a result — README, protocol entries, figure captions, commit messages, PR descriptions. Every number must trace to a committed file; forbidden real-world wording; required qualifiers. Use whenever writing or reviewing such text.
---

# Claim tracing

## Every number traces
For each number in the text, name the committed file (and column/row) it comes from, in the text or
in a nearby "Sources:" line. If it cannot be traced, remove it or mark it as not committed. Copy
numbers from files; never retype them from memory. Rounding: as the source, or stated.

## Forbidden unless explicitly negated
real flight, flight-tested, deployed on a drone, real camera, real images, onboard camera (unqualified),
real ship/deck, at sea, sea trials, sim-to-real, transfers to hardware, guarantees, safe (as a
property), solves, state of the art, first (as a novelty claim), outperforms Angelis et al.,
real-time on Jetson (or any embedded target not measured).

## Required qualifiers
- "in simulation" or "simulated" on every headline sentence about landing.
- "rendered (PyBullet TinyRenderer)" when describing images.
- "model scale, λ = 1/25" for every length/time/speed outside a table that states it.
- "unpaired reading" for any comparison not tested by a paired bootstrap.
- "hand-picked" for GIFs and galleries.
- "on this machine (RTX A4000, WSL2)" for latency.
- Teacher named for every student ("`dp`, distilled from P8 `ppo` seed 0").

## Hypothesis language
Use the verdict words exactly: supported / not supported / inconclusive, with the number and CI.
A null is reported in the body, not a footnote.

## Self-check before finishing
grep the text for the forbidden list; list every number with its source; confirm each source file
exists in the commit you are describing.
