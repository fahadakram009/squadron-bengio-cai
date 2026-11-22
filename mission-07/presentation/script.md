# Week 7 lightning talk: Branching beyond greedy decoding (Tree of Thoughts)

Speaker: Fahad Akram, Systems Architect, Squadron Bengio. Target length 3:30. Six slides, one live terminal.

**Before class.** Open a terminal in `mission-07`, font size large, and keep this ready:

```
python presentation/tot_demo.py --step
```

`--step` pauses after each act until you press Enter. Run it once beforehand so the first run is warm.

---

## 0:00 Cover (slide 1), 15 s

"I am Fahad, Systems Architect of Squadron Bengio. In three minutes: why a single chain of thought fails, a riddle where the best-looking move is a trap, and how two searches get out of it. Everything runs from our `tot_solver.py`."

## 0:15 The problem (slide 2), 35 s

"Errors compound. If every reasoning step is 95 percent reliable, a ten-step chain contains at least one fatal step 40 percent of the time. Greedy decoding cannot undo a step: one bad token poisons everything after it.

Tree of Thoughts changes the shape of the computation. Generate several candidate thoughts, score each with a verifier, keep the promising ones, and search under a fixed budget of expansions. One expansion is one thought generated and scored. Two ways to search: a beam of width B, or depth-first with backtracking."

## 0:50 The riddle (slide 3), 40 s

"Here is my riddle. The ship has to escape the accretion disk of Sagittarius A star in three burns. Three first moves. The verifier likes A, the full retrograde burn, best at 0.90. Below A, coasting through the jet scores 0.95, the best thought in the whole tree. And it leads only to dead ends: jet drag or a hull breach.

The goal is behind B, the slingshot, which scores only 0.80. A greedy pilot, which is what a one-shot model is, follows the scores straight into the trap."

## 1:30 Live demo (slide 4 on screen, terminal in front), 70 s

Switch to the terminal, press Enter through the acts.

Act 1, beam width 1: "Greedy keeps A, then A1, then everything under A1 is pruned. Stuck at depth 2 after seven expansions. Look at B: dropped from the beam at level one and never seen again. That is the failure of greedy decoding in one tree."

Act 2, beam width 2: "Width two keeps a second opinion per level. B survives beside A, B1 is kept, and the goal is reached in ten expansions."

Act 3, depth-first: "Depth-first dives into A, hits the wall, backtracks twice, climbs out and takes B. Same cost here, but it only pays for siblings when a branch dies."

Act 4, the table: leave it on screen and switch back to the slides.

## 2:40 Results (slide 5), 30 s

"Does it hold on a real problem? Game of 24, forty puzzles, a verifier with noise. The greedy beam is stuck at 20 percent whatever the budget. Width three reaches 47. Backtracking reaches 50 at the lowest cost.

And every method plateaus at the same ceiling. A correct step that the verifier prunes is lost to every algorithm. The verifier sets the limit, not the search. That is why Part 3 of our mission, the process reward model, matters as much as the search."

## 3:10 Graph of Thoughts and take-away (slide 6), 20 s

"Graph of Thoughts goes one step further. A thought may have several parents, so two partial solutions that each got half of it right are merged into one, and a thought can be refined in place rather than regenerated.

Take-away: search turns compute into alternatives. The verifier decides which alternative deserves the next expansion. Code is in our repository. Thank you."

---

## If asked

- **Why does DFS cost the same as beam width 2 on the riddle?** Both evaluate the same ten thoughts: DFS pays for A's subtree before backtracking. DFS wins when the tree is wide and the first branch is right, and when budget is scarce, because it never evaluates a whole level of siblings up front.
- **Why does beam width 5 do worse than width 1 at budget 48?** It spends the entire budget on the first level and never reaches a terminal. Width is only useful once the budget covers all levels.
- **Where does the 0.5 ceiling come from?** With noise 0.5 a true step scores below the 0.45 prune threshold about 20 percent of the time, and once pruned it is gone. All searches share that loss.
- **Is the verifier an LLM?** In our module it is a solvability oracle plus noise, so the noise level is controllable. A real PRM is Part 3 of the mission.
- **Budget enforcement?** `expansion_count` never exceeds `max_expansions`; a candidate beyond the budget is never evaluated. Checked for every budget from 0 to 11 in the self-test.
