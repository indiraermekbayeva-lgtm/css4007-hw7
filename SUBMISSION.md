# HW7 submission — Blackjack

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

Record dependency versions, installation steps and exact commands for the
paper replay, training and frozen-policy evaluation.

## Part A — Count states and actions

See the [original handwritten answers](paper-work/HW7_Part_A_and_B.pdf).

| Game | Representation | State-count formula | Type of count | Action representation | Legal-action count in the example | Hidden information |
| --- | --- | --- | --- | --- | --- | --- |
| Blackjack | Player sum 12–21, dealer card 1–10, usable ace yes/no. The full state also tracks both hands, the turn, and the end of play. | 10 × 10 × 2 = 200 | Exact for the restricted observation table | 0 = stick, 1 = hit | 2 at (18, 6, no), 0 at END | Dealer's hidden card |
| Durak | Two players, 36 cards (6 through ace), no transfers. The full state tracks both hands, attack and defense pairs, trump, stock order, discards, phase, and turn. | C(36, 6) = 1,947,792; C(29, 6) = 475,020 | Exact unordered hand counts, each only part of a full state | Play one named card, pick up, or end attack | 6 at opening, 4 in defense | Opponent's hand and hidden stock order |
| Tic-tac-toe | 3×3 board. X starts, players take turns, and a win or full board ends play. | 3⁹ = 19,683 | Exact board encoding count and upper bound on legal boards | Place a mark in one of 9 squares | 5 on the given board | None |
| Nim | Three labeled piles start at (2, 3, 4), with player to move. Taking the last stone wins. | (2 + 1) × (3 + 1) × (4 + 1) = 60; 60 × 2 = 120 | Exact encoding counts within the limits | Choose a pile and a positive number to remove | 9 at (2, 3, 4), 3 at (0, 1, 2), 0 at (0, 0, 0) | None |

- **Blackjack.** The table leaves out lower sums and END and has 200 × 2 = 400 Q-entries. The hands 10 + 8 and 9 + 9 give the same observation when the dealer shows 6.
- **Durak.** For the opponent's hand, set aside your six cards and the exposed trump, and ignore who attacked first. With hearts as trumps, reply to 10♣ with A♣, 7♥, Q♥, or pick up.
- **Tic-tac-toe.** Equal counts mean X moves next, and one extra X means O moves next while play continues. On the given board, X can play at (2,1), (2,3), (3,1), (3,2), or (3,3).
- **Nim.** At the start, remove 1–2 from pile 1, 1–3 from pile 2, or 1–4 from pile 3. At (0, 1, 2), the moves are (2,1), (3,1), and (3,2), in (pile, amount) order.

Durak needs a large Q-table because even one hand has nearly two million
possibilities. Some action labels are illegal in a given position, such as
placing a mark in an occupied square.

## Part B — Paper calculations

- [Original handwritten PDF](paper-work/HW7_Part_A_and_B.pdf)
- [Corrections](paper-work/corrections.md)

## Part C — Reproduce the calculations

Run this command from the project folder.

```sh
python3 replay_paper.py
```

The program replays the same four games as the paper. The
[replay report](outputs/paper-replay/replay.txt) shows the calculations and
final tables. The [comparison CSV](outputs/paper-replay/comparison.csv) gives
the program result minus the paper result for each number we can compare.
It uses the [original handwritten work](paper-work/HW7_Part_A_and_B.pdf).

- **Model-based learning.** The counts, probabilities, mean rewards, and
  both sweeps agree. Small differences in the decimal rows come from rounding.
  The note for sticking at E says game 4, but it happens in game 3.
- **Monte Carlo.** All returns, visit counts, and averages agree. The notes
  for D and E say game 1, but both visits are in game 3. This does not change
  the averages.
- **TD(0).** All 16 updates and both pass tables agree.
- **Q-learning.** The results shown for pass 1 and the final Q-table agree.
  The paper does not show the eight updates from pass 2. The CSV shows the
  program values and leaves the difference blank where a paper value is missing.

Selected results from the detailed CSV.

| Method | State / action | Sweep / pass / update | Paper result | Program result | Difference | Explanation |
| --- | --- | --- | --- | --- | --- | --- |
| Model-based learning | A | Sweep 2 value (exact) | -1/3 | -1/3 | 0 | The replay model uses the same fractions. The report displays a decimal. |
| Model-based learning | A / hit -> B | Transition probability (as displayed) | 0.667 | 0.6666666667 | -0.0003333333 | The paper also gives the matching exact fraction. This difference is from rounding. |
| Every-visit Monte Carlo | A | Pass 1 / mean return (exact) | -1/3 | -1/3 | 0 | Match. |
| TD(0) | A | Pass 2 / final value table | -0.14003 | -0.14003 | 0 | Match. |
| Q-learning | A / hit | Pass 2 / final Q-table | -0.13922 | -0.13922 | 0 | Match. |
| Q-learning | A / hit -> B | Pass 2 / game 1 / step 1 / New value | Not shown | -0.072 |  | The PDF gives the final Q-table but does not show the pass 2 updates. |

## Part D — Simulated games

### Fixed-policy value estimates

Link the shared logs, checkpoint value tables and visit counts. Explain
missing states and differences among model-based, Monte Carlo and TD estimates.

### Q-learning and baseline evaluation

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Plots and learned policies

Link training curves, Q-tables, hit/stick policy tables and state-action counts.
Separate usable-ace and no-usable-ace results.

### Explain your results

Address prediction versus control, terminal updates, usable aces, exploration,
coverage, variation across seeds and whether the learned policy improved.
