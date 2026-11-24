# HW7 submission — Blackjack

## AI-use declaration

I used ChatGPT to understand the assignment, explain the algorithms, check
my handwritten calculations, and prepare prompts. I used Codex to implement
the algorithms, test the code, compare results, and write the documentation.

## Setup and reproduction

Tested with Python 3.12.0, Gymnasium 1.4.0, and Matplotlib 3.11.2.
Run these commands in order from the project folder.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-fixed-policy.txt -r requirements-plots.txt
python3 replay_paper.py
python3 generate_fixed_policy_games.py
python3 evaluate_fixed_policy.py
python3 train_q_learning.py --epsilon 0.2
python3 train_q_learning.py --epsilon 0
python3 evaluate_frozen_policies.py
python3 plot_results.py
```

The replay reads the two supplied CSVs. The fixed-policy methods then use
the same newly generated game logs. The frozen evaluation loads the saved
Q-tables, and the plot script reads the saved results.
[Simulation dependencies](requirements-fixed-policy.txt) ·
[Plot dependencies](requirements-plots.txt)

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

All three methods read the same 10,000 games per seed in the same order.
The policy hits below 20 and sticks at 20 or 21. Each value CSV includes
visit counts and all three estimates. The model CSVs include transition
counts, probabilities, mean rewards, and unobserved state-action pairs.

| Seed | Shared game log | Games | No usable ace | Usable ace | Model and pair counts |
| --- | --- | ---: | --- | --- | --- |
| 11 | [Games](outputs/fixed-policy/seed-11/games.csv) | 100 | [Values and visits](outputs/fixed-policy-evaluation/seed-11/games-100/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-11/games-100/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-11/games-100/transitions.csv) |
| 11 | — | 1,000 | [Values and visits](outputs/fixed-policy-evaluation/seed-11/games-1000/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-11/games-1000/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-11/games-1000/transitions.csv) |
| 11 | — | 10,000 | [Values and visits](outputs/fixed-policy-evaluation/seed-11/games-10000/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-11/games-10000/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-11/games-10000/transitions.csv) |
| 22 | [Games](outputs/fixed-policy/seed-22/games.csv) | 100 | [Values and visits](outputs/fixed-policy-evaluation/seed-22/games-100/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-22/games-100/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-22/games-100/transitions.csv) |
| 22 | — | 1,000 | [Values and visits](outputs/fixed-policy-evaluation/seed-22/games-1000/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-22/games-1000/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-22/games-1000/transitions.csv) |
| 22 | — | 10,000 | [Values and visits](outputs/fixed-policy-evaluation/seed-22/games-10000/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-22/games-10000/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-22/games-10000/transitions.csv) |
| 33 | [Games](outputs/fixed-policy/seed-33/games.csv) | 100 | [Values and visits](outputs/fixed-policy-evaluation/seed-33/games-100/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-33/games-100/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-33/games-100/transitions.csv) |
| 33 | — | 1,000 | [Values and visits](outputs/fixed-policy-evaluation/seed-33/games-1000/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-33/games-1000/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-33/games-1000/transitions.csv) |
| 33 | — | 10,000 | [Values and visits](outputs/fixed-policy-evaluation/seed-33/games-10000/no_usable_ace.csv) | [Values and visits](outputs/fixed-policy-evaluation/seed-33/games-10000/usable_ace.csv) | [Transitions](outputs/fixed-policy-evaluation/seed-33/games-10000/transitions.csv) |

[Game log summary](outputs/fixed-policy/summary.csv) ·
[Coverage and convergence](outputs/fixed-policy-evaluation/summary.csv) ·
[Comparison across seeds and checkpoints](outputs/fixed-policy-evaluation/comparison.csv)

States with zero visits have blank estimates. Unseen action pairs have
no estimated transition probabilities or rewards. If the model has no
data for a next state, it uses zero as a temporary continuation and marks
any estimates that depend on that gap. No estimates needed that mark in
these runs. The model used only the fixed policy's action and updated all
states from the previous sweep. All nine runs converged in 5–12 sweeps,
with a tolerance of 0.000001 and a limit of 10,000 sweeps.

For seed 11, hard 20 against dealer 10 had 6, 60, and 606 visits at the
three checkpoints. The model and Monte Carlo agreed at each point (0.333,
0.467, and 0.416) because sticking ends the game. TD(0) gave 0.173, 0.511,
and 0.439 because it moves its estimate part of the way after each result.
Hard 4 against dealer 6 was unseen at 100 games and had only one visit by
10,000. Its final estimates were −0.693 (model), −1 (Monte Carlo), and 0
(TD). Monte Carlo used the full loss from that game. At the time of the
TD update, the next state's value was still zero. The model used later
data about that next state.

### Q-learning and baseline evaluation

Each row reports 10,000 new games in a fresh environment. The seed pairs
are the same for all three policies. Different actions can change how
many cards are drawn. The learned policies use the saved Q-values without
updates. A seeded random choice breaks ties. A state missing from the
Q-table would use two zero values and the same tie rule. No such visits
or cut-offs occurred.

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | Fixed-policy baseline | 1011 | 2962 | 6462 | 576 | 0 | -0.3500 |
| 11 | Q-learning / 0.2 | 1011 | 4228 | 4882 | 890 | 0 | -0.0654 |
| 11 | Q-learning / 0 | 1011 | 4100 | 4929 | 971 | 0 | -0.0829 |
| 22 | Fixed-policy baseline | 1022 | 2935 | 6481 | 584 | 0 | -0.3546 |
| 22 | Q-learning / 0.2 | 1022 | 4220 | 4919 | 861 | 0 | -0.0699 |
| 22 | Q-learning / 0 | 1022 | 4100 | 4975 | 925 | 0 | -0.0875 |
| 33 | Fixed-policy baseline | 1033 | 2950 | 6496 | 554 | 0 | -0.3546 |
| 33 | Q-learning / 0.2 | 1033 | 4124 | 5025 | 851 | 0 | -0.0901 |
| 33 | Q-learning / 0 | 1033 | 4098 | 4943 | 959 | 0 | -0.0845 |

[Evaluation results](outputs/evaluation/summary.csv) ·
[Evaluation cut-offs](outputs/evaluation/cutoffs.csv)

### Plots and learned policies

- [Training curves](outputs/plots/training_curves.png) show all 50 blocks of 1,000 games for each run.
- [Frozen-policy evaluation](outputs/plots/frozen_policy_evaluation.png) shows the mean rewards on new games.
- [Greedy actions with a usable ace](outputs/plots/greedy_policy_usable_ace.png)
- [Greedy actions with no usable ace](outputs/plots/greedy_policy_no_usable_ace.png)

Each policy CSV has both Q-values, counts for each action, and the greedy
action. It also marks ties and unvisited states or actions.

| Training seed | Epsilon | Training rewards | No usable ace | Usable ace |
| --- | --- | --- | --- | --- |
| 11 | 0.2 | [1,000-game blocks](outputs/q-learning/epsilon-0.2/seed-11/training_rewards.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.2/seed-11/no_usable_ace.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.2/seed-11/usable_ace.csv) |
| 22 | 0.2 | [1,000-game blocks](outputs/q-learning/epsilon-0.2/seed-22/training_rewards.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.2/seed-22/no_usable_ace.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.2/seed-22/usable_ace.csv) |
| 33 | 0.2 | [1,000-game blocks](outputs/q-learning/epsilon-0.2/seed-33/training_rewards.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.2/seed-33/no_usable_ace.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.2/seed-33/usable_ace.csv) |
| 11 | 0 | [1,000-game blocks](outputs/q-learning/epsilon-0.0/seed-11/training_rewards.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.0/seed-11/no_usable_ace.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.0/seed-11/usable_ace.csv) |
| 22 | 0 | [1,000-game blocks](outputs/q-learning/epsilon-0.0/seed-22/training_rewards.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.0/seed-22/no_usable_ace.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.0/seed-22/usable_ace.csv) |
| 33 | 0 | [1,000-game blocks](outputs/q-learning/epsilon-0.0/seed-33/training_rewards.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.0/seed-33/no_usable_ace.csv) | [Q, policy, and counts](outputs/q-learning/epsilon-0.0/seed-33/usable_ace.csv) |

[Training summary for epsilon 0.2](outputs/q-learning/epsilon-0.2/summary.csv) ·
[Training summary for epsilon 0](outputs/q-learning/epsilon-0.0/summary.csv)

### Explain your results

- **Prediction and control.** Model-based learning, Monte Carlo, and TD(0)
  estimate the value of the fixed policy. The model learns transition
  probabilities and mean rewards. Monte Carlo averages complete returns
  from every visit. TD(0) updates after each step using the next state's
  current value. Q-learning learns a value for each action and uses the
  best next action value in its update. It can then choose a policy.
- **End of a game and usable aces.** Blackjack gives +1 for a win, −1 for
  a loss, and 0 for a draw. A bust ends the game, and sticking lets the
  dealer finish. A terminal transition has no future value, so its target
  is just the reward. A usable ace can change from 11 to 1 after a hit.
  In both seed 11 policies, the agent sticks on hard 12 against dealer 5
  and hits on soft 12 against the same card.
- **Exploration and coverage.** Choosing a random move sometimes helps
  test both actions. Choosing the higher Q-value uses what the agent
  has learned. Both settings visited the same 280 states for each seed.
  Epsilon 0.2 visited 560 state-action pairs per seed. Epsilon 0 visited
  538, 537, and 545 pairs for seeds 11, 22, and 33. It still used random
  tie-breaking. No cut-offs occurred in game collection, training, or
  evaluation. The safety limit was 100 player actions.
- **Results across seeds.** Both learned policies beat the baseline on
  all three evaluation seeds. Training with epsilon 0.2 gave the higher
  frozen score on 1011 and 1022. On 1033, epsilon 0 scored higher
  (−0.0845 compared with −0.0901). Epsilon 0 had higher mean training
  rewards for all three seeds, when the epsilon 0.2 agent was still
  exploring. All frozen mean rewards were negative. These tests show
  improvement over the baseline for the three runs. More runs could give
  different results.
