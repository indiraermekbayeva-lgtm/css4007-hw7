# Part D results

## Fixed-policy value estimates

Model-based learning builds a model from transition counts and mean rewards.
Monte Carlo averages complete returns from each visit. TD(0) updates after
each step and uses the current value of the next state. All three estimate
the same fixed policy. Q-learning instead estimates action values and uses
them to choose a policy. At a terminal transition the target is just the
reward, since no more rewards follow.

For seed 11, the hard 20 against dealer 10 was visited 6, 60, and 606 times
at the three checkpoints. The model and Monte Carlo agreed at each point
(0.333, 0.467, and 0.416), since sticking ends the game. TD(0) gave 0.173,
0.511, and 0.439. Hard 4 against dealer 6 was unseen at 100 games and had
only one visit by 10,000 games. Its final estimates were −0.693, −1, and 0
for the model, Monte Carlo, and TD(0). One visit gives little evidence.
The model can still change as it learns more about the next state. All
nine model runs converged in 5–12 sweeps. [Fixed-policy comparison](../fixed-policy-evaluation/comparison.csv)
· [Fixed-policy summary](../fixed-policy-evaluation/summary.csv)

## Training and coverage

All runs visited the same 280 states. With epsilon 0.2, all 560 corresponding
state-action pairs were visited for each seed. With epsilon 0, the counts
were 538, 537, and 545. Random exploration tests both actions more often.
Epsilon 0 still breaks ties randomly. The last training block scored
higher than the first for all six runs, though scores moved up and down
between blocks. Training with epsilon 0 gave a higher mean training reward
for all three seeds. The frozen tests measure performance with exploration
turned off for both learned policies.

The usable ace can change from 11 to 1 after a hit. In both seed 11 tables,
the greedy policy sticks on hard 12 against dealer 5 and hits on soft 12
against dealer 5. The policy maps show this difference and mark unvisited
cells and actions. None of the observed states had tied final Q-values.
[Training curves](training_curves.png) · [No usable ace](greedy_policy_no_usable_ace.png)
· [Usable ace](greedy_policy_usable_ace.png)

## Frozen-policy evaluation

| Evaluation seed | Baseline | Trained with epsilon 0.2 | Trained with epsilon 0 | Gain over baseline, epsilon 0.2 / 0 |
| ---: | ---: | ---: | ---: | ---: |
| 1011 | −0.3500 | −0.0654 | −0.0829 | +0.2846 / +0.2671 |
| 1022 | −0.3546 | −0.0699 | −0.0875 | +0.2847 / +0.2671 |
| 1033 | −0.3546 | −0.0901 | −0.0845 | +0.2645 / +0.2701 |

Both learned policies beat the baseline for each seed. Training with
epsilon 0.2 gave the higher frozen score on 1011 and 1022. On 1033,
epsilon 0 scored higher by 0.0056. All mean rewards remained negative.
There were no evaluation cut-offs or visits to unknown states. These
results show variation across seeds, and finite training gives no guarantee
of improvement in a new run. [Evaluation CSV](../evaluation/summary.csv)
· [Evaluation chart](frozen_policy_evaluation.png)
