# Corrections to the handwritten work

These notes were added after the paper replay. The
[original handwritten PDF](HW7_Part_A_and_B.pdf) is the source for the paper
results. The [replay report](../outputs/paper-replay/replay.txt) shows the
program results.

## Three game labels

| Paper location | Original label | Correct label | Effect |
| --- | --- | --- | --- |
| PDF page 3, model count for E / stick → END | Game 4 | Game 3, step 3 | The transition count is still 1. |
| PDF page 4, Monte Carlo note for D | Game 1 | Game 3, step 2 | The return is still 0. |
| PDF page 4, Monte Carlo note for E | Game 1 | Game 3, step 3 | The return is still 0. |

## Q-learning pass 1 — missing old value

At game 4, step 2, the old Q-value for **B / stick is 0.1**. It came from
game 1, step 2. There was no other update to B / stick before this step.
The reward is −1 and the next state is END, so the continuation is 0.
The target is `−1 + 1 × 0 = −1`. The new value is
`0.1 + 0.1 × (−1 − 0.1) = −0.01`, as shown on paper.

## Q-learning pass 2 — added calculations

The paper includes the final Q-table. The eight updates below fill in the
missing work. At the start of pass 2, Q(A, hit) is −0.08, Q(B, stick) is
−0.01, and all other Q-values are 0. Use discount 1 and learning rate 0.1.
For a normal next state, take the larger of its two Q-values. For END, use 0.

| Game / step | State / action → next | Old Q | Reward | Maximum next Q | Target = reward + 1 × maximum next Q | New Q = old Q + 0.1 × (target − old Q) |
| --- | --- | ---: | ---: | --- | --- | --- |
| 1 / 1 | A / hit → B | −0.08 | 0 | max(−0.01, 0) = 0 | 0 + 1 × 0 = 0 | −0.08 + 0.1 × (0 + 0.08) = −0.072 |
| 1 / 2 | B / stick → END | −0.01 | +1 | 0 | 1 + 1 × 0 = 1 | −0.01 + 0.1 × (1 + 0.01) = 0.091 |
| 2 / 1 | A / hit → END | −0.072 | −1 | 0 | −1 + 1 × 0 = −1 | −0.072 + 0.1 × (−1 + 0.072) = −0.1648 |
| 3 / 1 | C / hit → D | 0 | 0 | max(0, 0) = 0 | 0 + 1 × 0 = 0 | 0 + 0.1 × (0 − 0) = 0 |
| 3 / 2 | D / hit → E | 0 | 0 | max(0, 0) = 0 | 0 + 1 × 0 = 0 | 0 + 0.1 × (0 − 0) = 0 |
| 3 / 3 | E / stick → END | 0 | 0 | 0 | 0 + 1 × 0 = 0 | 0 + 0.1 × (0 − 0) = 0 |
| 4 / 1 | A / hit → B | −0.1648 | 0 | max(0.091, 0) = 0.091 | 0 + 1 × 0.091 = 0.091 | −0.1648 + 0.1 × (0.091 + 0.1648) = −0.13922 |
| 4 / 2 | B / stick → END | 0.091 | −1 | 0 | −1 + 1 × 0 = −1 | 0.091 + 0.1 × (−1 − 0.091) = −0.0181 |

## Rounding

The paper sometimes writes 2/3 as 0.667, 1/3 as 0.333, and −1/3 as
−0.333. The program prints more decimal places. The exact fractions agree.
The small differences in the displayed decimals come from rounding.
