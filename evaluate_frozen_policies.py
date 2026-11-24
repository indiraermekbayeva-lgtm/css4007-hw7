import csv
import random
from collections import Counter
from pathlib import Path

import gymnasium as gym


ROOT = Path(__file__).resolve().parent
Q_TABLE_DIR = ROOT / "outputs" / "q-learning"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
SEED_PAIRS = ((11, 1011), (22, 1022), (33, 1033))
TRAINING_EPSILONS = (None, 0.2, 0.0)
GAMES_PER_POLICY = 10_000
ACTION_LIMIT = 100


def load_q_table(training_seed, epsilon):
    # Загружаем обученную Q-таблицу для обоих вариантов туза.
    directory = Q_TABLE_DIR / f"epsilon-{epsilon}" / f"seed-{training_seed}"
    q_values = {}
    for ace, filename in ((0, "no_usable_ace.csv"), (1, "usable_ace.csv")):
        with (directory / filename).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["state_status"] == "observed":
                    state = int(row["player_sum"]), int(row["dealer_showing"]), ace
                    q_values[state] = float(row["q_stick"]), float(row["q_hit"])
    return q_values


def evaluate_policy(env, training_seed, evaluation_seed, epsilon, q_values, cutoff_log):
    # Проверяем стратегию на новых играх и считаем результаты.
    policy = "baseline" if q_values is None else "q_learning"
    run = training_seed, policy, epsilon, evaluation_seed
    action_rng = random.Random(evaluation_seed)
    outcomes = Counter()
    cutoffs = 0
    unknown_states = Counter()

    for game in range(1, GAMES_PER_POLICY + 1):
        # Задаём seed только при первом сбросе новой среды.
        if game == 1:
            state, _ = env.reset(seed=evaluation_seed)
        else:
            state, _ = env.reset()

        for step in range(1, ACTION_LIMIT + 1):
            # Выбираем лучшее действие. Для неизвестного состояния оба Q равны нулю.
            if q_values is None:
                action = int(state[0] < 20)
            else:
                if state not in q_values:
                    unknown_states[state] += 1
                stick, hit = q_values.get(state, (0.0, 0.0))
                action = action_rng.choice((0, 1)) if stick == hit else int(hit > stick)
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        # Записываем прерывания отдельно от результатов завершённых игр.
        if terminated:
            outcomes[reward] += 1
        else:
            reason = "environment_truncated" if truncated else "action_limit"
            cutoff_log.writerow((*run, game, step, reason))
            cutoffs += 1

    wins, losses, draws = (outcomes[result] for result in (1, -1, 0))
    completed = wins + losses + draws
    mean_reward = (wins - losses) / completed if completed else None
    return (*run, wins, losses, draws, cutoffs, mean_reward,
            sum(unknown_states.values()), len(unknown_states))


def main():
    # Запускаем девять оценок и сохраняем сводку и журнал прерываний.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Gymnasium {gym.__version__} | Blackjack-v1 | natural=False | sab=False")
    print("Unknown Q states use two zero values and seeded random tie-breaking.")
    print("Unknown-state visits and distinct states are counted separately per run.")
    run_columns = ("training_seed", "policy", "training_epsilon", "evaluation_seed")
    with (
        (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as results_file,
        (OUTPUT_DIR / "cutoffs.csv").open("w", newline="", encoding="utf-8") as cutoffs_file,
    ):
        results = csv.writer(results_file)
        results.writerow((*run_columns, "wins", "losses", "draws", "cutoffs", "mean_reward",
                          "unknown_state_visits", "distinct_unknown_states"))
        cutoff_log = csv.writer(cutoffs_file)
        cutoff_log.writerow((*run_columns, "game", "actions", "reason"))
        for training_seed, evaluation_seed in SEED_PAIRS:
            for epsilon in TRAINING_EPSILONS:
                q_values = None if epsilon is None else load_q_table(training_seed, epsilon)
                env = gym.make("Blackjack-v1", natural=False, sab=False)
                try:
                    result = evaluate_policy(
                        env, training_seed, evaluation_seed, epsilon, q_values, cutoff_log
                    )
                finally:
                    env.close()
                results.writerow(result)
                _, policy, _, _, wins, losses, draws, cutoffs, mean, unknown, _ = result
                label = policy if epsilon is None else f"{policy} (training epsilon={epsilon})"
                mean_text = "N/A" if mean is None else f"{mean:.4f}"
                print(f"Seed {evaluation_seed}, {label}: {wins} wins, {losses} losses, "
                      f"{draws} draws, {cutoffs} cut-offs, mean reward {mean_text}, "
                      f"{unknown} unknown-state visits")


if __name__ == "__main__":
    main()
