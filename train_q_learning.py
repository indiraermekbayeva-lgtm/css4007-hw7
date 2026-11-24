import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import gymnasium as gym


SEEDS = (11, 22, 33)
GAMES_PER_SEED = 50_000
BLOCK_SIZE = 1_000
ACTION_LIMIT = 100
ALPHA = 0.1
GAMMA = 1.0
ACTIONS = (0, 1)
ACTION_NAMES = ("stick", "hit")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "q-learning"


def choose_action(values, epsilon, action_rng):
    # Выбираем случайное действие при exploration или при равных Q-значениях.
    if epsilon > 0 and action_rng.random() < epsilon:
        return action_rng.choice(ACTIONS)
    if values[0] == values[1]:
        return action_rng.choice(ACTIONS)
    return 0 if values[0] > values[1] else 1


def train_agent(env, seed, epsilon, blocks_writer, cutoffs_writer):
    # Обучаем агента с новой таблицей и отдельным генератором действий.
    action_rng = random.Random(seed)
    q_values = defaultdict(lambda: [0.0, 0.0])
    visits = Counter()
    outcomes = Counter()
    block_outcomes = Counter()
    attempts = completed = cutoffs = block_cutoffs = 0

    while completed < GAMES_PER_SEED:
        attempts += 1
        # Задаём seed среды только в начале запуска.
        if attempts == 1:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()

        for step in range(1, ACTION_LIMIT + 1):
            action = choose_action(q_values[state], epsilon, action_rng)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Обновляем Q-значение сразу после шага. Для END продолжение равно нулю.
            continuation = 0.0 if terminated else max(q_values[next_state])
            old_value = q_values[state][action]
            target = reward + GAMMA * continuation
            q_values[state][action] = old_value + ALPHA * (target - old_value)
            visits[state, action] += 1
            state = next_state
            if terminated or truncated:
                break

        # Прерванные игры учитываем отдельно от побед, поражений и ничьих.
        if not terminated:
            reason = "environment_truncated" if truncated else "action_limit"
            cutoffs_writer.writerow((seed, epsilon, attempts, step, reason))
            cutoffs += 1
            block_cutoffs += 1
            continue

        completed += 1
        outcomes[reward] += 1
        block_outcomes[reward] += 1
        if completed % BLOCK_SIZE == 0:
            wins, losses, draws = (block_outcomes[result] for result in (1, -1, 0))
            blocks_writer.writerow((
                seed, epsilon, completed // BLOCK_SIZE, completed - BLOCK_SIZE + 1, completed,
                BLOCK_SIZE, wins, losses, draws, block_cutoffs, (wins - losses) / BLOCK_SIZE,
            ))
            block_outcomes.clear()
            block_cutoffs = 0

    return q_values, visits, outcomes, completed, cutoffs


def save_tables(directory, seed, q_values, visits):
    # Сохраняем Q-значения, посещения и жадные действия отдельно по флагу туза.
    policy_rng = random.Random(seed)
    for ace, filename in ((0, "no_usable_ace.csv"), (1, "usable_ace.csv")):
        with (directory / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow((
                "player_sum", "dealer_showing", "state_status",
                "q_stick", "stick_visits", "stick_status",
                "q_hit", "hit_visits", "hit_status", "greedy_action", "greedy_tie",
            ))

            # Показываем всю сетку сумм, включая ненаблюдавшиеся состояния.
            for player_sum in range(4, 22):
                for dealer in range(1, 11):
                    state = player_sum, dealer, ace
                    stick_visits, hit_visits = (visits[state, action] for action in ACTIONS)
                    seen = stick_visits + hit_visits > 0
                    values = q_values.get(state, (0.0, 0.0))
                    greedy = ACTION_NAMES[choose_action(values, 0.0, policy_rng)] if seen else ""
                    tie = str(values[0] == values[1]).lower() if seen else ""
                    writer.writerow((
                        player_sum, dealer, "observed" if seen else "unobserved",
                        values[0], stick_visits, "observed" if stick_visits else "unobserved",
                        values[1], hit_visits, "observed" if hit_visits else "unobserved",
                        greedy, tie,
                    ))


def main():
    # Читаем epsilon, запускаем обучение для каждого seed и сохраняем сводку.
    parser = argparse.ArgumentParser(description="Train Q-learning on Gymnasium Blackjack.")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Exploration probability (0 to 1).")
    args = parser.parse_args()
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must be between 0 and 1")
    epsilon = args.epsilon or 0.0
    run_dir = OUTPUT_DIR / f"epsilon-{epsilon}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Gymnasium {gym.__version__} | Blackjack-v1 | natural=False | sab=False")
    print(f"Epsilon = {epsilon}, alpha = {ALPHA}, gamma = {GAMMA}")

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        summary = csv.writer(file)
        summary.writerow((
            "seed", "epsilon", "completed_games", "wins", "losses", "draws",
            "cutoffs", "mean_reward", "updates", "observed_states", "observed_pairs",
        ))
        for seed in SEEDS:
            directory = run_dir / f"seed-{seed}"
            directory.mkdir(exist_ok=True)
            env = gym.make("Blackjack-v1", natural=False, sab=False)
            try:
                with (
                    (directory / "training_rewards.csv").open("w", newline="", encoding="utf-8") as blocks_file,
                    (directory / "cutoffs.csv").open("w", newline="", encoding="utf-8") as cutoffs_file,
                ):
                    blocks = csv.writer(blocks_file)
                    cutoffs_log = csv.writer(cutoffs_file)
                    blocks.writerow((
                        "seed", "epsilon", "block", "first_completed_game", "last_completed_game",
                        "completed_games", "wins", "losses", "draws", "cutoffs", "mean_reward",
                    ))
                    cutoffs_log.writerow(("seed", "epsilon", "game_attempt", "actions", "reason"))
                    q_values, visits, outcomes, completed, cutoffs = train_agent(
                        env, seed, epsilon, blocks, cutoffs_log
                    )
            finally:
                env.close()

            save_tables(directory, seed, q_values, visits)
            wins, losses, draws = (outcomes[result] for result in (1, -1, 0))
            mean_reward = (wins - losses) / completed
            updates = sum(visits.values())
            observed_states = len({state for state, action in visits})
            summary.writerow((
                seed, epsilon, completed, wins, losses, draws, cutoffs, mean_reward,
                updates, observed_states, len(visits),
            ))
            print(f"Seed {seed}: {completed} games, {wins} wins, {losses} losses, "
                  f"{draws} draws, {cutoffs} cut-offs, mean reward {mean_reward:.5f}, "
                  f"{updates} updates, {observed_states} observed states, {len(visits)} observed pairs")


if __name__ == "__main__":
    main()
