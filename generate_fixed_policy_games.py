import csv
import json
from pathlib import Path

import gymnasium as gym


SEEDS = (11, 22, 33)
GAMES_PER_SEED = 10_000
ACTION_LIMIT = 100
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "fixed-policy"
TRANSITION_COLUMNS = (
    "seed", "game", "step", "observation", "action", "reward",
    "next_observation", "terminated",
)


def generate_games(env, seed, games_writer, cutoffs_writer):
    # Генерируем игры для одного seed и отдельно записываем прерванные попытки.
    game = 0
    completed = 0
    cutoff_count = 0
    transition_count = 0
    outcomes = {1: 0, -1: 0, 0: 0}

    while completed < GAMES_PER_SEED:
        game += 1
        # Задаём seed только при первом сбросе среды.
        if game == 1:
            observation, _ = env.reset(seed=seed)
        else:
            observation, _ = env.reset()
        transitions = []

        for step in range(1, ACTION_LIMIT + 1):
            player_sum = observation[0]
            action = 1 if player_sum < 20 else 0
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_label = "END" if terminated else json.dumps(next_observation)
            transitions.append((
                seed, game, step, json.dumps(observation), action, reward,
                next_label, str(terminated).lower(),
            ))
            observation = next_observation
            if terminated or truncated:
                break

        # Сохраняем только полные игры в общем наборе историй.
        if terminated:
            games_writer.writerows(transitions)
            completed += 1
            transition_count += len(transitions)
            outcomes[reward] += 1
        else:
            reason = "environment_truncated" if truncated else "action_limit"
            cutoffs_writer.writerows(row + (reason,) for row in transitions)
            cutoff_count += 1

    wins = outcomes[1]
    losses = outcomes[-1]
    draws = outcomes[0]
    mean_reward = (wins - losses) / completed
    return seed, completed, transition_count, wins, losses, draws, cutoff_count, mean_reward


def main():
    # Создаём файлы для каждого seed и сохраняем общую статистику.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Gymnasium {gym.__version__} | Blackjack-v1 | natural=False | sab=False")
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        summary = csv.writer(summary_file)
        summary.writerow((
            "seed", "completed_games", "transitions", "wins", "losses",
            "draws", "cutoffs", "mean_reward",
        ))
        for seed in SEEDS:
            seed_dir = OUTPUT_DIR / f"seed-{seed}"
            seed_dir.mkdir(exist_ok=True)
            env = gym.make("Blackjack-v1", natural=False, sab=False)
            try:
                with (
                    (seed_dir / "games.csv").open("w", newline="", encoding="utf-8") as games_file,
                    (seed_dir / "cutoffs.csv").open("w", newline="", encoding="utf-8") as cutoffs_file,
                ):
                    games_writer = csv.writer(games_file)
                    cutoffs_writer = csv.writer(cutoffs_file)
                    games_writer.writerow(TRANSITION_COLUMNS)
                    cutoffs_writer.writerow(TRANSITION_COLUMNS + ("cutoff_reason",))
                    result = generate_games(env, seed, games_writer, cutoffs_writer)
            finally:
                env.close()

            summary.writerow(result)
            _, completed, transitions, wins, losses, draws, cutoffs, mean_reward = result
            print(
                f"Seed {seed}: {completed} games, {transitions} transitions, "
                f"{wins} wins, {losses} losses, {draws} draws, "
                f"{cutoffs} cut-offs, mean reward {mean_reward:.4f}"
            )


if __name__ == "__main__":
    main()
