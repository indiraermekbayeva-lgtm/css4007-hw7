import csv
import json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "outputs" / "fixed-policy"
OUTPUT_DIR = ROOT / "outputs" / "fixed-policy-evaluation"
SEEDS = (11, 22, 33)
CHECKPOINTS = (100, 1_000, 10_000)
GAMMA = 1.0
ALPHA = 0.1
TOLERANCE = 0.000001
MAX_SWEEPS = 10_000


def load_games(seed):
    # Загружаем истории игр в порядке строк исходного CSV.
    path = INPUT_DIR / f"seed-{seed}" / "games.csv"
    games = []
    previous_game = 0
    with path.open(newline="", encoding="utf-8") as file:
        for game_id, rows in groupby(csv.DictReader(file), key=lambda row: int(row["game"])):
            game = []
            terminal = False
            if game_id <= previous_game:
                raise ValueError(f"{path}: game {game_id} is out of order")
            for step, row in enumerate(rows, 1):
                state = tuple(json.loads(row["observation"]))
                action = int(row["action"])
                next_state = (
                    None if row["next_observation"] == "END"
                    else tuple(json.loads(row["next_observation"]))
                )
                # Проверяем порядок шагов, политику и границы игры.
                if (
                    int(row["seed"]) != seed or int(row["step"]) != step
                    or terminal or (game and game[-1][3] != state)
                    or action != int(state[0] < 20)
                    or row["terminated"] not in ("true", "false")
                    or (row["terminated"] == "true") != (next_state is None)
                ):
                    raise ValueError(f"{path}: invalid transition in game {game_id}, step {step}")
                terminal = next_state is None
                game.append((state, action, float(row["reward"]), next_state))
            if not terminal:
                raise ValueError(f"{path}: game {game_id} is incomplete")
            games.append(game)
            previous_game = game_id
    if len(games) != CHECKPOINTS[-1]:
        raise ValueError(f"{path}: expected {CHECKPOINTS[-1]} complete games, found {len(games)}")
    return games


def evaluate_model(pair_visits, outcome_counts, reward_sums):
    # Строим модель по наблюдениям и оцениваем фиксированную стратегию.
    model = defaultdict(list)
    for (state, action, next_state), count in outcome_counts.items():
        # Вычисляем вероятность перехода и среднюю награду.
        pair = state, action
        probability = count / pair_visits[pair]
        mean_reward = reward_sums[state, action, next_state] / count
        model[pair].append((next_state, count, probability, mean_reward))

    successors = {}
    for (state, action), outcomes in model.items():
        if action == int(state[0] < 20):
            successors[state] = outcomes

    # Отмечаем состояния, из которых можно попасть к неизвестному переходу.
    affected = {
        state for state, outcomes in successors.items()
        if any(next_state is not None and next_state not in successors
               for next_state, _, _, _ in outcomes)
    }
    while True:
        upstream = {
            state for state, outcomes in successors.items()
            if any(next_state in affected for next_state, _, _, _ in outcomes)
        }
        if upstream <= affected:
            break
        affected.update(upstream)

    values = dict.fromkeys(successors, 0.0)
    for sweep in range(1, MAX_SWEEPS + 1):
        # Используем только значения предыдущего прохода.
        updated = {}
        for state, outcomes in successors.items():
            updated[state] = sum(
                probability * (reward + GAMMA * values.get(next_state, 0.0))
                for next_state, _, probability, reward in outcomes
            )
        delta = max((abs(updated[state] - values[state]) for state in values), default=0.0)
        values = updated
        if delta <= TOLERANCE:
            break
    return model, values, affected, sweep, delta <= TOLERANCE, delta


def save_checkpoint(directory, states, estimates, pair_visits, model):
    # Сохраняем таблицы значений и наблюдения для каждой пары состояния и действия.
    directory.mkdir(parents=True, exist_ok=True)
    for ace, filename in ((0, "no_usable_ace.csv"), (1, "usable_ace.csv")):
        with (directory / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow((
                "player_sum", "dealer_showing", "visits", "model_value",
                "model_status", "mc_value", "td_value",
            ))
            for state in states:
                if state[2] == ace:
                    writer.writerow((*state[:2], *estimates[state]))

    with (directory / "transitions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "player_sum", "dealer_showing", "usable_ace", "action", "pair_visits",
            "pair_status", "next_observation", "outcome_count", "probability", "mean_reward",
        ))
        for state in states:
            for action in (0, 1):
                pair = state, action
                if pair not in model:
                    writer.writerow((*state, action, 0, "unobserved", "", "", "", ""))
                    continue
                for next_state, count, probability, reward in model[pair]:
                    label = "END" if next_state is None else json.dumps(next_state)
                    writer.writerow((
                        *state, action, pair_visits[pair], "observed",
                        label, count, probability, reward,
                    ))


def evaluate_seed(seed, games, states, summaries, comparison):
    # Обрабатываем игры по порядку и сохраняем результаты на контрольных точках.
    visits = Counter()
    pair_visits = Counter()
    outcome_counts = Counter()
    reward_sums = defaultdict(float)
    return_sums = defaultdict(float)
    td_values = defaultdict(float)
    snapshots = {}

    for game_number, game in enumerate(games, 1):
        # Считаем данные для модели и сразу обновляем значения TD.
        for state, action, reward, next_state in game:
            visits[state] += 1
            pair_visits[state, action] += 1
            outcome = state, action, next_state
            outcome_counts[outcome] += 1
            reward_sums[outcome] += reward
            continuation = 0.0 if next_state is None else td_values.get(next_state, 0.0)
            target = reward + GAMMA * continuation
            td_values[state] += ALPHA * (target - td_values[state])

        # Считаем полный возврат для каждого посещения, начиная с конца игры.
        visit_return = 0.0
        for state, _, reward, _ in reversed(game):
            visit_return = reward + GAMMA * visit_return
            return_sums[state] += visit_return

        if game_number not in CHECKPOINTS:
            continue

        model, model_values, affected, sweeps, converged, delta = evaluate_model(
            pair_visits, outcome_counts, reward_sums
        )
        estimates = {}
        for state in states:
            count = visits[state]
            if not count:
                estimates[state] = (0, "", "unobserved", "", "")
                continue
            if state not in model_values:
                status = "missing_policy_action"
            elif state in affected:
                status = "partial_missing_successor"
            else:
                status = "estimated"
            estimates[state] = (
                count, model_values.get(state, ""), status, return_sums[state] / count, td_values[state],
            )
        snapshots[game_number] = estimates
        directory = OUTPUT_DIR / f"seed-{seed}" / f"games-{game_number}"
        save_checkpoint(directory, states, estimates, pair_visits, model)
        summaries.writerow((
            seed, game_number, sum(visits.values()), len(visits), len(states) - len(visits),
            len(pair_visits), 2 * len(states) - len(pair_visits),
            len(affected), sweeps, str(converged).lower(), delta,
        ))
        status = "converged" if converged else "did not converge"
        print(f"Seed {seed}, {game_number:>5} games: {len(visits)} observed states, "
              f"{len(affected)} partial model estimates, {sweeps} sweeps ({status})")

    # Сравниваем одни и те же частые и редкие состояния на всех точках.
    by_frequency = sorted(visits, key=lambda state: (-visits[state], state))
    frequent = by_frequency[:3]
    rare = sorted(by_frequency[3:], key=lambda state: (visits[state], state))[:3]
    for category, examples in (("frequent", frequent), ("rare", rare)):
        for state in examples:
            for checkpoint in CHECKPOINTS:
                comparison.writerow((
                    seed, category, checkpoint, *state, *snapshots[checkpoint][state],
                ))


def main():
    # Загружаем общие истории и запускаем оценку для каждого seed.
    all_games = {seed: load_games(seed) for seed in SEEDS}
    observed_sums = {
        state[0] for games in all_games.values()
        for game in games for state, _, _, _ in game
    }
    # Общая сетка нужна только для вывода, включая пустые ячейки.
    states = [
        (player_sum, dealer, ace)
        for player_sum in range(min(observed_sums), max(observed_sums) + 1)
        for dealer in range(1, 11)
        for ace in (0, 1)
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (
        (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file,
        (OUTPUT_DIR / "comparison.csv").open("w", newline="", encoding="utf-8") as comparison_file,
    ):
        summaries = csv.writer(summary_file)
        summaries.writerow((
            "seed", "games", "transitions", "observed_states", "unobserved_states",
            "observed_pairs", "unobserved_pairs", "partial_model_estimates",
            "sweeps", "converged", "final_delta",
        ))
        comparison = csv.writer(comparison_file)
        comparison.writerow((
            "seed", "frequency", "games", "player_sum", "dealer_showing", "usable_ace",
            "visits", "model_value", "model_status", "mc_value", "td_value",
        ))
        for seed, games in all_games.items():
            evaluate_seed(seed, games, states, summaries, comparison)


if __name__ == "__main__":
    main()
