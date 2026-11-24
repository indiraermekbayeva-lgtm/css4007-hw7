import csv
from pathlib import Path


GAMMA = 1.0
ALPHA = 0.1
ACTIONS = {0: "stick", 1: "hit"}
ROOT = Path(__file__).resolve().parent


def load_data():
    # Загружаем состояния и группируем переходы по играм.
    with (ROOT / "data/states.csv").open(newline="", encoding="utf-8") as file:
        states = {}
        for row in csv.DictReader(file):
            states[row["label"]] = (
                int(row["player_sum"]),
                int(row["dealer_showing"]),
                bool(int(row["usable_ace"])),
            )

    games = {}
    with (ROOT / "data/paper-games.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            game = int(row["game"])
            transition = (
                int(row["step"]),
                row["state"],
                int(row["action"]),
                row["next_state"],
                int(row["reward"]),
                row["terminated"] == "true",
            )
            games.setdefault(game, []).append(transition)

    return states, games


def model_based(states, games, report):
    # Строим модель переходов и дважды оцениваем заданную стратегию.
    report.append("\nModel-based learning")
    outcomes = {}
    for transitions in games.values():
        for _, state, action, next_state, reward, _ in transitions:
            successors = outcomes.setdefault((state, action), {})
            successors.setdefault(next_state, []).append(reward)

    # Считаем вероятности переходов и средние награды.
    model = {}
    report.append("State  Action  Next  Pair count  Outcome count  Probability              Mean reward")
    for state in states:
        for action, action_name in ACTIONS.items():
            pair = (state, action)
            if pair not in outcomes:
                report.append(f"{state:<6} {action_name:<7} unobserved")
                continue

            successors = outcomes[pair]
            pair_count = sum(len(rewards) for rewards in successors.values())
            model[pair] = {}
            for next_state, rewards in successors.items():
                count = len(rewards)
                probability = count / pair_count
                mean_reward = sum(rewards) / count
                model[pair][next_state] = (probability, mean_reward)
                probability_text = f"{count}/{pair_count} = {probability:.10g}"
                reward_text = f"{rewards} -> {sum(rewards)}/{count} = {mean_reward:.10g}"
                report.append(
                    f"{state:<6} {action_name:<7} {next_state:<5} {pair_count:<11} "
                    f"{count:<14} {probability_text:<24} {reward_text}"
                )

    values = dict.fromkeys(states, 0.0)
    sweeps = [values]
    for sweep in (1, 2):
        report.append(f"\nSweep {sweep} uses V{sweep - 1}")
        new_values = {}
        for state, (player_sum, _, _) in states.items():
            action = 1 if player_sum < 20 else 0
            value = 0.0
            terms = []
            # Используем только значения из предыдущего прохода.
            for next_state, (probability, reward) in model[state, action].items():
                continuation = 0.0 if next_state == "END" else values[next_state]
                value += probability * (reward + GAMMA * continuation)
                terms.append(
                    f"{probability:.10g} * ({reward:.10g} + "
                    f"{GAMMA} * {continuation:.10g}) [{next_state}]"
                )
            new_values[state] = value
            report.append(f"{state} ({ACTIONS[action]}): {' + '.join(terms)} = {value:.10g}")
        values = new_values
        sweeps.append(values)

    report.append("\nState  V0             V1             V2")
    for state in states:
        row = " ".join(f"{table[state]:<14.10g}" for table in sweeps)
        report.append(f"{state:<6} {row}")


def monte_carlo(states, games, report):
    # Считаем полный возврат для каждого посещения состояния.
    report.append("\nEvery-visit Monte Carlo (one pass)")
    report.append("Game  Step  State  Return calculation")
    totals = dict.fromkeys(states, 0.0)
    counts = dict.fromkeys(states, 0)

    for game, transitions in games.items():
        future_return = 0.0
        visit_lines = []
        # Идём от конца игры и накапливаем награды.
        for step, state, _, _, reward, _ in reversed(transitions):
            visit_return = reward + GAMMA * future_return
            totals[state] += visit_return
            counts[state] += 1
            visit_lines.append(
                f"{game:<5} {step:<5} {state:<6} "
                f"{reward} + {GAMMA} * {future_return:.10g} = {visit_return:.10g}"
            )
            future_return = visit_return
        report.extend(reversed(visit_lines))

    report.append("\nState  Count  Return sum     Average")
    for state in states:
        average = totals[state] / counts[state]
        report.append(f"{state:<6} {counts[state]:<6} {totals[state]:<14.10g} {average:.10g}")


def td_learning(states, games, report):
    # Обновляем значения состояний после каждого перехода в двух проходах.
    report.append("\nTD(0)")
    report.append("Target = reward + gamma * successor value")
    report.append("New value = old value + alpha * (target - old value)")
    values = dict.fromkeys(states, 0.0)
    pass_values = []

    for pass_number in (1, 2):
        report.append(f"\nTD(0) pass {pass_number}")
        report.append("Game  Step  State  Next  Old value      Reward  Successor      Target         New value")
        for game, transitions in games.items():
            for step, state, _, next_state, reward, terminated in transitions:
                # Для конечного перехода значение продолжения равно нулю.
                old_value = values[state]
                successor = 0.0 if terminated else values[next_state]
                target = reward + GAMMA * successor
                values[state] = old_value + ALPHA * (target - old_value)
                report.append(
                    f"{game:<5} {step:<5} {state:<6} {next_state:<5} {old_value:<14.10g} "
                    f"{reward:<7} {successor:<14.10g} {target:<14.10g} {values[state]:.10g}"
                )
        pass_values.append(values.copy())

    report.append("\nState  Pass 1         Pass 2 (final)")
    for state in states:
        report.append(f"{state:<6} {pass_values[0][state]:<14.10g} {pass_values[1][state]:.10g}")


def q_learning(states, games, report):
    # Обновляем значения действий, дважды проходя по тем же играм.
    report.append("\nQ-learning")
    report.append("Target = reward + gamma * maximum successor Q-value")
    report.append("New Q-value = old Q-value + alpha * (target - old Q-value)")
    q_values = {state: dict.fromkeys(ACTIONS, 0.0) for state in states}

    for pass_number in (1, 2):
        report.append(f"\nQ-learning pass {pass_number}")
        report.append("Game  Step  State  Action  Next  Old Q          Reward  Max next Q     Target         New Q")
        for game, transitions in games.items():
            for step, state, action, next_state, reward, terminated in transitions:
                # Выбираем максимальное Q-значение из двух действий.
                old_value = q_values[state][action]
                successor = 0.0 if terminated else max(q_values[next_state].values())
                target = reward + GAMMA * successor
                q_values[state][action] = old_value + ALPHA * (target - old_value)
                report.append(
                    f"{game:<5} {step:<5} {state:<6} {ACTIONS[action]:<7} {next_state:<5} "
                    f"{old_value:<14.10g} {reward:<7} {successor:<14.10g} "
                    f"{target:<14.10g} {q_values[state][action]:.10g}"
                )

    report.append("\nState  Q(stick)       Q(hit)         Greedy action")
    for state, action_values in q_values.items():
        best_value = max(action_values.values())
        best_actions = [
            ACTIONS[action] for action, value in action_values.items() if value == best_value
        ]
        greedy = " / ".join(best_actions)
        if len(best_actions) > 1:
            greedy += " (tie)"
        report.append(f"{state:<6} {action_values[0]:<14.10g} {action_values[1]:<14.10g} {greedy}")


def main():
    # Запускаем все четыре метода и сохраняем общий отчёт.
    states, games = load_data()
    report = [
        "Blackjack fixed-history replay",
        f"Gamma = {GAMMA}, alpha = {ALPHA}, initial values = 0",
        "Fixed policy: hit below 20, stick at 20 or 21",
        "Terminal continuation = 0",
        "Game order: " + ", ".join(str(game) for game in games),
        "States (player sum, dealer showing, usable ace):",
    ]
    for state, observation in states.items():
        report.append(f"{state}: {observation}")
    model_based(states, games, report)
    monte_carlo(states, games, report)
    td_learning(states, games, report)
    q_learning(states, games, report)

    # Выводим расчёты на экран и сохраняем их в текстовый файл.
    output_dir = ROOT / "outputs/paper-replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = "\n".join(report) + "\n"
    (output_dir / "replay.txt").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
