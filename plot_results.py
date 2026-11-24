import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


RESULTS = Path(__file__).resolve().parent / "outputs"
PLOTS = RESULTS / "plots"
SEEDS = (11, 22, 33)
EPSILONS = ("0.2", "0.0")
POLICY_COLORS = ("#dce2e8", "#92c5de", "#f6bf75", "#c8ace2")


def read_rows(path):
    # Загружаем сохранённые результаты из CSV.
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_training():
    # Строим графики обучения по всем 50 блокам для каждого seed.
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, layout="constrained")
    for axis, epsilon in zip(axes, EPSILONS):
        for seed in SEEDS:
            rows = read_rows(RESULTS / "q-learning" / f"epsilon-{epsilon}" / f"seed-{seed}" / "training_rewards.csv")
            games = [int(row["last_completed_game"]) for row in rows]
            if games != list(range(1000, 50001, 1000)):
                raise ValueError(f"Incomplete training curve for seed {seed}, epsilon {epsilon}")
            rewards = [float(row["mean_reward"]) for row in rows]
            axis.plot(games, rewards, marker="o", markersize=2.8, linewidth=1.2, label=f"Seed {seed}")
        axis.set(title=f"Training epsilon = {epsilon}", xlabel="Completed training games")
        axis.xaxis.set_major_locator(MultipleLocator(10000))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.axhline(0, color="#8a9299", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right")
    axes[0].set_ylabel("Mean training reward per 1,000 games")
    figure.suptitle("Q-learning training curves")
    figure.savefig(PLOTS / "training_curves.png", dpi=160)
    plt.close(figure)


def plot_evaluation():
    # Сравниваем результаты замороженных стратегий с baseline.
    rows = read_rows(RESULTS / "evaluation" / "summary.csv")
    values = {(int(row["evaluation_seed"]), row["training_epsilon"]): float(row["mean_reward"])
              for row in rows}
    seeds = sorted({seed for seed, _ in values})
    figure, axis = plt.subplots(figsize=(9, 5.2), layout="constrained")
    width = 0.24
    policies = (
        ("", "Fixed-policy baseline", "#64748b"),
        ("0.2", "Q-learning (training epsilon = 0.2)", "#17688b"),
        ("0.0", "Q-learning (training epsilon = 0)", "#a65114"),
    )
    for index, (epsilon, label, color) in enumerate(policies):
        positions = [position + (index - 1) * width for position in range(len(seeds))]
        bars = axis.bar(positions, [values[seed, epsilon] for seed in seeds],
                        width=width, label=label, color=color)
        axis.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    axis.set_xticks(range(len(seeds)), seeds)
    axis.set(title="Frozen-policy evaluation (10,000 games per run)",
             xlabel="Evaluation seed", ylabel="Mean evaluation reward")
    axis.margins(y=0.13)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9, ncol=3)
    figure.savefig(PLOTS / "frozen_policy_evaluation.png", dpi=160)
    plt.close(figure)


def plot_policies(ace_file, ace_title):
    # Показываем жадные действия, равные значения и пропуски в таблицах.
    figure, axes = plt.subplots(2, 3, figsize=(14, 13), sharex=True, sharey=True)
    categories = {"unknown": 0, "hit": 1, "stick": 2, "tie": 3}
    cmap = ListedColormap(POLICY_COLORS)
    for row_index, epsilon in enumerate(EPSILONS):
        for column_index, seed in enumerate(SEEDS):
            axis = axes[row_index, column_index]
            rows = read_rows(RESULTS / "q-learning" / f"epsilon-{epsilon}" / f"seed-{seed}" / ace_file)
            states = {(int(row["player_sum"]), int(row["dealer_showing"])): row for row in rows}
            sums = sorted({player_sum for player_sum, _ in states})
            dealers = sorted({dealer for _, dealer in states})
            grid = []
            # Отмечаем ненаблюдавшиеся действия буквами поверх цветной ячейки.
            for y, player_sum in enumerate(sums):
                grid_row = []
                for x, dealer in enumerate(dealers):
                    state = states[player_sum, dealer]
                    if state["state_status"] == "unobserved":
                        category = "unknown"
                    else:
                        category = "tie" if state["greedy_tie"] == "true" else state["greedy_action"]
                        missing = "".join(letter for action, letter in (("hit", "H"), ("stick", "S"))
                                          if state[f"{action}_status"] == "unobserved")
                        if missing:
                            axis.text(x, y, missing, ha="center", va="center", fontsize=7)
                    grid_row.append(categories[category])
                grid.append(grid_row)
            axis.imshow(grid, cmap=cmap, vmin=-0.5, vmax=3.5, origin="lower",
                        aspect="auto", interpolation="nearest")
            axis.set(title=f"Seed {seed}, epsilon = {epsilon}")
            axis.set_xticks(range(len(dealers)), dealers)
            axis.set_yticks(range(len(sums)), sums)
            if row_index == 1:
                axis.set_xlabel("Dealer showing")
            if column_index == 0:
                axis.set_ylabel("Player sum")
    labels = ("Unknown state (both actions unvisited)", "Hit", "Stick", "Tie")
    handles = [Patch(facecolor=color, edgecolor="#627181", label=label)
               for color, label in zip(POLICY_COLORS, labels)]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.037), ncol=4)
    figure.text(0.5, 0.025, "H = hit unvisited; S = stick unvisited in a visited state",
                ha="center", fontsize=10)
    figure.suptitle(f"Greedy Q-learning policies — {ace_title}", fontsize=17)
    figure.tight_layout(rect=(0, 0.075, 1, 0.965))
    figure.savefig(PLOTS / f"greedy_policy_{ace_file.removesuffix('.csv')}.png", dpi=160)
    plt.close(figure)


def main():
    # Сохраняем графики из готовых CSV.
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_training()
    plot_evaluation()
    plot_policies("no_usable_ace.csv", "No usable ace")
    plot_policies("usable_ace.csv", "Usable ace")
    print(f"Saved four PNG plots to {PLOTS} (Matplotlib {matplotlib.__version__}).")


if __name__ == "__main__":
    main()
