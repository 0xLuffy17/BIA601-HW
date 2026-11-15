import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FeatureSelectionProblem:
    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, alpha: float = 0.98, k_neighbors: int = 5):
        self.name = name
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int64)
        self.n_samples, self.n_features = self.X.shape
        self.alpha = float(alpha)
        self.k = int(k_neighbors)

        norm_y = self.y - np.mean(self.y)
        std_y = np.std(self.y)
        std_y = 1e-6 if std_y == 0 else std_y
        corrs = np.zeros(self.n_features, dtype=np.float64)
        for j in range(self.n_features):
            col = self.X[:, j]
            std_c = np.std(col)
            std_c = 1e-6 if std_c == 0 else std_c
            corrs[j] = abs(np.mean((col - np.mean(col)) * norm_y) / (std_c * std_y))
        self.heuristics = np.clip(corrs, 0.01, 1.0)

    def evaluate_chromosome(self, solution: np.ndarray):
        sol = np.asarray(solution, dtype=np.int64)
        selected_idx = np.where(sol == 1)[0]
        num_selected = len(selected_idx)
        if num_selected == 0:
            return 1.0, 0.0, 0

        X_sub = self.X[:, selected_idx]
        G = np.dot(X_sub, X_sub.T)
        sq_norms = np.diag(G)
        dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * G
        np.fill_diagonal(dists, np.inf)

        k_val = min(self.k, self.n_samples - 1)
        nn_indices = np.argpartition(dists, k_val, axis=1)[:, :k_val]
        neighbor_labels = self.y[nn_indices]
        votes = np.round(np.mean(neighbor_labels, axis=1))
        error_rate = float(np.mean(votes != self.y))
        accuracy = 1.0 - error_rate

        fitness = self.alpha * error_rate + (1.0 - self.alpha) * (num_selected / float(self.n_features))
        return float(fitness), float(accuracy), int(num_selected)

def generate_benchmark_instance(name: str, n_features: int, n_samples: int = 200, seed: int = 42) -> FeatureSelectionProblem:
    rng = np.random.RandomState(seed)
    n_informative = max(4, int(n_features * 0.12))
    n_redundant = max(3, int(n_features * 0.08))
    n_noise = n_features - n_informative - n_redundant

    X_informative = rng.randn(n_samples, n_informative)
    weights = rng.uniform(0.7, 1.5, size=n_informative)
    logit = np.dot(X_informative, weights) + rng.randn(n_samples) * 0.4
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (prob > 0.5).astype(np.int64)

    X_redundant = np.zeros((n_samples, n_redundant))
    for r in range(n_redundant):
        source_idx = r % n_informative
        X_redundant[:, r] = X_informative[:, source_idx] + rng.randn(n_samples) * 0.25

    X_noise = rng.randn(n_samples, n_noise)
    X_all = np.hstack([X_informative, X_redundant, X_noise])
    perm = rng.permutation(n_features)
    X_all = X_all[:, perm]
    X_all = (X_all - np.mean(X_all, axis=0)) / (np.std(X_all, axis=0) + 1e-6)

    return FeatureSelectionProblem(name, X_all, y)

class SequentialForwardSelection:
    def __init__(self, problem: FeatureSelectionProblem):
        self.problem = problem

    def solve(self):
        start_time = time.perf_counter()
        n = self.problem.n_features
        current_mask = np.zeros(n, dtype=np.int64)
        best_fitness = 1.0
        best_accuracy = 0.0
        best_count = 0

        ranks = np.argsort(-self.problem.heuristics)
        for feat in ranks[:max(5, int(n * 0.25))]:
            current_mask[feat] = 1
            fit, acc, count = self.problem.evaluate_chromosome(current_mask)
            if fit < best_fitness:
                best_fitness = fit
                best_accuracy = acc
                best_count = count
            else:
                current_mask[feat] = 0

        elapsed = time.perf_counter() - start_time
        if best_count == 0:
            current_mask[ranks[0]] = 1
            best_fitness, best_accuracy, best_count = self.problem.evaluate_chromosome(current_mask)

        return {
            "algorithm": "Greedy Forward Selection (SFS)",
            "solution": current_mask,
            "best_fitness": best_fitness,
            "accuracy": best_accuracy,
            "num_features": best_count,
            "execution_time": elapsed,
            "iterations_to_converge": 0
        }

class GeneticAlgorithmFS:
    def __init__(self, problem: FeatureSelectionProblem, pop_size: int = 40, max_iter: int = 70,
                 crossover_rate: float = 0.85, mutation_rate: float = None,
                 tournament_size: int = 3, elitism_count: int = 2, seed: int = None):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate if mutation_rate is not None else (1.0 / problem.n_features)
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.rng = np.random.RandomState(seed)

    def solve(self):
        start_time = time.perf_counter()
        n = self.problem.n_features

        pop = self.rng.choice([0, 1], size=(self.pop_size, n), p=[0.75, 0.25]).astype(np.int64)
        for i in range(self.pop_size):
            if np.sum(pop[i]) == 0:
                pop[i, self.rng.randint(0, n)] = 1

        fits = np.zeros(self.pop_size)
        accs = np.zeros(self.pop_size)
        counts = np.zeros(self.pop_size, dtype=np.int64)
        for i in range(self.pop_size):
            f, a, c = self.problem.evaluate_chromosome(pop[i])
            fits[i], accs[i], counts[i] = f, a, c

        lead_idx = np.argmin(fits)
        global_best_sol = pop[lead_idx].copy()
        global_best_fit = fits[lead_idx]
        global_best_acc = accs[lead_idx]
        global_best_cnt = counts[lead_idx]

        convergence_curve = [global_best_fit]
        conv_iter = 0

        for it in range(1, self.max_iter + 1):
            elite_idx = np.argsort(fits)[:self.elitism_count]
            elites = [pop[e].copy() for e in elite_idx]

            new_pop = []
            while len(new_pop) < (self.pop_size - self.elitism_count):
                c1 = self.rng.choice(self.pop_size, size=self.tournament_size, replace=False)
                p1 = pop[c1[np.argmin(fits[c1])]].copy()
                c2 = self.rng.choice(self.pop_size, size=self.tournament_size, replace=False)
                p2 = pop[c2[np.argmin(fits[c2])]].copy()

                if self.rng.rand() < self.crossover_rate:
                    mask = self.rng.rand(n) < 0.5
                    ch1 = np.where(mask, p1, p2)
                    ch2 = np.where(mask, p2, p1)
                else:
                    ch1, ch2 = p1.copy(), p2.copy()

                m1 = self.rng.rand(n) < self.mutation_rate
                ch1[m1] = 1 - ch1[m1]
                m2 = self.rng.rand(n) < self.mutation_rate
                ch2[m2] = 1 - ch2[m2]

                if np.sum(ch1) == 0:
                    ch1[self.rng.randint(0, n)] = 1
                if np.sum(ch2) == 0:
                    ch2[self.rng.randint(0, n)] = 1

                new_pop.append(ch1)
                if len(new_pop) < (self.pop_size - self.elitism_count):
                    new_pop.append(ch2)

            pop = np.array(new_pop + elites)
            for i in range(self.pop_size):
                f, a, c = self.problem.evaluate_chromosome(pop[i])
                fits[i], accs[i], counts[i] = f, a, c

            curr_lead = np.argmin(fits)
            if fits[curr_lead] < global_best_fit:
                global_best_fit = fits[curr_lead]
                global_best_sol = pop[curr_lead].copy()
                global_best_acc = accs[curr_lead]
                global_best_cnt = counts[curr_lead]
                conv_iter = it

            convergence_curve.append(global_best_fit)

        elapsed = time.perf_counter() - start_time
        return {
            "algorithm": "Genetic Algorithm (GA)",
            "solution": global_best_sol,
            "best_fitness": global_best_fit,
            "accuracy": global_best_acc,
            "num_features": global_best_cnt,
            "convergence_curve": convergence_curve,
            "execution_time": elapsed,
            "iterations_to_converge": conv_iter
        }

class BinaryParticleSwarmFS:
    def __init__(self, problem: FeatureSelectionProblem, swarm_size: int = 40, max_iter: int = 70,
                 w_start: float = 0.9, w_end: float = 0.4, c1: float = 2.0, c2: float = 2.0,
                 v_max: float = 6.0, seed: int = None):
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.rng = np.random.RandomState(seed)

    def solve(self):
        start_time = time.perf_counter()
        n = self.problem.n_features

        positions = self.rng.choice([0, 1], size=(self.swarm_size, n), p=[0.75, 0.25]).astype(np.int64)
        for i in range(self.swarm_size):
            if np.sum(positions[i]) == 0:
                positions[i, self.rng.randint(0, n)] = 1
        velocities = self.rng.uniform(-self.v_max, self.v_max, size=(self.swarm_size, n))

        pbest_pos = positions.copy()
        pbest_fit = np.zeros(self.swarm_size)
        pbest_acc = np.zeros(self.swarm_size)
        pbest_cnt = np.zeros(self.swarm_size, dtype=np.int64)

        for i in range(self.swarm_size):
            f, a, c = self.problem.evaluate_chromosome(positions[i])
            pbest_fit[i], pbest_acc[i], pbest_cnt[i] = f, a, c

        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]
        gbest_acc = pbest_acc[gbest_idx]
        gbest_cnt = pbest_cnt[gbest_idx]

        convergence_curve = [gbest_fit]
        conv_iter = 0

        for it in range(1, self.max_iter + 1):
            w = self.w_start - ((self.w_start - self.w_end) * (it / float(self.max_iter)))
            r1 = self.rng.rand(self.swarm_size, n)
            r2 = self.rng.rand(self.swarm_size, n)

            velocities = (w * velocities +
                          self.c1 * r1 * (pbest_pos - positions) +
                          self.c2 * r2 * (gbest_pos - positions))
            velocities = np.clip(velocities, -self.v_max, self.v_max)

            sigmoid = 1.0 / (1.0 + np.exp(-velocities))
            positions = (self.rng.rand(self.swarm_size, n) < sigmoid).astype(np.int64)

            for i in range(self.swarm_size):
                if np.sum(positions[i]) == 0:
                    positions[i, self.rng.randint(0, n)] = 1
                f, a, c = self.problem.evaluate_chromosome(positions[i])
                if f < pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest_pos[i] = positions[i].copy()
                    pbest_acc[i] = a
                    pbest_cnt[i] = c
                    if f < gbest_fit:
                        gbest_fit = f
                        gbest_pos = positions[i].copy()
                        gbest_acc = a
                        gbest_cnt = c
                        conv_iter = it

            convergence_curve.append(gbest_fit)

        elapsed = time.perf_counter() - start_time
        return {
            "algorithm": "Binary Particle Swarm (BPSO)",
            "solution": gbest_pos,
            "best_fitness": gbest_fit,
            "accuracy": gbest_acc,
            "num_features": gbest_cnt,
            "convergence_curve": convergence_curve,
            "execution_time": elapsed,
            "iterations_to_converge": conv_iter
        }

class BinaryAntColonyFS:
    def __init__(self, problem: FeatureSelectionProblem, num_ants: int = 30, max_iter: int = 50,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.15,
                 tau_min: float = 0.05, tau_max: float = 5.0, seed: int = None):
        self.problem = problem
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rng = np.random.RandomState(seed)

        self.tau = np.full(problem.n_features, tau_max)
        self.eta = self.problem.heuristics.copy()

    def solve(self):
        start_time = time.perf_counter()
        n = self.problem.n_features

        global_best_sol = np.ones(n, dtype=np.int64)
        global_best_fit = 1.0
        global_best_acc = 0.0
        global_best_cnt = n
        conv_iter = 0
        convergence_curve = []

        for it in range(1, self.max_iter + 1):
            desirability = (self.tau ** self.alpha) * (self.eta ** self.beta)
            prob_select = desirability / (1.0 + desirability)
            prob_select = np.clip(prob_select, 0.05, 0.95)

            iter_sols = []
            iter_fits = []
            iter_accs = []
            iter_cnts = []

            for _ in range(self.num_ants):
                sol = (self.rng.rand(n) < prob_select).astype(np.int64)
                if np.sum(sol) == 0:
                    sol[self.rng.randint(0, n)] = 1
                f, a, c = self.problem.evaluate_chromosome(sol)
                iter_sols.append(sol)
                iter_fits.append(f)
                iter_accs.append(a)
                iter_cnts.append(c)

            best_ant_idx = np.argmin(iter_fits)
            if iter_fits[best_ant_idx] < global_best_fit:
                global_best_fit = iter_fits[best_ant_idx]
                global_best_sol = iter_sols[best_ant_idx].copy()
                global_best_acc = iter_accs[best_ant_idx]
                global_best_cnt = iter_cnts[best_ant_idx]
                conv_iter = it

            convergence_curve.append(global_best_fit)

            self.tau = (1.0 - self.rho) * self.tau
            selected_features = np.where(global_best_sol == 1)[0]
            deposit = (1.0 - global_best_fit)
            self.tau[selected_features] += deposit
            self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

        elapsed = time.perf_counter() - start_time
        return {
            "algorithm": "Ant Colony Optimization (ACO)",
            "solution": global_best_sol,
            "best_fitness": global_best_fit,
            "accuracy": global_best_acc,
            "num_features": global_best_cnt,
            "convergence_curve": convergence_curve,
            "execution_time": elapsed,
            "iterations_to_converge": conv_iter
        }

def run_experiments():
    os.makedirs("results", exist_ok=True)

    instances = {
        "Small (D=30)": generate_benchmark_instance("Credit_Risk_D30", n_features=30, n_samples=200, seed=101),
        "Medium (D=100)": generate_benchmark_instance("Customer_Analytics_D100", n_features=100, n_samples=200, seed=202),
        "Large (D=300)": generate_benchmark_instance("HighDim_Finance_D300", n_features=300, n_samples=200, seed=303)
    }

    single_run_records = []
    stability_records = []

    for scale_name, problem in instances.items():
        all_features = np.ones(problem.n_features, dtype=np.int64)
        base_fit, base_acc, base_cnt = problem.evaluate_chromosome(all_features)
        single_run_records.append({
            "Scale": scale_name,
            "Algorithm": "Full Features (Baseline)",
            "Fitness": base_fit,
            "Accuracy_pct": base_acc * 100.0,
            "Features_Selected": base_cnt,
            "Reduction_pct": 0.0,
            "Time_s": 0.001,
            "Convergence_Iter": 0
        })

        sfs_res = SequentialForwardSelection(problem).solve()
        sfs_red = (1.0 - sfs_res["num_features"] / float(problem.n_features)) * 100.0
        single_run_records.append({
            "Scale": scale_name,
            "Algorithm": "Greedy Forward (SFS)",
            "Fitness": sfs_res["best_fitness"],
            "Accuracy_pct": sfs_res["accuracy"] * 100.0,
            "Features_Selected": sfs_res["num_features"],
            "Reduction_pct": sfs_red,
            "Time_s": sfs_res["execution_time"],
            "Convergence_Iter": 0
        })

        ga_res = GeneticAlgorithmFS(problem, seed=42).solve()
        ga_red = (1.0 - ga_res["num_features"] / float(problem.n_features)) * 100.0
        single_run_records.append({
            "Scale": scale_name,
            "Algorithm": "Genetic Algorithm (GA)",
            "Fitness": ga_res["best_fitness"],
            "Accuracy_pct": ga_res["accuracy"] * 100.0,
            "Features_Selected": ga_res["num_features"],
            "Reduction_pct": ga_red,
            "Time_s": ga_res["execution_time"],
            "Convergence_Iter": ga_res["iterations_to_converge"]
        })

        pso_res = BinaryParticleSwarmFS(problem, seed=42).solve()
        pso_red = (1.0 - pso_res["num_features"] / float(problem.n_features)) * 100.0
        single_run_records.append({
            "Scale": scale_name,
            "Algorithm": "Binary Swarm (BPSO)",
            "Fitness": pso_res["best_fitness"],
            "Accuracy_pct": pso_res["accuracy"] * 100.0,
            "Features_Selected": pso_res["num_features"],
            "Reduction_pct": pso_red,
            "Time_s": pso_res["execution_time"],
            "Convergence_Iter": pso_res["iterations_to_converge"]
        })

        aco_res = BinaryAntColonyFS(problem, seed=42).solve()
        aco_red = (1.0 - aco_res["num_features"] / float(problem.n_features)) * 100.0
        single_run_records.append({
            "Scale": scale_name,
            "Algorithm": "Ant Colony (ACO)",
            "Fitness": aco_res["best_fitness"],
            "Accuracy_pct": aco_res["accuracy"] * 100.0,
            "Features_Selected": aco_res["num_features"],
            "Reduction_pct": aco_red,
            "Time_s": aco_res["execution_time"],
            "Convergence_Iter": aco_res["iterations_to_converge"]
        })

        plt.figure(figsize=(8.0, 4.5), dpi=180)
        plt.axhline(y=base_fit, color="#7f7f7f", linestyle=":", linewidth=1.5, label=f"Full Features Baseline ({base_fit:.4f})")
        plt.axhline(y=sfs_res["best_fitness"], color="#9467bd", linestyle="--", linewidth=1.5, label=f"SFS Greedy Baseline ({sfs_res['best_fitness']:.4f})")
        plt.plot(ga_res["convergence_curve"], label="Genetic Algorithm (GA)", color="#1f77b4", linewidth=2.0)
        plt.plot(pso_res["convergence_curve"], label="Binary PSO (BPSO)", color="#ff7f0e", linewidth=2.0)
        plt.plot(aco_res["convergence_curve"], label="Ant Colony (ACO)", color="#2ca02c", linewidth=2.0)
        plt.title(f"Convergence Trajectory: {scale_name}", fontsize=11, fontweight="bold")
        plt.xlabel("Iteration", fontsize=10)
        plt.ylabel("Objective Fitness (Lower is Better)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
        plt.tight_layout()
        tag = scale_name.split()[0].lower()
        plt.savefig(f"results/convergence_{tag}_{problem.name.split('_')[-1].lower()}.png")
        plt.close()

        for run_id in range(10):
            current_seed = 1000 + run_id * 83
            g_run = GeneticAlgorithmFS(problem, seed=current_seed).solve()
            p_run = BinaryParticleSwarmFS(problem, seed=current_seed).solve()
            a_run = BinaryAntColonyFS(problem, seed=current_seed).solve()

            for alg_name, r in [("GA", g_run), ("BPSO", p_run), ("ACO", a_run)]:
                stability_records.append({
                    "Scale": scale_name,
                    "Algorithm": alg_name,
                    "Trial": run_id + 1,
                    "Fitness": r["best_fitness"],
                    "Accuracy_pct": r["accuracy"] * 100.0,
                    "Features_Selected": r["num_features"],
                    "Reduction_pct": (1.0 - r["num_features"] / float(problem.n_features)) * 100.0,
                    "Time_s": r["execution_time"]
                })

    single_df = pd.DataFrame(single_run_records)
    stability_df = pd.DataFrame(stability_records)

    single_df.to_csv("results/single_run_summary.csv", index=False)
    stability_df.to_csv("results/stability_10_runs.csv", index=False)

    stats_agg = stability_df.groupby(["Scale", "Algorithm"]).agg(
        Mean_Fitness=("Fitness", "mean"),
        Std_Fitness=("Fitness", "std"),
        Min_Fitness=("Fitness", "min"),
        Max_Fitness=("Fitness", "max"),
        Mean_Accuracy=("Accuracy_pct", "mean"),
        Std_Accuracy=("Accuracy_pct", "std"),
        Mean_Features=("Features_Selected", "mean"),
        Mean_Reduction=("Reduction_pct", "mean"),
        Mean_Time=("Time_s", "mean")
    ).reset_index()
    stats_agg.to_csv("results/statistical_analysis.csv", index=False)

    scales = ["Small (D=30)", "Medium (D=100)", "Large (D=300)"]
    algs = ["GA", "BPSO", "ACO"]
    colors = {"GA": "#1f77b4", "BPSO": "#ff7f0e", "ACO": "#2ca02c"}

    plt.figure(figsize=(8.5, 4.8), dpi=180)
    positions = []
    box_data = []
    box_colors = []
    for s_idx, sc in enumerate(scales):
        base_pos = s_idx * 4.0
        for a_idx, alg in enumerate(algs):
            pos = base_pos + a_idx
            positions.append(pos)
            data_pts = stability_df[(stability_df["Scale"] == sc) & (stability_df["Algorithm"] == alg)]["Fitness"].values
            box_data.append(data_pts)
            box_colors.append(colors[alg])

    bp = plt.boxplot(box_data, positions=positions, patch_artist=True, widths=0.7)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    plt.xticks([1.0, 5.0, 9.0], scales, fontsize=10, fontweight="bold")
    plt.ylabel("Objective Fitness Across 10 Trials", fontsize=10)
    plt.title("Statistical Stability & Robustness Distribution (10 Repeated Trials)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5)
    legend_handles = [plt.Line2D([0], [0], color=colors[a], lw=5, label=a) for a in algs]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/stability_boxplots.png")
    plt.close()

    plt.figure(figsize=(8.5, 4.5), dpi=180)
    acc_data = stats_agg.pivot(index="Scale", columns="Algorithm", values="Mean_Accuracy").reindex(scales)
    x = np.arange(len(scales))
    width = 0.25
    plt.bar(x - width, acc_data["GA"], width=width, label="GA", color="#1f77b4", alpha=0.85)
    plt.bar(x, acc_data["BPSO"], width=width, label="BPSO", color="#ff7f0e", alpha=0.85)
    plt.bar(x + width, acc_data["ACO"], width=width, label="ACO", color="#2ca02c", alpha=0.85)
    plt.xticks(x, scales, fontsize=10, fontweight="bold")
    plt.ylabel("Classification Accuracy (%)", fontsize=10)
    plt.title("Mean Classification Accuracy Across Scales", fontsize=11, fontweight="bold")
    plt.ylim(80, 100)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_features.png")
    plt.close()

    plt.figure(figsize=(8.5, 4.5), dpi=180)
    red_data = stats_agg.pivot(index="Scale", columns="Algorithm", values="Mean_Reduction").reindex(scales)
    plt.bar(x - width, red_data["GA"], width=width, label="GA", color="#1f77b4", alpha=0.85)
    plt.bar(x, red_data["BPSO"], width=width, label="BPSO", color="#ff7f0e", alpha=0.85)
    plt.bar(x + width, red_data["ACO"], width=width, label="ACO", color="#2ca02c", alpha=0.85)
    plt.xticks(x, scales, fontsize=10, fontweight="bold")
    plt.ylabel("Feature Reduction Rate (%)", fontsize=10)
    plt.title("Feature Dimensionality Reduction Rate Across Scales", fontsize=11, fontweight="bold")
    plt.ylim(50, 100)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("results/feature_reduction_comparison.png")
    plt.close()

    plt.figure(figsize=(8.5, 4.5), dpi=180)
    time_data = stats_agg.pivot(index="Scale", columns="Algorithm", values="Mean_Time").reindex(scales)
    plt.bar(x - width, time_data["GA"], width=width, label="GA", color="#1f77b4", alpha=0.85)
    plt.bar(x, time_data["BPSO"], width=width, label="BPSO", color="#ff7f0e", alpha=0.85)
    plt.bar(x + width, time_data["ACO"], width=width, label="ACO", color="#2ca02c", alpha=0.85)
    plt.xticks(x, scales, fontsize=10, fontweight="bold")
    plt.ylabel("Execution Time (Seconds - Log Scale)", fontsize=10)
    plt.yscale("log")
    plt.title("Computational Runtime Comparison Across Scales (Logarithmic Scale)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5, which="both")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("results/runtime_comparison.png")
    plt.close()

    print(single_df.to_string(index=False))
    print("\nStatistical Analysis (10 Runs):")
    print(stats_agg.to_string(index=False))

if __name__ == "__main__":
    run_experiments()
