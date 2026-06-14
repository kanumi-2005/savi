import numpy as np
import matplotlib.pyplot as plt


def simulate_log_growths(p, sample_size, lambda_grid, seed):
    rng = np.random.default_rng(seed)
    flips = rng.binomial(1, p, size=sample_size)

    log_growths = []

    for lam in lambda_grid:
        log_terms = np.log(1 + lam * (2 * flips - 1))
        log_growths.append(np.mean(log_terms))

    return np.array(log_growths)

def find_best_lambda(lambda_grid, log_growths):
    idx_best = np.argmax(log_growths)
    return idx_best, lambda_grid[idx_best], log_growths[idx_best]

def kl_bernoulli_to_half(p):
    return p * np.log(p / 0.5) + (1 - p) * np.log((1 - p) / 0.5)

def theoretical_log_growth(p, lam):
    return p * np.log(1 + lam) + (1 - p) * np.log(1 - lam)

def plot_result(
    lambda_grid,
    log_growths,
    kelly_lambda,
    best_lambda_sim,
    kl_value,
):
    plt.figure(figsize=(8, 5))

    plt.plot(lambda_grid, log_growths, label="Simulated log growth")

    plt.axvline(
        kelly_lambda,
        linestyle="--",
        label=f"Kelly lambda = {kelly_lambda:.3f}",
    )

    plt.axvline(
        best_lambda_sim,
        linestyle=":",
        label=f"Best simulated lambda = {best_lambda_sim:.3f}",
    )

    plt.axhline(
        kl_value,
        linestyle="-.",
        label=f"H(p | 0.5) = {kl_value:.5f}",
    )

    plt.xlabel("lambda")
    plt.ylabel("average log wealth")
    plt.title("Kelly lambda and KL log-growth check")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    p = 0.6
    sample_size = 1_000_000
    seed = 42

    kelly_lambda = 2 * (p - 0.5)
    lambda_grid = np.linspace(0, 0.999, 1000)

    log_growths = simulate_log_growths(
        p=p,
        sample_size=sample_size,
        lambda_grid=lambda_grid,
        seed=seed,
    )

    idx_best, best_lambda_sim, best_log_growth_sim = find_best_lambda(
        lambda_grid=lambda_grid,
        log_growths=log_growths,
    )

    idx_kelly = np.argmin(np.abs(lambda_grid - kelly_lambda))
    kelly_log_growth_sim = log_growths[idx_kelly]

    kl_value = kl_bernoulli_to_half(p)
    kelly_log_growth_theory = theoretical_log_growth(p, kelly_lambda)

    print("p =", p)
    print("T =", sample_size)
    print("Kelly lambda =", kelly_lambda)
    print("Best lambda from simulator =", best_lambda_sim)
    print("Difference in lambda =", abs(best_lambda_sim - kelly_lambda))
    print()

    print("Simulated log-growth at Kelly =", kelly_log_growth_sim)
    print("Best simulated log-growth =", best_log_growth_sim)
    print("Theoretical log-growth at Kelly =", kelly_log_growth_theory)
    print("KL H(p | 0.5) =", kl_value)
    print()

    print(
        "Difference: simulated Kelly growth vs KL =",
        abs(kelly_log_growth_sim - kl_value),
    )

    print(
        "Difference: theoretical Kelly growth vs KL =",
        abs(kelly_log_growth_theory - kl_value),
    )

    plot_result(
        lambda_grid=lambda_grid,
        log_growths=log_growths,
        kelly_lambda=kelly_lambda,
        best_lambda_sim=best_lambda_sim,
        kl_value=kl_value,
    )


if __name__ == "__main__":
    main()
