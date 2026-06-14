import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, norm, expon, t, uniform


def get_distribution(dist_name):
    if dist_name == "cauchy":
        return {
            "name": "Cauchy(0,1)",
            "sample": lambda rng, n: rng.standard_cauchy(size=n),
            "cdf": lambda x: cauchy.cdf(x),
            "ppf": lambda q: cauchy.ppf(q),
            "x_grid": np.linspace(-6, 6, 500),
            "ylim": (1, 16),
        }

    if dist_name == "normal":
        return {
            "name": "Normal(0,1)",
            "sample": lambda rng, n: rng.normal(0, 1, size=n),
            "cdf": lambda x: norm.cdf(x),
            "ppf": lambda q: norm.ppf(q),
            "x_grid": np.linspace(-4, 4, 500),
            "ylim": None,
        }

    if dist_name == "exponential":
        return {
            "name": "Exponential(1)",
            "sample": lambda rng, n: rng.exponential(scale=1, size=n),
            "cdf": lambda x: expon.cdf(x, scale=1),
            "ppf": lambda q: expon.ppf(q, scale=1),
            "x_grid": np.linspace(0, 8, 500),
            "ylim": None,
        }

    if dist_name == "t":
        df = 3
        return {
            "name": f"Student-t(df={df})",
            "sample": lambda rng, n: rng.standard_t(df=df, size=n),
            "cdf": lambda x: t.cdf(x, df=df),
            "ppf": lambda q: t.ppf(q, df=df),
            "x_grid": np.linspace(-6, 6, 500),
            "ylim": None,
        }

    if dist_name == "uniform":
        return {
            "name": "Uniform(0,1)",
            "sample": lambda rng, n: rng.uniform(0, 1, size=n),
            "cdf": lambda x: uniform.cdf(x, loc=0, scale=1),
            "ppf": lambda q: uniform.ppf(q, loc=0, scale=1),
            "x_grid": np.linspace(0, 1, 500),
            "ylim": None,
        }

    raise ValueError(f"Unknown distribution name: {dist_name}")

def u_fixed_quantile(n, alpha):
    return np.sqrt(
        (
            0.73 * np.log(np.log(2.04 * n))
            + 0.52 * np.log(9.97 / alpha)
        ) / n
    )

def u_all_quantiles(n, alpha):
    return np.sqrt(
        (
            np.log(np.log(np.e * n))
            + 0.75 * np.log(34 / alpha)
        ) / n
    )

def empirical_quantile_for_plot(sample, q):
    q = np.clip(q, 0.0, 1.0)
    return np.quantile(sample, q, method="higher")

def empirical_quantile_for_check(sample, q):
    if q <= 0.0:
        return -np.inf
    if q >= 1.0:
        return np.inf
    return np.quantile(sample, q, method="higher")

def ecdf_values(sample, x_grid):
    sample_sorted = np.sort(sample)
    return np.searchsorted(sample_sorted, x_grid, side="right") / len(sample)

def compute_quantile_bounds(sample, q_level, alpha, t_grid):
    lower_bounds = []
    upper_bounds = []

    for n in t_grid:
        sample_n = sample[:n]
        u_n = u_fixed_quantile(n, alpha)

        lower = empirical_quantile_for_plot(sample_n, q_level - u_n)
        upper = empirical_quantile_for_plot(sample_n, q_level + u_n)

        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return np.array(lower_bounds), np.array(upper_bounds)

def print_quantile_coverage(sample, dist, q_level, alpha, t_values):
    true_q = dist["ppf"](q_level)

    print()
    print(f"Coverage check for {q_level} quantile")
    print("Distribution:", dist["name"])
    print("True quantile:", true_q)
    print()

    for n in t_values:
        sample_n = sample[:n]
        u_n = u_fixed_quantile(n, alpha)

        lower_level = q_level - u_n
        upper_level = q_level + u_n

        lower = empirical_quantile_for_check(sample_n, lower_level)
        upper = empirical_quantile_for_check(sample_n, upper_level)

        covered = lower <= true_q <= upper

        print(f"t = {n}")
        print(f"  u_t = {u_n:.6f}")
        print(f"  lower level raw = {lower_level:.6f}")
        print(f"  upper level raw = {upper_level:.6f}")
        print(f"  lower bound = {lower}")
        print(f"  true Q({q_level}) = {true_q:.6f}")
        print(f"  upper bound = {upper}")
        print(f"  contains true quantile? {covered}")
        print()

def check_fixed_quantile_one_trial(sample, dist, q_level, alpha, t_grid):
    true_q = dist["ppf"](q_level)

    for n in t_grid:
        sample_n = sample[:n]
        u_n = u_fixed_quantile(n, alpha)

        lower = empirical_quantile_for_check(sample_n, q_level - u_n)
        upper = empirical_quantile_for_check(sample_n, q_level + u_n)

        if not (lower <= true_q <= upper):
            return False

    return True


def check_all_quantiles_one_trial(sample, dist, p_grid, alpha, t_grid):
    true_q_values = dist["ppf"](p_grid)

    for n in t_grid:
        sample_n = sample[:n]
        u_n = u_all_quantiles(n, alpha)

        for p_value, true_q in zip(p_grid, true_q_values):
            lower = empirical_quantile_for_check(sample_n, p_value - u_n)
            upper = empirical_quantile_for_check(sample_n, p_value + u_n)

            if not (lower <= true_q <= upper):
                return False

    return True

def estimate_fixed_quantile_coverage(
    dist,
    q_level,
    alpha,
    sample_size,
    n_trials,
    seed,
):
    rng = np.random.default_rng(seed)
    t_grid = np.unique(np.logspace(2, np.log10(sample_size), 100).astype(int))

    success = 0

    for _ in range(n_trials):
        sample = dist["sample"](rng, sample_size)

        ok = check_fixed_quantile_one_trial(
            sample=sample,
            dist=dist,
            q_level=q_level,
            alpha=alpha,
            t_grid=t_grid,
        )

        if ok:
            success += 1

    coverage = success / n_trials

    print()
    print("Estimated probability for fixed quantile")
    print("Distribution:", dist["name"])
    print("q_level:", q_level)
    print("alpha:", alpha)
    print("target coverage:", 1 - alpha)
    print("estimated coverage:", coverage)
    print("success / n_trials:", success, "/", n_trials)

    return coverage

def estimate_all_quantiles_coverage(
    dist,
    alpha,
    sample_size,
    n_trials,
    seed,
):
    rng = np.random.default_rng(seed)

    t_grid = np.unique(np.logspace(2, np.log10(sample_size), 50).astype(int))
    p_grid = np.linspace(0.01, 0.99, 99)

    success = 0

    for _ in range(n_trials):
        sample = dist["sample"](rng, sample_size)

        ok = check_all_quantiles_one_trial(
            sample=sample,
            dist=dist,
            p_grid=p_grid,
            alpha=alpha,
            t_grid=t_grid,
        )

        if ok:
            success += 1

    coverage = success / n_trials

    print()
    print("Estimated probability for all quantiles")
    print("Distribution:", dist["name"])
    print("alpha:", alpha)
    print("target coverage:", 1 - alpha)
    print("estimated coverage:", coverage)
    print("success / n_trials:", success, "/", n_trials)
    print("p_grid size:", len(p_grid))
    print("t_grid size:", len(t_grid))

    return coverage


def plot_confidence_sequence(
    sample,
    dist,
    q_level,
    alpha,
    t_grid,
    t_list,
    lower_bounds,
    upper_bounds,
):
    true_q = dist["ppf"](q_level)
    x_grid = dist["x_grid"]
    true_cdf = dist["cdf"](x_grid)

    fig = plt.figure(figsize=(12, 5))

    ax_left = fig.add_subplot(1, 2, 1)

    ax_left.plot(t_grid, lower_bounds, linewidth=1.7, color="mediumseagreen")
    ax_left.plot(t_grid, upper_bounds, linewidth=1.7, color="mediumseagreen")
    ax_left.axhline(true_q, color="black", linestyle=":", linewidth=1.6)

    ax_left.set_xscale("log")
    ax_left.set_xlabel("Number of samples, t")
    ax_left.set_ylabel("Confidence bounds for 90%ile")
    ax_left.set_title("Only 0.9 quantile")

    if dist["ylim"] is not None:
        ax_left.set_ylim(*dist["ylim"])

    ax_left.grid(False)

    for i, n in enumerate(t_list):
        ax = fig.add_subplot(3, 2, 2 + 2 * i)

        sample_n = sample[:n]
        u_n = u_all_quantiles(n, alpha)

        f_hat = ecdf_values(sample_n, x_grid)

        lower_band = np.maximum(f_hat - u_n, 0)
        upper_band = np.minimum(f_hat + u_n, 1)

        ax.plot(x_grid, true_cdf, color="gray", linewidth=1.5)
        ax.plot(x_grid, f_hat, color="black", linestyle=":", linewidth=1.3)
        ax.plot(x_grid, lower_band, color="black", linewidth=1.2)
        ax.plot(x_grid, upper_band, color="black", linewidth=1.2)

        ax.set_ylim(0, 1)

        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x")

        if i == 1:
            ax.set_ylabel("CDF confidence band")

        if i == 0:
            ax.set_title("All quantiles simultaneously")

        ax.text(
            1.02,
            0.5,
            f"t = {n:,}",
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="left",
        )

        ax.grid(False)

    plt.tight_layout()
    plt.show()

def main():
    dist_name = "cauchy"
    alpha = 0.05
    sample_size = 100000
    seed = 42
    q_level = 0.9

    n_trials = 300
    coverage_sample_size = 10000

    rng = np.random.default_rng(seed)
    dist = get_distribution(dist_name)

    sample = dist["sample"](rng, sample_size)

    t_grid = np.unique(np.logspace(2, 5, 250).astype(int))
    t_list = [100, 1000, 10000]

    lower_bounds, upper_bounds = compute_quantile_bounds(
        sample=sample,
        q_level=q_level,
        alpha=alpha,
        t_grid=t_grid,
    )

    print_quantile_coverage(
        sample=sample,
        dist=dist,
        q_level=q_level,
        alpha=alpha,
        t_values=[100, 1000, 10000, 100000],
    )

    estimate_fixed_quantile_coverage(
        dist=dist,
        q_level=q_level,
        alpha=alpha,
        sample_size=coverage_sample_size,
        n_trials=n_trials,
        seed=1,
    )

    estimate_all_quantiles_coverage(
        dist=dist,
        alpha=alpha,
        sample_size=coverage_sample_size,
        n_trials=n_trials,
        seed=2,
    )

    plot_confidence_sequence(
        sample=sample,
        dist=dist,
        q_level=q_level,
        alpha=alpha,
        t_grid=t_grid,
        t_list=t_list,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


if __name__ == "__main__":
    main()
