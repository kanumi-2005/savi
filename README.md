# Game-Theoretic Statistics and Sequential Anytime-Valid Inference

> **Project 3** - CSC14005 - Introduction to Machine Learning (Nhập môn Học máy)
> University of Science, Vietnam National University Ho Chi Minh City (VNUHCM-US)
> Faculty of Information Technology

## Table of Contents

- [Overview](#overview)
- [Course Information](#course-information)
- [Selected Topic](#selected-topic)
- [Project Structure](#project-structure)
- [What We Implemented](#what-we-implemented)
- [Extensions and Experiments](#extensions-and-experiments)
- [Code Instructions](#code-instructions)
- [Report Instructions](#report-instructions)
- [References](#references)
- [Team](#team)
- [License](#license)

## Overview

This project studies **game-theoretic statistics** and **Sequential
Anytime-Valid Inference (SAVI)**, a framework for statistical inference that
remains valid under continuous monitoring and data-dependent stopping. Instead
of relying only on fixed-sample p-values and confidence intervals, SAVI uses
betting interpretations, nonnegative martingales, e-values, e-processes, and
confidence sequences to quantify evidence over time.

The report develops the mathematical foundations of this framework and
connects hypothesis testing with betting, Kelly-optimal growth, likelihood
ratios, Kullback-Leibler divergence, and sequential estimation. The
experimental code provides reproducible demonstrations of two central ideas:

| Experiment | File | Goal |
|---|---|---|
| Kelly betting game | `code/kelly-game/main.py` | Verify the Kelly-optimal betting fraction and its connection to KL divergence |
| Quantile confidence sequences | `code/confidence-sequence/main.py` | Construct and empirically evaluate anytime-valid quantile bounds and CDF confidence bands |

## Course Information

| | |
|---|---|
| **Course** | Introduction to Machine Learning (Nhập môn Học máy) |
| **Instructor (Theory)** | Dr. Bùi Tiến Lên |
| **Instructor (Lab)** | MSc. Lê Nhựt Nam |
| **Class** | 23_24 |
| **Group** | 13 |
| **Semester** | Semester 2, 2026 |

## Selected Topic

The project focuses on **Game-Theoretic Statistics and Sequential
Anytime-Valid Inference** through the following topics:

1. Limitations of fixed-sample p-values and confidence intervals under
   optional stopping and continuous monitoring.
2. E-variables, p-variables, tests, test supermartingales, and e-processes.
3. Sequential tests and anytime-valid p-processes derived from e-processes.
4. Ville's inequality and its role in controlling Type-I error uniformly over
   time.
5. Testing by betting for simple and composite hypotheses.
6. Kelly betting, logarithmic growth optimality, likelihood ratios, and
   Kullback-Leibler divergence.
7. Confidence sequences, asymptotic confidence sequences, and sequential
   estimation for means, quantiles, and treatment effects.

## Project Structure

```text
savi/
├── code/
│   ├── confidence-sequence/
│   │   └── main.py                  # Quantile confidence sequence experiment
│   └── kelly-game/
│       └── main.py                  # Kelly betting simulation
├── report/
│   ├── core-theory/                 # SAVI definitions and theory
│   ├── experiments/                 # Experimental results
│   ├── information/                 # Team information
│   ├── overview/                    # Motivation and topic overview
│   ├── resources/                   # Figures, tables, and university logo
│   ├── preamble.tex                 # LaTeX packages and shared commands
│   ├── references.bib               # Bibliography
│   ├── report.tex                   # Main LaTeX source
│   └── report.pdf                   # Compiled report
├── requirements.txt                 # Python dependencies
└── README.md                        # Project overview and usage guide
```

## What We Implemented

The report and code cover both theoretical and practical aspects of SAVI:

1. A structured Vietnamese report written in LaTeX.
2. A discussion of optional stopping and why conventional fixed-sample
   inference may become invalid under continuous monitoring.
3. Definitions and relationships among e-values, p-values, sequential tests,
   test supermartingales, e-processes, and confidence sequences.
4. A betting-based interpretation of hypothesis testing for simple and
   composite null hypotheses.
5. A Kelly betting simulation for a biased coin with $p=0.6$, verifying the
   theoretical optimum $\lambda^*=2p-1=0.2$.
6. A numerical check that optimal expected log-wealth growth agrees with the
   KL divergence from Bernoulli($p$) to Bernoulli($0.5$).
7. Quantile confidence sequences and simultaneous CDF confidence bands,
   demonstrated primarily on the heavy-tailed Cauchy distribution.
8. Monte Carlo coverage checks for a fixed quantile and a finite grid of
   quantiles and observation times.

## Extensions and Experiments

Beyond the core theoretical presentation, this project adds:

1. A direct numerical comparison between the simulated Kelly-optimal betting
   fraction and its analytical solution.
2. A visualization of average log-wealth growth over a grid of betting
   fractions.
3. Confidence-sequence experiments for Cauchy, normal, exponential,
   Student-t, and uniform distributions.
4. A visualization comparing a fixed-quantile confidence sequence with
   simultaneous empirical CDF confidence bands.
5. Repeated simulations that estimate time-uniform coverage on finite
   evaluation grids.

## Code Instructions

Python 3.11 or newer is required by the pinned dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the Kelly betting experiment:

```bash
python code/kelly-game/main.py
```

Run the quantile confidence-sequence experiment:

```bash
python code/confidence-sequence/main.py
```

Both programs print numerical results and open a Matplotlib figure. The
confidence-sequence script performs 300 Monte Carlo trials by default, so it
may take longer to finish. Its distribution, sample size, confidence level,
and trial count can be adjusted in the `main()` function.

## Report Instructions

The report uses Vietnamese typesetting and is configured for LuaLaTeX through
`latexmk`.

```bash
cd report
latexmk
```

The compiled document is written to `report/report.pdf`. A TeX distribution
with LuaLaTeX and the packages listed in `report/preamble.tex` is required.

## References

The main theoretical material is based on:

- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023).
  *Game-theoretic statistics and safe anytime-valid inference*.
- Ramdas, A., & Wang, R. (2025). *Hypothesis Testing with E-values*.

See [`report/references.bib`](report/references.bib) for bibliography details.

## Team

| Name | Student ID | Role |
|---|---|---|
| Hoàng Ngọc Phú | 23120010 | Team Lead |
| Hoàng Ngọc Quí | 23120077 | Member |
| Nguyễn Duy Bảo | 23120113 | Member |

## License

This project is developed for educational purposes as part of the Introduction
to Machine Learning course at VNUHCM-US.
