import matplotlib.pyplot as plt

def plot_residuals(res_gmres, res_bcg, res_bi, direct_norm):
    plt.semilogy(res_gmres, label="GMRES")
    plt.semilogy(res_bcg,   label="BCG")
    plt.semilogy(res_bi,    label="BiCGStab")
    plt.axhline(direct_norm, color="k", linestyle="--", label="Direct solver")

    plt.xlabel("Iteration")
    plt.ylabel(r"$\|r_k\|_2$")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_performance_vs_Qin(results_df):
    solvers = ["gmres", "bcg", "bicgstab"]
    colors = ["blue", "red", "green"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # iterations to convergence vs Qin
    for solver, color in zip(solvers, colors):
        df = results_df[results_df["solver"] == solver]
        ax[0].plot(df["Qin"], df["iterations"], marker='o', label=solver, color=color)

    ax[0].set_ylabel("Iterations to Convergence")
    ax[0].set_title("Solver Iterations vs Qin")
    ax[0].grid(True)
    ax[0].legend()

    # time to solve vs Qin
    solvers_with_direct = ["direct", "gmres", "bcg", "bicgstab"]
    colors2 = ["black", "blue", "red", "green"]

    for solver, color in zip(solvers_with_direct, colors2):
        df = results_df[results_df["solver"] == solver]
        ax[1].plot(df["Qin"], df["time"], marker='o', label=solver, color=color)

    ax[1].set_ylabel("Solve Time (s)")
    ax[1].set_xlabel("Qin")
    ax[1].set_title("Solver Time vs Qin")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()
