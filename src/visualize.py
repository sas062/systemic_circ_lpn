import matplotlib.pyplot as plt

def plot_residuals(res_gmres, res_bcg, res_bi, direct_norm):
    plt.semilogy(res_gmres, label="GMRES")
    plt.semilogy(res_bcg,   label="BCG")
    plt.semilogy(res_bi,    label="BiCGStab")
    plt.axhline(direct_norm, color="k", linestyle="--", label="Direct solver residual")

    plt.xlabel("Iteration")
    plt.ylabel("Residual norm ||r_k||_2")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()
