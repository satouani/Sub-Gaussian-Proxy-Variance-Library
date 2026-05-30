"""
Figures for the paper:
"Optimal sub-Gaussian variance proxy for 3-mass distributions"
Atouani, Marchal, Arbel

Reproduces Figures 1, 2, 3, 4.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for pre-render
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────
# Shared helpers: asymmetric 3-mass on {-1, 0, 1}
# ─────────────────────────────────────────────────────────────

def u0(lam, p1, p2):
    """E[exp(lambda*Y)] = p1*e^{-lam} + p2*e^{lam} + p3"""
    p3 = 1 - p1 - p2
    return p1 * np.exp(-lam) + p2 * np.exp(lam) + p3


def u1(lam, p1, p2):
    """Derivative of u0"""
    return -p1 * np.exp(-lam) + p2 * np.exp(lam)


def MY(lam, p1, p2):
    """Cumulant generating function of Y - mu.
    MY(lambda) = log(u0(lambda)) - lambda * mu,   mu = p2 - p1
    """
    mu = p2 - p1
    return np.log(u0(lam, p1, p2)) - lam * mu


def MY_prime(lam, p1, p2):
    """M'Y(lambda) = u1/u0 - mu"""
    mu = p2 - p1
    return u1(lam, p1, p2) / u0(lam, p1, p2) - mu


def F(lam, p1, p2):
    """Black curve: lambda * M'Y(lambda) - 2 * MY(lambda).
    Zeros give candidate lambda_c values (eq. 3 of the paper).
    """
    return lam * MY_prime(lam, p1, p2) - 2 * MY(lam, p1, p2)


def gY(lam, sigma2, p1, p2):
    """gY(lambda; sigma^2) = lambda^2 * sigma^2 / 2 - MY(lambda)"""
    return 0.5 * lam**2 * sigma2 - MY(lam, p1, p2)


# ─────────────────────────────────────────────────────────────
# Shared helpers: symmetric 3-mass on {-1, 0, 1}
# ─────────────────────────────────────────────────────────────

def g_sym(lam, sigma2, p):
    """gY for symmetric case: lambda^2*sigma^2/2 - log(2p*cosh(lambda) + 1 - 2p)"""
    return 0.5 * lam**2 * sigma2 - np.log(2 * p * np.cosh(lam) + 1 - 2 * p)


def sigma2_1_sym(p):
    """Upper bound eq. (4): (1-2p)^2 / (4*(1-4p))"""
    return (1 - 2 * p) ** 2 / (4 * (1 - 4 * p))


def lambda2_sym(sigma2, p):
    """lambda_2(sigma) for symmetric case (larger root of g''=0)"""
    disc = (1 - 2 * p) ** 2 - 4 * (1 - 4 * p) * sigma2
    if disc < 0:
        return None
    num = (1 - 2 * p) * (1 - 2 * sigma2) + np.sqrt(disc)
    den = 4 * p * sigma2
    val = num / den
    if val < 1:
        return None
    return np.arccosh(val)


def sigma2_2_sym(p):
    """Numerical solution of eq. (5): fixed-point for sigma^2_2(p)."""
    def eq(sig2):
        l2 = lambda2_sym(sig2, p)
        if l2 is None:
            return np.nan
        denom = l2 * (1 + 2 * p * np.cosh(l2) - 2 * p)
        if abs(denom) < 1e-14:
            return np.nan
        return sig2 - 2 * p * np.sinh(l2) / denom

    lo, hi = 2 * p + 1e-10, sigma2_1_sym(p) - 1e-10
    try:
        v_lo, v_hi = eq(lo), eq(hi)
        if np.isnan(v_lo) or np.isnan(v_hi):
            return None
        if v_lo * v_hi > 0:
            return None
        return brentq(eq, lo, hi, xtol=1e-12)
    except Exception:
        return None


def sigma2_opt_sym(p):
    """Optimal variance proxy for symmetric 3-mass (Theorem 3.1)."""
    if p >= 1 / 6:
        return 2 * p          # strictly sub-Gaussian
    # Solve: p*lam*sinh(lam) - (1-2p+2p*cosh(lam)) * log(1-2p+2p*cosh(lam)) = 0
    lambda0 = np.arccosh((1 - 4 * p - 4 * p**2) / (2 * p * (1 - 2 * p)))

    def eq(lam):
        v = 1 - 2 * p + 2 * p * np.cosh(lam)
        return p * lam * np.sinh(lam) - v * np.log(v)

    try:
        lam_c = brentq(eq, lambda0 + 1e-9, 60, xtol=1e-12)
        return 2 * p * np.sinh(lam_c) / (lam_c * (2 * p * np.cosh(lam_c) + 1 - 2 * p))
    except Exception:
        return 2 * p



# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Illustration of Theorem 2.1  (p1=0.05, p2=0.01)
# ─────────────────────────────────────────────────────────────

def find_nonzero_zeros(p1, p2, lam_min=-9, lam_max=12, n=50_000):
    """Find all zeros of F(lambda) away from 0."""
    lams = np.linspace(lam_min, lam_max, n)
    fvals = F(lams, p1, p2)
    zeros = []
    for i in range(len(fvals) - 1):
        if lams[i] * lams[i+1] > 0 and fvals[i] * fvals[i+1] < 0:
            try:
                z = brentq(lambda l: F(l, p1, p2), lams[i], lams[i+1], xtol=1e-12)
                zeros.append(z)
            except Exception:
                pass
    return zeros


def is_local_minimum(lam_c, p1, p2, eps=0.3, n=500):
    """Check whether gY(.; sigma^2_c) has a local minimum at lam_c."""
    sigma2_c = MY_prime(lam_c, p1, p2) / lam_c
    lams = np.linspace(lam_c - eps, lam_c + eps, n)
    gvals = gY(lams, sigma2_c, p1, p2)
    return float(np.min(gvals)) >= -1e-6, sigma2_c


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Computation
# ─────────────────────────────────────────────────────────────

def compute_figure2(p1=0.05, p2=0.01, lam_min=-7.0, lam_max=11.5, n_pts=60_000):
    """Compute zeros of the non-centered F and classify them as local minima."""
    p3 = 1 - p1 - p2
    def _u0(l): return p1*np.exp(-l) + p2*np.exp(l) + p3
    def _u1(l): return -p1*np.exp(-l) + p2*np.exp(l)
    def _F(l):  return l * (_u1(l) / _u0(l)) - 2*np.log(_u0(l))
    def _sc(l): return (_u1(l) / _u0(l)) / l
    def _gY(l, s2): return 0.5*l**2*s2 - np.log(_u0(l))

    lams = np.linspace(lam_min, lam_max, n_pts)
    fvals = np.vectorize(_F)(lams)

    zeros = []
    for i in range(len(lams) - 1):
        if lams[i] * lams[i+1] > 0 and fvals[i] * fvals[i+1] < 0:
            try:
                zeros.append(brentq(_F, lams[i], lams[i+1], xtol=1e-10))
            except Exception:
                pass

    sc_vals = [_sc(z) for z in zeros]

    def _is_min(z, eps=0.4):
        ls = np.linspace(z - eps, z + eps, 500)
        return bool(np.min(np.vectorize(lambda l: _gY(l, _sc(z)))(ls)) >= -1e-6)

    colors = ["#448C2A" if _is_min(z) else "#A32920" for z in zeros]

    return {
        "lams": lams, "fvals": fvals,
        "zeros": zeros, "sc_vals": sc_vals, "colors": colors,
        "var_Y": p1 + p2 - (p2 - p1)**2,
        "p1": p1, "p2": p2,
    }


def plot_figure2(data=None, save=True):
    if data is None:
        data = compute_figure2()
    lams_main = data["lams"]
    fvals     = data["fvals"]
    zeros     = data["zeros"]
    sc_vals   = data["sc_vals"]
    colors    = data["colors"]
    var_Y     = data["var_Y"]
    p1        = data["p1"]
    p2        = data["p2"]
    p3        = 1 - p1 - p2

    print(f"Zeros of F_alt: {[f'{z:.4f}' for z in zeros]}")
    print(f"sc values:      {[f'{s:.4f}' for s in sc_vals]}")

    def u0(lam): return p1*np.exp(-lam) + p2*np.exp(lam) + p3
    def gY_alt(lam, sigma2): return 0.5 * lam**2 * sigma2 - np.log(u0(lam))

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 11})

    fig, ax = plt.subplots(figsize=(13, 7.5))

    ax.plot(lams_main, fvals, "k-", linewidth=2.2, zorder=3)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)

    ax.tick_params(axis='both', colors='#666666', labelcolor='black', length=5)
    ax.set_xticks([-6, -4, -2, 2, 4, 6, 8, 10])
    ax.set_yticks([-0.2, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_xlim(-7.5, 11.5)
    ax.set_ylim(-0.55, 1.75)

    ax.annotate("", xy=(11.8, 0), xytext=(11.0, 0),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.6), annotation_clip=False)
    ax.text(11.3, 0.05, r"$\lambda$", fontsize=13)
    ax.annotate("", xy=(0, 1.77), xytext=(0, 1.70),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.5), annotation_clip=False)

    zero_labels = [r"$\lambda_{c_1}$", r"$\lambda^*$", r"$\lambda_{c_2}$"]
    for z, lbl in zip(zeros, zero_labels):
        ax.scatter([z], [0], color="k", s=40, zorder=6)
    ax.text(zeros[0] - 0.1, 0.03, zero_labels[0], ha="right", va="bottom", fontsize=13)
    ax.text(zeros[1] - 0.15, 0.03, zero_labels[1], ha="left", va="bottom", fontsize=13)
    ax.text(zeros[2] - 0.1, 0.03, zero_labels[2], ha="right", va="bottom", fontsize=13)

    inset_pos = [
        [0.06, 0.60, 0.22, 0.30],
        [0.4,  0.60, 0.21, 0.30],
        [0.84, 0.60, 0.22, 0.30],
    ]
    inset_xlim = [
        (zeros[0] - 0.33, zeros[0] + 0.33),
        (zeros[1] - 0.30, zeros[1] + 0.33),
        (zeros[2] - 0.33, zeros[2] + 0.33),
    ]

    for i, (z, col, pos, xlim) in enumerate(zip(zeros, colors, inset_pos, inset_xlim)):
        axins = ax.inset_axes(pos)
        sc = sc_vals[i]
        lz = np.linspace(xlim[0], xlim[1], 600)
        gz = np.array([gY_alt(l, sc) for l in lz])

        axins.plot(lz, gz, color=col, linewidth=1.8)
        axins.axvline(z, color="blue", linestyle="--", linewidth=1.5)

        axins.set_xlim(xlim)
        margin = (np.max(gz) - np.min(gz)) * 0.1
        if margin == 0:
            margin = 0.1
        axins.set_ylim(np.min(gz) - margin, np.max(gz) + margin)

        axins.set_yticks([])
        xt = np.linspace(xlim[0], xlim[1], 3)
        axins.set_xticks(np.round(xt, 1))
        axins.tick_params(axis='x', labelsize=10, colors='black', direction='in', pad=4)

        # --- NOUVEAU STYLE DE BOX ---
        # Masquer les bordures du haut et de droite
        axins.spines['top'].set_visible(False)
        axins.spines['right'].set_visible(False)
        
        # Garder et styliser uniquement les bordures de gauche et du bas
        for spine_name in ['left', 'bottom']:
            axins.spines[spine_name].set_edgecolor('#333333')
            axins.spines[spine_name].set_linewidth(0.8)
            axins.spines[spine_name].set_alpha(0.8)
        # -----------------------------

        ax.annotate("",
                    xy=(0.4, 0), xycoords=axins.transAxes,
                    xytext=(z, 0), textcoords="data",
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=11),
                    zorder=2)

    box_green = "#448C2A"
    box_blue  = "#4A5AFF"

    ax.annotate(rf"$\sigma_{{\mathrm{{opt}}}}^2 = s_{{c_1}} \approx {sc_vals[0]:.2f}$",
                xy=(zeros[0], 0), xycoords="data",
                xytext=(zeros[0] - 0.2, -0.25), textcoords="data",
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=box_green, lw=1.5),
                arrowprops=dict(arrowstyle="-", color=box_green, linestyle="--", lw=1.2), zorder=4)

    ax.annotate(rf"$\mathrm{{Var}} \approx {var_Y:.3f}$",
                xy=(0, 0), xycoords="data",
                xytext=(1.2, -0.25), textcoords="data",
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=box_blue, lw=1.5),
                arrowprops=dict(arrowstyle="-", color=box_blue, linestyle="--", lw=1.2), zorder=4)

    ax.annotate(rf"$s_{{c_2}} \approx {sc_vals[2]:.2f}$",
                xy=(zeros[2], 0), xycoords="data",
                xytext=(zeros[2] - 0.7, -0.25), textcoords="data",
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=box_blue, lw=1.5),
                arrowprops=dict(arrowstyle="-", color=box_blue, linestyle="--", lw=1.2), zorder=4)

    if save:
        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/illustration_thm21.pdf", dpi=300, bbox_inches="tight",
                    pad_inches=0.3)
        fig.savefig("figures/illustration_thm21.png", dpi=150, bbox_inches="tight",
                    pad_inches=0.3)
        print("Saved figures/illustration_thm21.pdf/.png")
        plt.close(fig)
    else:
        plt.show()




def compute_figure3_sym(n_pts=250):
    """Variance proxy curves for the symmetric 3-mass case."""
    p_vals = np.linspace(0.002, 1/6 - 0.002, n_pts)
    s2_v = np.array([sigma2_2_sym(p) for p in p_vals])
    return {
        "p_vals": p_vals,
        "var_v":  2 * p_vals,
        "s1_v":   sigma2_1_sym(p_vals),
        "s2_v":   s2_v,
        "opt_v":  np.array([sigma2_opt_sym(p) for p in p_vals]),
    }


def compute_figure3_asym(n=600):
    """Regime classification grid and boundary for the asymmetric case."""
    p1g = np.linspace(0, 1, n)
    P1, P2 = np.meshgrid(p1g, p1g)
    P3 = 1 - P1 - P2
    inside = P3 > 0
    Z = np.full_like(P1, np.nan)
    Z[inside & (P3 <= 4*np.sqrt(P1*P2))] = 1.0   # closed-form regime
    Z[inside & (P3 >  4*np.sqrt(P1*P2))] = 2.0   # numerical regime

    p1b = np.linspace(0.0001, 0.9999, 4000)
    sqp2 = -2*np.sqrt(p1b) + np.sqrt(3*p1b + 1)
    p2b  = sqp2**2
    mask = (sqp2 > 0) & (p1b + p2b < 1)
    return {
        "grid": (P1, P2, Z),
        "boundary": (p1b[mask], p2b[mask]),
    }


# ────────────────────────────────────────────────────────────
# FIGURE 3 — Symmetric (left) + Asymmetric regimes (right)
# ─────────────────────────────────────────────────────────────

def plot_figure3(data_sym=None, data_asym=None, save=True):
    if data_sym is None:
        data_sym = compute_figure3_sym()
    if data_asym is None:
        data_asym = compute_figure3_asym()

    p_vals = data_sym["p_vals"]
    var_v  = data_sym["var_v"]
    s1_v   = data_sym["s1_v"]
    s2_v   = data_sym["s2_v"]
    opt_v  = data_sym["opt_v"]

    P1, P2, Z        = data_asym["grid"]
    p1b_bnd, p2b_bnd = data_asym["boundary"]

    def _make_left():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(p_vals, opt_v, "k-",  linewidth=2.5, label=r"$\sigma^2_{\mathrm{opt}}$")
        ax.plot(p_vals, var_v, "b-",  linewidth=2,   label=r"$\mathrm{Var}[Y]=2p$")
        ax.plot(p_vals, s1_v,  "r-",  linewidth=2,   label=r"$\sigma^2_1(p)$")
        valid = [i for i, v in enumerate(s2_v) if v is not None and not np.isnan(v)]
        if valid:
            ax.plot(p_vals[valid], s2_v[valid], "g-", linewidth=2, label=r"$\sigma^2_2(p)$")
        ax.axvline(1 / 6, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(r"$p$", fontsize=12)
        ax.set_ylabel(r"$\sigma^2$", fontsize=12)
        ax.set_xlim(0, 1 / 6 + 0.005)
        ax.set_ylim(0, 0.35)
        ax.set_xticks([0, 0.1, 1 / 6])
        ax.set_xticklabels(["$0$", "$0.1$", r"$\frac{1}{6}$"])
        ax.legend(fontsize=10, loc="upper left")
        plt.tight_layout()
        return fig

    def _make_right():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.contourf(P1, P2, Z, levels=[0.5, 1.5], colors=["steelblue"], alpha=0.45)
        ax.contourf(P1, P2, Z, levels=[1.5, 2.5], colors=["firebrick"], alpha=0.45)
        ax.plot(p1b_bnd, p2b_bnd, "k-", linewidth=2.5)
        ax.plot([0, 1], [1, 0], "k-", linewidth=1.5)
        ax.text(0.45, 0.35, r"$p_3 \leq 4\sqrt{p_1 p_2}$",
                fontsize=11, ha="center", color="steelblue")
        ax.text(0.12, 0.08, r"$p_3 > 4\sqrt{p_1 p_2}$",
                fontsize=11, ha="center", color="firebrick")
        ax.set_xlabel(r"$p_1$", fontsize=12)
        ax.set_ylabel(r"$p_2$", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        return fig

    if save:
        os.makedirs("figures", exist_ok=True)
        fig_l = _make_left()
        fig_l.savefig("figures/illustration_symmetric_case.pdf", bbox_inches="tight")
        fig_l.savefig("figures/illustration_symmetric_case.png", bbox_inches="tight", dpi=150)
        plt.close(fig_l)
        print("Saved figures/illustration_symmetric_case.pdf/.png")
        fig_r = _make_right()
        fig_r.savefig("figures/illustration_asym_limits.pdf", bbox_inches="tight")
        fig_r.savefig("figures/illustration_asym_limits.png", bbox_inches="tight", dpi=150)
        plt.close(fig_r)
        print("Saved figures/illustration_asym_limits.pdf/.png")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # rebuild on shared canvas for interactive display
        fig_l = _make_left()
        fig_r = _make_right()
        plt.show()



def compute_figure4(p1=0.13, p2=0.25, n_curves=9):
    """Compute g_sigma curves for the closed-form asymmetric regime."""
    p3 = 1 - p1 - p2
    var_Y = p1 + p2 - (p2 - p1)**2
    sigma2_opt  = 2*(p2 - p1) / (np.log(p2) - np.log(p1))
    sigma2_upper = 2*np.sqrt(p1*p2) / (p3 + 2*np.sqrt(p1*p2))

    lams = np.linspace(-1.05, 0.45, 2000)
    sigma2_list = np.linspace(sigma2_upper, var_Y, n_curves)
    g_curves = [gY(lams, s2, p1, p2) for s2 in sigma2_list]

    return {
        "lams": lams, "g_curves": g_curves,
        "sigma2_list": sigma2_list, "sigma2_opt": sigma2_opt,
        "sigma2_upper": sigma2_upper, "var_Y": var_Y,
        "p1": p1, "p2": p2,
    }


# ────────────────────────────────────────────────────────────
# FIGURE 4 — g_{sigma, p1, p2}(lambda) for (p1,p2)=(0.13, 0.25)
# ─────────────────────────────────────────────────────────────

def plot_figure4(data=None, save=True):
    if data is None:
        data = compute_figure4()
    lams          = data["lams"]
    g_curves      = data["g_curves"]
    sigma2_list   = data["sigma2_list"]
    sigma2_opt    = data["sigma2_opt"]

    colors = plt.cm.RdYlBu(np.linspace(0.05, 0.95, len(sigma2_list)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (s2, gvals) in enumerate(zip(sigma2_list, g_curves)):
        diff = abs(s2 - sigma2_opt)
        if diff == min(abs(sigma2_list - sigma2_opt)):
            ax.plot(lams, gvals, color="darkorange", linewidth=3, zorder=5,
                    label=rf"$\sigma^2_{{\mathrm{{opt}}}} \approx {sigma2_opt:.3f}$")
        else:
            ax.plot(lams, gvals, color=colors[i], linewidth=1.6, alpha=0.85)

    # Reference lines
    ax.axhline(0, color="k", linewidth=0.8)

    # Remove ticks and frame
    ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4])
    ax.set_xticklabels(["-1", "-0.8", "-0.6", "-0.4", "-0.2", "0.2", "0.4"], fontsize=9)
    ax.set_yticks([])
    ax.tick_params(axis='x', which='both', length=4, direction='out', colors='#444444')
    ax.xaxis.set_tick_params(bottom=True)
    ax.tick_params(axis='y', which='both', length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Move x-axis spine to y=0 so ticks sit on the actual axis line
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['bottom'].set_linewidth(0.0)  # hidden — arrow handles it

    ax.set_xlim(-1.05, 0.48)

    ymax = max(np.max(g) for g in g_curves)
    ymin = min(np.min(g) for g in g_curves)
    ax.set_ylim(ymin * 1.08, ymax * 1.18)

    # x-axis arrow + λ label above the arrow at the right end
    ax.annotate("", xy=(0.50, 0), xytext=(-1.08, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2), annotation_clip=False)
    ax.text(0.485, ymax * 0.08, r"$\lambda$", fontsize=13, ha="left", va="bottom")

    # y-axis arrow + g label at the top, centered on the axis
    ax.annotate("", xy=(0, ymax * 1.15), xytext=(0, ymin * 1.08),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2), annotation_clip=False)
    ax.text(-0.015, ymax * 1.12, r"$g_{\sigma,p}(\lambda)$", fontsize=12,
            ha="right", va="center")

    ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    if save:
        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/g_plot_p1p2_1325.pdf", bbox_inches="tight")
        fig.savefig("figures/g_plot_p1p2_1325.png", bbox_inches="tight", dpi=150)
        print("Saved figures/g_plot_p1p2_1325.pdf/.png")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("\n=== Figure 2 ===")
    plot_figure2()
    print("\n=== Figure 3 ===")
    plot_figure3()
    print("\n=== Figure 4 ===")
    plot_figure4()
    print("\nAll figures saved to figures/")
