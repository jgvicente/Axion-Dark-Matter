#!/usr/bin/env python3
"""
Matching error of the naive "switch to CDM" scheme for ultralight axions.

Reference: Passaglia & Hu, PRD 105, 123529 (2022), discussion around Eq. (5).

Setup
-----
* Background: LCDM from CLASS (classy). The CDM component is REMOVED from
  CLASS's budget and replaced by the axion field, which is evolved
  self-consistently (backreaction included):
        (H/m)^2 = rho_rest(a)/m^2 + (phi'^2 + phi^2)/2
  where rho_rest = rho_tot - rho_cdm from CLASS (photons, neutrinos, baryons, Lambda).
* Units: CLASS uses c = 1 and Mpc, with H^2 = rho_tot [Mpc^-2]. The axion mass is
  converted once, m[Mpc^-1] = m[eV] / (hbar c [eV Mpc]). After that everything is
  dimensionless: time tau = m t, and the field is normalized so that
        rho_ax [CLASS units] = m^2 (phi'^2 + phi^2) / 2,   ' = d/dtau.
* Equation of motion (cosmic time t -> tau = m t):
        phi'' + 3 (H/m) phi' + phi = 0,      (ln a)' = H/m
  integrated with scipy's DOP853 (explicit RK8), rtol = 1e-11, dense output.
* Abundance: the initial field amplitude is found by shooting so that the late-time
  cycle-averaged a^3 rho_ax equals CLASS's rho_cdm today (axion = all of the DM).
* Matching error: switching to a pressureless fluid at tau_* gives, for all later
  times, a^3 rho_CDM = a_*^3 rho_ax(tau_*). The late-time fractional error is
        Delta(m/H_*) = a_*^3 rho_ax(tau_*) / <a^3 rho_ax>_late - 1,
  so ONE exact integration yields Delta for every switch point.
  Radiation-domination prediction (Bessel asymptotics, test-field limit):
        Delta ~ (3/2) (m/H_*)^-1 sin(2 tau_* - 3 pi/4).

Run:  python matching_error.py            (uses classy if importable)
      python matching_error.py --background analytic   (no CLASS, for testing)
"""

import argparse
import json
import math
import os
import time
from bisect import bisect_right

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import gamma, jv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# Constants and cosmology
# ----------------------------------------------------------------------------
HBARC_EV_M = 197.3269804e6 * 1e-15          # hbar c in eV m
MPC_M = 3.0856775814913673e22               # 1 Mpc in m
HBARC_EV_MPC = HBARC_EV_M / MPC_M           # ~ 6.39e-30 eV Mpc
C_KM_S = 299792.458

# Planck 2018 (TT,TE,EE+lowE+lensing) baseline, one massive neutrino
PLANCK18 = {
    "h": 0.6736,
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "T_cmb": 2.7255,
    "N_ur": 2.0328,
    "N_ncdm": 1,
    "m_ncdm": 0.06,
}

# Colors (categorical slots 1-4 of the reference palette)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_INK, C_MUTED = "#222222", "#777777"


class FastSpline:
    """Scalar cubic-spline evaluation without numpy call overhead (used in the RHS)."""

    def __init__(self, x, y):
        cs = CubicSpline(x, y)
        self.x = list(map(float, x))
        self.coef = [tuple(map(float, cs.c[:, i])) for i in range(cs.c.shape[1])]
        self.n = len(self.coef)
        self.vec = cs  # vectorized evaluation for arrays

    def __call__(self, xv):
        i = bisect_right(self.x, xv) - 1
        i = 0 if i < 0 else (self.n - 1 if i >= self.n else i)
        c0, c1, c2, c3 = self.coef[i]
        d = xv - self.x[i]
        return ((c0 * d + c1) * d + c2) * d + c3


def background_class(params):
    """rho_rest(a) = rho_tot - rho_cdm and rho_cdm today, in CLASS units [Mpc^-2]."""
    from classy import Class

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute(level=["background"])
    bg = cosmo.get_background()
    a = 1.0 / (1.0 + np.asarray(bg["z"]))
    rho_rest = np.asarray(bg["(.)rho_tot"]) - np.asarray(bg["(.)rho_cdm"])
    rho_cdm0 = float(np.asarray(bg["(.)rho_cdm"])[np.argmax(a)])
    try:
        cosmo.struct_cleanup()
    except AttributeError:
        pass
    order = np.argsort(a)
    a, rho_rest = a[order], rho_rest[order]
    keep = np.concatenate([[True], np.diff(a) > 0])
    return a[keep], rho_rest[keep], rho_cdm0


def background_analytic(params):
    """Fallback with the same parameters, massless neutrinos (N_eff = 3.044)."""
    h = params["h"]
    H0 = h / (C_KM_S / 100.0)                              # Mpc^-1
    omega_g = 2.4728e-5 * (params["T_cmb"] / 2.7255) ** 4
    omega_r = omega_g * (1.0 + 0.22711 * 3.044)
    Om_r, Om_b, Om_c = omega_r / h**2, params["omega_b"] / h**2, params["omega_cdm"] / h**2
    Om_L = 1.0 - Om_r - Om_b - Om_c
    a = np.logspace(-14, 0, 20000)
    rho_rest = H0**2 * (Om_r * a**-4 + Om_b * a**-3 + Om_L)
    return a, rho_rest, H0**2 * Om_c


# ----------------------------------------------------------------------------
# Klein-Gordon integration
# ----------------------------------------------------------------------------
class AxionKG:
    def __init__(self, a_tab, rho_rest_tab, rho_cdm0, m_eV):
        self.m_mpc = m_eV / HBARC_EV_MPC
        r = rho_rest_tab / self.m_mpc**2                  # rho_rest / m^2, dimensionless
        self.lnr = FastSpline(np.log(a_tab), np.log(r))
        self.lna_min, self.lna_max = math.log(a_tab[0]), math.log(a_tab[-1])
        self.target = rho_cdm0 / self.m_mpc**2           # <a^3 (phi'^2+phi^2)/2> today (a0 = 1)

    def h_of(self, phi, dphi, lna):
        """H/m including the axion's own energy density."""
        return math.sqrt(math.exp(self.lnr(lna)) + 0.5 * (dphi * dphi + phi * phi))

    def rhs(self, tau, y):
        phi, dphi, lna = y
        h = self.h_of(phi, dphi, lna)
        return [dphi, -3.0 * h * dphi - phi, h]

    def initial_conditions(self, A, mH_ini=1e-2):
        """Frozen field deep in Hubble drag; RD attractor phi = A (1 - tau^2/5)."""
        h_ini = 1.0 / mH_ini
        lna_i = brentq(lambda x: math.exp(self.lnr(x)) - h_ini**2, self.lna_min, self.lna_max)
        tau_i = 1.0 / (2.0 * h_ini)                       # tau = m t, t = 1/(2H) in RD
        return tau_i, [A * (1 - tau_i**2 / 5), -2.0 * A * tau_i / 5.0, lna_i]

    def solve(self, A, mH_end, rtol=1e-11, atol_field=1e-16):
        tau_i, y0 = self.initial_conditions(A)

        def stop(tau, y):
            return 1.0 / self.h_of(*y) - mH_end
        stop.terminal, stop.direction = True, 1

        sol = solve_ivp(self.rhs, (tau_i, 10.0 * mH_end), y0, method="DOP853",
                        rtol=rtol, atol=[atol_field, atol_field, 1e-13],
                        dense_output=True, events=stop)
        if sol.status != 1:
            raise RuntimeError(f"integration did not reach m/H = {mH_end}: {sol.message}")
        return sol, tau_i, sol.t_events[0][0]


def a3rho(y):
    """a^3 rho / m^2 = a^3 (phi'^2 + phi^2)/2 for an array of states (3, N)."""
    return np.exp(3 * y[2]) * 0.5 * (y[0] ** 2 + y[1] ** 2)


def cycle_average(sol, tau_end, n_periods=40, pts_per_period=256):
    """Mean of a^3 rho over an integer number of its periods (pi in tau) ending at tau_end."""
    tau = np.linspace(tau_end - n_periods * np.pi, tau_end, n_periods * pts_per_period + 1)
    return simpson(a3rho(sol.sol(tau)), x=tau) / (tau[-1] - tau[0])


def matching_scan(sol, tau_i, tau_end, late, mHs_min, mHs_max, pts_per_period=64):
    """Delta(m/H_*) on a grid of switch times fine enough to resolve every oscillation."""
    tau_c = np.geomspace(tau_i, tau_end, 400000)
    y_c = sol.sol(tau_c)
    mH_c = np.array([1.0 / math.sqrt(math.exp(lnr) + 0.5 * (p * p + d * d))
                     for p, d, lnr in zip(y_c[0], y_c[1], KG.lnr.vec(y_c[2]))])
    t_lo, t_hi = np.interp([mHs_min, mHs_max], mH_c, tau_c)
    tau_s = np.arange(t_lo, t_hi, np.pi / pts_per_period)
    y_s = sol.sol(tau_s)
    mH_s = np.interp(tau_s, tau_c, mH_c)
    return tau_s, mH_s, a3rho(y_s) / late - 1.0, (tau_c, mH_c, y_c)


# ----------------------------------------------------------------------------
# Validation: pure radiation domination, test field, exact Bessel solution
# ----------------------------------------------------------------------------
def validate_rd(mH_end=2e4, rtol=1e-11):
    tau_i = 5e-3
    rhs = lambda t, y: [y[1], -1.5 / t * y[1] - y[0]]                   # H/m = 1/(2 tau)
    sol = solve_ivp(rhs, (tau_i, mH_end / 2), [1 - tau_i**2 / 5, -2 * tau_i / 5],
                    method="DOP853", rtol=rtol, atol=1e-16, dense_output=True)
    N = gamma(1.25) * 2**0.25
    tau = np.geomspace(1.0, mH_end / 2, 200000)
    phi_num = sol.sol(tau)[0]
    phi_ex = N * tau**-0.25 * jv(0.25, tau)
    envelope = N * math.sqrt(2 / math.pi) * tau**-0.75
    resid = np.abs(phi_num - phi_ex) / envelope

    # late-time a^3 rho with a = tau^(1/2): analytic N'^2/2 = Gamma(5/4)^2 sqrt(2) / pi
    tau_end = mH_end / 2
    tw = np.linspace(tau_end - 40 * np.pi, tau_end, 40 * 256 + 1)
    yw = sol.sol(tw)
    late_num = simpson(tw**1.5 * 0.5 * (yw[0] ** 2 + yw[1] ** 2), x=tw) / (tw[-1] - tw[0])
    late_ex = gamma(1.25) ** 2 * math.sqrt(2) / math.pi

    ts = np.arange(1.5, 500.0, np.pi / 64)
    ys = sol.sol(ts)
    delta_num = ts**1.5 * 0.5 * (ys[0] ** 2 + ys[1] ** 2) / late_num - 1
    return dict(tau=tau, resid=resid, late_num=late_num, late_ex=late_ex,
                ts=ts, mHs=2 * ts, delta=delta_num, nsteps=sol.t.size)


def delta_rd_analytic(tau_s):
    """Leading-order RD prediction: Delta = (3/2)(m/H_*)^-1 sin(2 tau_* - 3pi/4), m/H = 2 tau."""
    return 1.5 / (2 * tau_s) * np.sin(2 * tau_s - 0.75 * np.pi)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK, which="both")
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)


def fig_overview(path, tau_c, mH_c, y_c, late, switch=3.0, mH_max=60.0):
    sel = mH_c <= mH_max
    mH, y = mH_c[sel], y_c[:, sel]
    a = np.exp(y[2])
    i_s = np.searchsorted(mH, switch)
    phi_scaled = y[0] * a**1.5
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                                   gridspec_kw=dict(hspace=0.08))
    ax1.plot(mH, phi_scaled / abs(phi_scaled[i_s]), color=C_BLUE, lw=1.4)
    ax1.set_ylabel(r"$a^{3/2}\phi$  (normalized at $m/H_*$)")
    a3r = a3rho(y)
    ax2.plot(mH, a3r / late, color=C_BLUE, lw=1.4, label=r"exact axion  $a^3\rho_{\rm ax}$")
    ax2.plot(mH[i_s:], np.full(mH.size - i_s, a3r[i_s] / late), color=C_ORANGE, lw=2,
             label=rf"CDM after switch at $m/H_*={switch:g}$")
    ax2.axhline(1.0, color=C_MUTED, lw=1, ls="--", label=r"late-time $\langle a^3\rho_{\rm ax}\rangle$")
    ax2.set_ylim(0.4, 1.6)
    ax2.set_ylabel(r"$a^3\rho\ /\ \langle a^3\rho_{\rm ax}\rangle_{\rm late}$")
    ax2.set_xlabel(r"$m/H$")
    ax2.legend(frameon=False, loc="lower right")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.axvline(switch, color=C_MUTED, lw=0.8, ls=":")
        style(ax)
    ax1.set_title("Axion background: frozen, released, oscillating", loc="left", color=C_INK)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_running(path, mH_c, y_c, late, switch=3.0, mH_max=2e3):
    """User's step 4: exact vs CDM, evolved together after the switch."""
    i_s = np.searchsorted(mH_c, switch)
    sel = (mH_c >= switch) & (mH_c <= mH_max)
    mH, a3r = mH_c[sel], a3rho(y_c[:, sel])
    a3r_star = a3rho(y_c[:, i_s:i_s + 1])[0]
    diff = a3r / a3r_star - 1.0                      # rho_exact / rho_CDM - 1
    offset = late / a3r_star - 1.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                                   gridspec_kw=dict(hspace=0.1, height_ratios=[1.2, 1]))
    ax1.plot(mH, diff, color=C_BLUE, lw=0.9, label=r"$\rho_{\rm exact}/\rho_{\rm CDM}-1$")
    ax1.axhline(offset, color=C_ORANGE, lw=2, label="frozen offset (matching error)")
    env = (1.0 + offset) * 1.5 / mH                 # (3/2)(m/H)^-1 in units of the late density
    ax1.plot(mH, offset + env, color=C_MUTED, lw=1, ls="--",
             label=r"offset $\pm\,\frac{3}{2}(m/H)^{-1}$")
    ax1.plot(mH, offset - env, color=C_MUTED, lw=1, ls="--")
    ax1.set_ylabel("fractional difference")
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_title(rf"Exact field vs CDM fluid after switching at $m/H_*={switch:g}$",
                  loc="left", color=C_INK)
    ax2.loglog(mH, np.abs(diff - offset) / (1.0 + offset), color=C_BLUE, lw=0.6,
               label=r"$|\rho_{\rm exact}-\langle\rho_{\rm exact}\rangle|/\langle\rho_{\rm exact}\rangle$  (oscillating part)")
    ax2.loglog(mH, 1.5 / mH, color=C_MUTED, lw=1.2, ls="--", label=r"$\frac{3}{2}(m/H)^{-1}$")
    ax2.set_ylim(1e-6, 2)
    ax2.set_xlabel(r"$m/H$")
    ax2.set_ylabel("absolute value")
    ax2.legend(frameon=False, loc="upper right", fontsize=8)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        style(ax)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_matching(path, mH_s, delta, tau_s, mH_s_rd, delta_rd, source):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.0), gridspec_kw=dict(hspace=0.32))
    ax1.loglog(mH_s, np.abs(delta), color=C_BLUE, lw=0.7, label=r"$|\Delta|$, full cosmology")
    ax1.loglog(mH_s, 1.5 / mH_s, color=C_ORANGE, lw=1.6, ls="--",
               label=r"$\frac{3}{2}\,(m/H_*)^{-1}$")
    ax1.set_ylim(1e-6, 1)
    ax1.set_ylabel(r"$|\Delta| = |a_*^3\rho_*/\langle a^3\rho\rangle_{\rm late}-1|$")
    ax1.legend(frameon=False, loc="lower left")
    ax1.set_title("Late-time matching error vs switch time", loc="left", color=C_INK)
    lab = "CLASS" if source == "class" else "analytic LCDM"
    s1, s2 = mH_s <= 60, mH_s_rd <= 60
    ax2.plot(mH_s_rd[s2], (delta_rd * mH_s_rd)[s2], color=C_AQUA, lw=2.6, alpha=0.6,
             label="pure radiation, test field")
    ax2.plot(mH_s[s1], (delta * mH_s)[s1], color=C_BLUE, lw=1.0,
             label=f"full cosmology ({lab} + backreaction)")
    ax2.axhline(1.5, color=C_ORANGE, lw=1.4, ls="--", label=r"$\pm 3/2$ (Bessel asymptotics)")
    ax2.axhline(-1.5, color=C_ORANGE, lw=1.4, ls="--")
    ax2.set_xlabel(r"switch time  $m/H_*$")
    ax2.set_ylabel(r"$\Delta \times (m/H_*)$")
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_xlim(2.8, 62)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=8)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        style(ax)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_validation(path, v):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), gridspec_kw=dict(hspace=0.35))
    ax1.loglog(v["tau"], v["resid"], color=C_BLUE, lw=0.6)
    ax1.set_xlabel(r"$\tau = mt$")
    ax1.set_ylabel(r"$|\phi_{\rm num}-\phi_{\rm Bessel}|$ / envelope")
    ax1.set_title(r"Radiation domination test: DOP853 vs $\Gamma(5/4)(\tau/2)^{-1/4}J_{1/4}(\tau)$",
                  loc="left", color=C_INK)
    sel = (v["mHs"] >= 10) & (v["mHs"] <= 150)
    ax2.plot(v["mHs"][sel], (v["delta"] * v["mHs"])[sel], color=C_BLUE, lw=1.2, label="numerical")
    ax2.plot(v["mHs"][sel], (delta_rd_analytic(v["ts"]) * v["mHs"])[sel], color=C_ORANGE, lw=1,
             ls="--", label=r"$\frac{3}{2}\sin(2\tau_*-3\pi/4)$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$m/H_*$")
    ax2.set_ylabel(r"$\Delta\times(m/H_*)$")
    ax2.legend(frameon=False, loc="lower right", fontsize=8)
    for ax in (ax1, ax2):
        style(ax)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    global KG
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mass", type=float, default=1e-22, help="axion mass in eV")
    p.add_argument("--mH-end", type=float, default=2e4, help="stop the exact integration at this m/H")
    p.add_argument("--mHs-max", type=float, default=1e3, help="largest switch m/H_* in the scan")
    p.add_argument("--background", choices=["auto", "class", "analytic"], default="auto")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    # --- background -------------------------------------------------------
    source = args.background
    if source in ("auto", "class"):
        try:
            a_tab, rho_rest, rho_cdm0 = background_class(PLANCK18)
            source = "class"
        except ImportError:
            if args.background == "class":
                raise
            say("WARNING: classy not importable -> using analytic LCDM background (massless nu).")
            source = "analytic"
    if source == "analytic":
        a_tab, rho_rest, rho_cdm0 = background_analytic(PLANCK18)
    KG = AxionKG(a_tab, rho_rest, rho_cdm0, args.mass)
    say(f"background: {source};  m = {args.mass:g} eV = {KG.m_mpc:.6e} Mpc^-1;  "
        f"m/H0 = {KG.m_mpc / math.sqrt(rho_rest[-1] + rho_cdm0):.3e}")

    # --- shooting for the DM abundance ------------------------------------
    A, t0 = 1.0, time.time()
    for it in range(12):
        sol, tau_i, tau_end = KG.solve(A, args.mH_end)
        late = cycle_average(sol, tau_end)
        ratio = KG.target / late
        say(f"  shoot {it}: A = {A:.12e}  <a^3 rho>/target - 1 = {1/ratio - 1:+.3e}  "
            f"(steps {sol.t.size}, {time.time() - t0:.1f}s)")
        if abs(ratio - 1) < 1e-10:
            break
        A *= math.sqrt(ratio)
    a_end = math.exp(sol.sol(tau_end)[2])
    say(f"converged: tau_i = {tau_i:.3e}, tau_end = {tau_end:.4e}, a_end = {a_end:.4e}, "
        f"steps = {sol.t.size}, ~{tau_end / (2 * np.pi):.0f} oscillations")

    # --- matching-error scan -------------------------------------------------
    tau_s, mH_s, delta, (tau_c, mH_c, y_c) = matching_scan(sol, tau_i, tau_end, late, 3.0, args.mHs_max)
    i3 = 0
    say(f"Delta at m/H_* = 3: {delta[i3]:+.6e}   (|Delta| (m/H_*) = {abs(delta[i3]) * 3:.4f})")

    # --- numerical checks ----------------------------------------------------
    late_half = None
    tau_half = np.interp(args.mH_end / 2, mH_c, tau_c)
    late_half = cycle_average(sol, tau_half)
    say(f"check: <a^3rho> at m/H={args.mH_end/2:g} vs {args.mH_end:g}: rel diff "
        f"{late_half / late - 1:+.2e}  (expected O((H/m)^2))")
    sol_lo, _, tau_end_lo = KG.solve(A, args.mH_end, rtol=1e-9, atol_field=1e-14)
    late_lo = cycle_average(sol_lo, tau_end_lo)
    delta_lo = a3rho(sol_lo.sol(tau_s)) / late_lo - 1
    say(f"check: max |Delta(rtol=1e-9) - Delta(rtol=1e-11)| = {np.max(np.abs(delta_lo - delta)):.2e}")

    v = validate_rd(args.mH_end)
    say(f"RD validation: max |phi - Bessel|/envelope (tau>1) = {v['resid'].max():.2e};  "
        f"late a^3rho num/exact - 1 = {v['late_num'] / v['late_ex'] - 1:+.2e}")
    sel = v["mHs"] >= 30
    say(f"RD validation: max |Delta_num - Delta_asympt|*(m/H_*) for m/H_*>=30: "
        f"{np.max(np.abs(v['delta'] - delta_rd_analytic(v['ts']))[sel] * v['mHs'][sel]):.3e}")

    # Same envelope, but the phase at fixed m/H differs (baryons and the axion itself
    # make H(t) deviate from 1/(2t)), so compare envelopes bin by bin.
    for lo, hi in [(3, 10), (10, 30), (30, 100), (100, 300), (300, 1000)]:
        s_f = (mH_s >= lo) & (mH_s < hi)
        s_r = (v["mHs"] >= lo) & (v["mHs"] < hi)
        say(f"envelope max |Delta|(m/H_*) on [{lo},{hi}): full {np.max(np.abs(delta[s_f]) * mH_s[s_f]):.4f}"
            f"   pure RD {np.max(np.abs(v['delta'][s_r]) * v['mHs'][s_r]):.4f}")

    # --- figures & data --------------------------------------------------------
    fig_overview(os.path.join(args.outdir, "fig1_background.png"), tau_c, mH_c, y_c, late)
    fig_running(os.path.join(args.outdir, "fig2_running_difference.png"), mH_c, y_c, late)
    fig_matching(os.path.join(args.outdir, "fig3_matching_error.png"), mH_s, delta, tau_s,
                 v["mHs"], v["delta"], source)
    fig_validation(os.path.join(args.outdir, "fig4_validation_RD.png"), v)
    np.savez_compressed(os.path.join(args.outdir, "matching_error_data.npz"),
                        mH_star=mH_s, tau_star=tau_s, delta=delta,
                        tau=tau_c[::20], mH=mH_c[::20], y=y_c[:, ::20],
                        late_a3rho=late, A=A, m_eV=args.mass, background=source)
    with open(os.path.join(args.outdir, "run_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
        f.write(json.dumps(dict(params=PLANCK18, args=vars(args)), indent=1) + "\n")
    say(f"wrote figures and data to {args.outdir}")


if __name__ == "__main__":
    main()
