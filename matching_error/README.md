# Matching error of the naive CDM switch (Passaglia & Hu 2022, Eq. 5)

## Run

```bash
cd Codes/matching_error
../../.venv/bin/python matching_error.py                 # CLASS background (classy)
../../.venv/bin/python matching_error.py --background analytic   # no CLASS
```

Options: `--mass` (eV, default 1e-22), `--mH-end` (stop of exact run, default 2e4),
`--mHs-max` (largest switch in the scan, default 1e3). Runtime about 1 min.

The figures in `output/` were made with `--background analytic`, because classy
was not available outside your Mac. Running the script with CLASS overwrites them.

## What it does

* CLASS gives `rho_rest = rho_tot - rho_cdm` (Planck 2018, one 0.06 eV neutrino). The axion replaces
  the CDM: `(H/m)^2 = rho_rest/m^2 + (phi'^2+phi^2)/2`, so it backreacts on the expansion.
* Units: `m[Mpc^-1] = m[eV] / 6.39e-30`. After that the code is dimensionless, with `tau = m t` and
  `rho_ax = m^2 (phi'^2+phi^2)/2` in CLASS units.
* Klein-Gordon: `phi'' + 3(H/m) phi' + phi = 0`, `(ln a)' = H/m`. Solver: `solve_ivp`, DOP853,
  `rtol=1e-11`, `atol=1e-16`, dense output. Starts at `m/H = 1e-2` on the radiation-era attractor.
* The initial amplitude is found by shooting until the late `<a^3 rho_ax>` equals CLASS's `rho_cdm` today.
* Matching error for a switch at `tau_*`: `Delta = a_*^3 rho(tau_*) / <a^3 rho>_late - 1`. A single
  exact run gives the whole scan over the switch time.

## Figures

1. `fig1_background` shows the field and `a^3 rho`, with the CDM branch after switching at `m/H = 3`.
2. `fig2_running_difference` shows the exact field against CDM evolved together. The difference is
   a frozen offset plus oscillations that decay like `(3/2)(m/H)^-1`.
3. `fig3_matching_error` shows `|Delta|` against `m/H_*`, with envelope `(3/2)(m/H_*)^-1`.
4. `fig4_validation_RD` compares the solver with the exact Bessel solution in pure radiation domination.

Analytic prediction in radiation domination: `Delta = (3/2)(m/H_*)^-1 sin(2 tau_* - 3 pi/4)`.
It comes from the J_{1/4} asymptotics.
