#!/usr/bin/env python3

# No-prompt ISF fitting tool, generalised to N exponentials (N = 1, 2, 3, 4, ...)
#
# Model:  ISF(t) = A + (1 - A) * sum_i w_i * exp(-t / tau_i),   sum_i w_i = 1
#
# The weights are parameterised with "stick-breaking" fractions f_1 .. f_(N-1),
# each bounded in [0, 1]:
#     w_1 = f_1,  w_2 = (1 - f_1) f_2,  ...,  w_N = prod_k (1 - f_k)
# This guarantees w_i >= 0 and sum w_i = 1 for any N (for N = 2 it reduces to
# the original A + (1-A)(f e^{-t/tau1} + (1-f) e^{-t/tau2}) model).
#
# After fitting, components are sorted so that tau_1 < tau_2 < ... < tau_N.

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HBAR_MEV_PS = 0.6582119569  # meV·ps

# =========================================================
# Fixed user settings
# =========================================================
DEFAULT_FILENAME = "ISF.csv"
DEFAULT_DT_PS = 1          # change depending on how often the trajectory prints
DEFAULT_CUTOFF_PS = 0.05
Q_HEADER_ROW = None        # None = auto-detect; or force line 0 / 1 of the CSV to hold the Q values

N_EXP = 2                  # number of exponentials (Lorentzians): 1, 2, 3, 4, ...

# Initial tau guesses (ps), fastest to slowest. If N_EXP is larger than the
# length of this list, the guesses are log-spaced between its min and max.
TAU_GUESSES = [2.0, 70.0]

OUTPUT_PREFIX = f"ISF_gmx_{N_EXP}exp"

FULL_FIT_OUTPUT = f"{OUTPUT_PREFIX}_fit_results.csv"
SUMMARY_OUTPUT = f"{OUTPUT_PREFIX}_q_summary.csv"
GNUPLOT_EISF = f"{OUTPUT_PREFIX}_plot_eisf.gp"
GNUPLOT_HWHM_ALL = f"{OUTPUT_PREFIX}_plot_hwhm_all.gp"


def gnuplot_hwhm_name(i):
    return f"{OUTPUT_PREFIX}_plot_hwhm{i}.gp"


DIAG_DIR = f"{OUTPUT_PREFIX}_fit_diagnostics"
GNUPLOT_TERMINAL = "qt"  # change to wxt, x11, or dumb if needed


# =========================================================
# Model
# =========================================================
def stick_breaking_weights(f):
    """Convert N-1 fractions in [0,1] into N weights that sum to 1."""
    weights = []
    remaining = 1.0
    for fk in f:
        weights.append(remaining * fk)
        remaining *= (1.0 - fk)
    weights.append(remaining)
    return np.array(weights)


def unpack_params(params, n):
    """params = [A, f_1..f_(n-1), tau_1..tau_n]"""
    A = params[0]
    f = np.asarray(params[1:n])
    taus = np.asarray(params[n:2 * n])
    return A, f, taus


def multi_exp_normalised(t, *params):
    n = len(params) // 2
    A, f, taus = unpack_params(params, n)
    w = stick_breaking_weights(f)
    t = np.asarray(t, dtype=float)
    decays = np.exp(-t[:, None] / taus[None, :])  # (n_t, n)
    return A + (1.0 - A) * (decays @ w)


# =========================================================
# Helpers
# =========================================================
def q_to_key(q):
    return f"{q:.4f}".replace(".", "p")


def estimate_tail_mean(y, n_tail=20):
    n_tail = max(1, min(n_tail, len(y)))
    return float(np.mean(y[-n_tail:]))


def get_tau_guesses(n):
    guesses = list(TAU_GUESSES)
    if len(guesses) >= n:
        return guesses[:n]
    lo, hi = min(guesses), max(guesses)
    if hi <= lo:
        hi = lo * 100.0
    return list(np.geomspace(lo, hi, n))


def _row_to_q_list(row, ncols):
    """Parse cells 1..ncols-1 of a raw row as floats (None where not numeric)."""
    cells = list(row) + [""] * (ncols - len(row))
    out = []
    for c in cells[1:ncols]:
        try:
            out.append(float(str(c).strip()))
        except ValueError:
            out.append(None)
    return out


def _row_is_valid_q_row(qs):
    valid = [q for q in qs if q is not None]
    return len(valid) > 0 and len(set(valid)) == len(valid)


def load_csv(filename, dt_ps):
    """
    Layout assumed: line 0 = (title / Q values), line 1 = header, data from line 2.
    The Q values are taken from line Q_HEADER_ROW (0 or 1). If Q_HEADER_ROW is
    None, line 1 is tried first (original behaviour), then line 0; a line is
    only accepted if its numeric entries are all unique (so a header made of
    repeated labels is rejected rather than silently mangled).
    """
    with open(filename, newline="") as fh:
        reader = csv.reader(fh)
        rows = [next(reader), next(reader)]

    data_df = pd.read_csv(filename, header=None, skiprows=2)
    ncols = data_df.shape[1]

    candidates = [Q_HEADER_ROW] if Q_HEADER_ROW is not None else [1, 0]
    q_per_col = None
    for r in candidates:
        qs = _row_to_q_list(rows[r], ncols)
        if _row_is_valid_q_row(qs):
            q_per_col = qs
            print(f"\nUsing line {r} of {filename} for Q values.")
            break

    if q_per_col is None:
        raise ValueError(
            "Could not find a line of unique numeric Q values.\n"
            f"  line 0: {rows[0][:6]} ...\n  line 1: {rows[1][:6]} ...\n"
            "Set Q_HEADER_ROW to 0 or 1 (or fix the file)."
        )

    time_ps = data_df.iloc[:, 0].to_numpy(dtype=float) * dt_ps

    q_vals = []
    data = {}
    skipped = []
    for j, q in enumerate(q_per_col, start=1):
        if q is None:
            skipped.append(j)
            continue
        q_vals.append(q)
        data[q] = data_df.iloc[:, j].to_numpy(dtype=float)

    q_vals = np.array(sorted(q_vals))

    print(f"Loaded {len(q_vals)} Q columns from {filename}.")
    print(f"Q range: {q_vals[0]:.6f} .. {q_vals[-1]:.6f}")
    if skipped:
        print(f"Skipped {len(skipped)} non-numeric/blank columns (positions: {skipped})")

    return time_ps, q_vals, data


def write_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# =========================================================
# Fit routine (any N)
# =========================================================
def fit_nexp(time_ps, y, fit_mask, n):
    t = time_ps[fit_mask]
    yy = y[fit_mask]

    n_params = 2 * n
    if len(t) < max(5, n_params + 3):
        raise RuntimeError("Not enough points to fit.")

    A0 = min(max(estimate_tail_mean(y, 20), 0.0), 1.0)
    # equal initial weights -> stick-breaking fractions 1/n, 1/(n-1), ..., 1/2
    f0 = [1.0 / (n - k) for k in range(n - 1)]
    tau0 = get_tau_guesses(n)

    p0 = [A0] + f0 + tau0
    lower = [0.0] + [0.0] * (n - 1) + [1e-6] * n
    upper = [1.0] + [1.0] * (n - 1) + [1e6] * n

    popt, _ = curve_fit(
        multi_exp_normalised,
        t,
        yy,
        p0=p0,
        bounds=(lower, upper),
        maxfev=200000,
    )

    A, f, taus = unpack_params(popt, n)
    weights = (1.0 - A) * stick_breaking_weights(f)  # absolute weights of each exponential

    # sort components by tau (fast -> slow)
    order = np.argsort(taus)
    taus = taus[order]
    weights = weights[order]

    fit_y = multi_exp_normalised(t, *popt)
    rss = float(np.sum((yy - fit_y) ** 2))

    scalars = {"A": A, "rss": rss, "A_plus_weights": A + float(np.sum(weights))}
    curves = []
    for i, (w, tau) in enumerate(zip(weights, taus), start=1):
        scalars[f"weight_{i}"] = float(w)
        scalars[f"tau_{i}_ps"] = float(tau)
        scalars[f"HWHM_{i}_meV"] = HBAR_MEV_PS / float(tau)
        curves.append(w * np.exp(-t / tau))

    return {
        "scalars": scalars,
        "fit_total": fit_y,
        "exp_curves": curves,
        "eisf_line": np.full_like(t, A, dtype=float),
    }


# =========================================================
# Summary table builder
# =========================================================
def build_q_summary_table(results_df, n, isf_lastpoint_map):
    rows = []
    for _, r in results_df.iterrows():
        q = r["Q"]
        row = {"Q": q, "EISF": r["A"], "EISF_ISF": isf_lastpoint_map[q]}
        for i in range(1, n + 1):
            row[f"weight_{i}"] = r[f"weight_{i}"]
            row[f"tau_{i}_ps"] = r[f"tau_{i}_ps"]
            row[f"HWHM_{i}_meV"] = r[f"HWHM_{i}_meV"]
        row["A_plus_weights"] = r["A_plus_weights"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Q")


# Summary CSV column layout (1-indexed, as gnuplot sees it):
#   1 Q, 2 EISF, 3 EISF_ISF, then for component i (1-based):
#   weight_i = 4+3(i-1), tau_i = 5+3(i-1), HWHM_i = 6+3(i-1)
def hwhm_col(i):
    return 6 + 3 * (i - 1)


# =========================================================
# Gnuplot summary script writers
# =========================================================
def write_gnuplot_hwhm(path, summary_file, i):
    content = (
        f'set terminal {GNUPLOT_TERMINAL}\n'
        'set datafile separator ","\n'
        f'set title "HWHM{i}_vs_Q2"\n'
        'set xlabel "Q^2_(A^{-2})"\n'
        f'set ylabel "HWHM{i}_(meV)"\n'
        'set grid\n'
        f'plot "{summary_file}" using ($1*$1):{hwhm_col(i)} with linespoints title "HWHM{i}"\n'
        'pause -1\n'
    )
    write_text(path, content)


def write_gnuplot_hwhm_all(path, summary_file, n):
    header = (
        f'set terminal {GNUPLOT_TERMINAL}\n'
        'set datafile separator ","\n'
        'set title "All_HWHM_vs_Q2"\n'
        'set xlabel "Q^2_(A^{-2})"\n'
        'set ylabel "HWHM_(meV)"\n'
        'set grid\n'
        'plot \\\n'
    )
    lines = []
    for i in range(1, n + 1):
        sep = ', \\\n' if i < n else '\n'
        lines.append(
            f'    "{summary_file}" using ($1*$1):{hwhm_col(i)} with linespoints title "HWHM{i}"{sep}'
        )
    write_text(path, header + "".join(lines) + 'pause -1\n')


def write_gnuplot_eisf(path, summary_file):
    content = (
        f'set terminal {GNUPLOT_TERMINAL}\n'
        'set datafile separator ","\n'
        'set title "EISF_vs_Q"\n'
        'set xlabel "Q_(A^{-1})"\n'
        'set ylabel "EISF"\n'
        'set grid\n'
        f'plot "{summary_file}" using 1:2 with linespoints title "Fitted_EISF", \\\n'
        '     "" using 1:3 with linespoints title "ISF_last_point"\n'
        'pause -1\n'
    )
    write_text(path, content)


# =========================================================
# Diagnostic writers
# =========================================================
def exp_col_names(n):
    return [f"exp{i}" for i in range(1, n + 1)]


def write_q_diagnostic_csv(path, time_fit, raw_fit_region, res, n):
    data = {
        "time_ps": time_fit,
        "isf_data": raw_fit_region,
        "eisf": res["eisf_line"],
    }
    for name, curve in zip(exp_col_names(n), res["exp_curves"]):
        data[name] = curve
    data["total_fit"] = res["fit_total"]
    pd.DataFrame(data).to_csv(path, index=False)


def write_q_gnuplot(path, csv_name, q, cutoff_ps, res, n):
    s = res["scalars"]
    taus_str = "__".join(f"tau{i}_{s[f'tau_{i}_ps']:.4f}_ps" for i in range(1, n + 1))
    title = f"Q_{q:.4f}_A^-1__RSS_{s['rss']:.6g}__EISF_{s['A']:.4f}__{taus_str}"

    # CSV columns: 1 time, 2 data, 3 eisf, 4..(3+n) exp_i, (4+n) total_fit
    plot_parts = [
        f'"{csv_name}" using 1:2 with points pt 7 ps 0.6 title "ISF_data"',
        '"" using 1:3 with lines lw 2 title "EISF"',
    ]
    for i in range(1, n + 1):
        plot_parts.append(f'"" using 1:{3 + i} with lines lw 2 title "exp{i}"')
    plot_parts.append(f'"" using 1:{4 + n} with lines lw 3 title "total_fit"')

    content = (
        f'set terminal {GNUPLOT_TERMINAL}\n'
        'set datafile separator ","\n'
        f'set title "{title}"\n'
        'set xlabel "time_(ps)"\n'
        'set ylabel "ISF"\n'
        'set grid\n'
        f'set arrow 1 from {cutoff_ps}, graph 0 to {cutoff_ps}, graph 1 nohead dt 2 lw 1\n'
        'plot ' + ', \\\n     '.join(plot_parts) + '\n'
        'pause -1\n'
    )
    write_text(path, content)


def write_combined_totalfits_csv(path, combined_rows, n):
    cols = ["Q", "time_ps", "isf_data", "total_fit"] + exp_col_names(n) + ["eisf"]
    pd.DataFrame(combined_rows)[cols].to_csv(path, index=False)


def write_combined_data_and_fits_csv(path, combined_rows):
    pd.DataFrame(combined_rows)[["Q", "time_ps", "isf_data", "total_fit"]].to_csv(path, index=False)


def build_all_totalfits_gnuplot_content(csv_name, q_values):
    header = (
        f'set terminal {GNUPLOT_TERMINAL}\n'
        'set datafile separator ","\n'
        'set title "All_Q_total_fits"\n'
        'set xlabel "time_(ps)"\n'
        'set ylabel "ISF"\n'
        'set grid\n'
        'plot \\\n'
    )
    lines = []
    for i, q in enumerate(q_values):
        sep = ', \\\n' if i < len(q_values) - 1 else '\n'
        lines.append(
            f'    "{csv_name}" using 2:(abs($1-{q:.10f})<1e-9 ? $4 : 1/0) '
            f'with lines title "Q_{q:.4f}"{sep}'
        )
    return header + "".join(lines) + 'pause -1\n'


def build_all_data_and_fits_gnuplot_content(csv_name, q_values):
    header = (
        f'set terminal {GNUPLOT_TERMINAL}\n'
        'set datafile separator ","\n'
        'set title "All_Q_data_and_total_fits"\n'
        'set xlabel "time_(ps)"\n'
        'set ylabel "ISF"\n'
        'set grid\n'
        'plot \\\n'
    )
    lines = []
    nq = len(q_values)
    for i, q in enumerate(q_values):
        q_str = f"{q:.10f}"
        label = f"Q_{q:.4f}"
        lines.append(
            f'    "{csv_name}" using 2:(abs($1-{q_str})<1e-9 ? $3 : 1/0) '
            f'with points pt 7 ps 0.5 title "{label}_data", \\\n'
        )
        sep = ', \\\n' if i < nq - 1 else '\n'
        lines.append(
            f'    "{csv_name}" using 2:(abs($1-{q_str})<1e-9 ? $4 : 1/0) '
            f'with lines lw 2 title "{label}_fit"{sep}'
        )
    return header + "".join(lines) + 'pause -1\n'


# =========================================================
# Main
# =========================================================
def main():
    n = int(N_EXP)
    if n < 1:
        raise ValueError("N_EXP must be >= 1")

    print("\nISF_exponential_fitting_tool_no_prompts_(Python_3)\n")
    print(f"Using_input_file: {DEFAULT_FILENAME}")
    print(f"dt_ps={DEFAULT_DT_PS}, n_exp={n}, cutoff_ps={DEFAULT_CUTOFF_PS}")
    print(f"tau_guesses={get_tau_guesses(n)}, gnuplot_terminal={GNUPLOT_TERMINAL}")

    if not Path(DEFAULT_FILENAME).exists():
        print(f"File_not_found: {DEFAULT_FILENAME}")
        return

    time_ps, q_values, isf_data = load_csv(DEFAULT_FILENAME, DEFAULT_DT_PS)
    fit_mask = time_ps >= DEFAULT_CUTOFF_PS
    time_fit = time_ps[fit_mask]

    isf_lastpoint_map = {q: float(isf_data[q][-1]) for q in q_values}

    Path(DIAG_DIR).mkdir(parents=True, exist_ok=True)

    results = []
    combined_rows = []
    fitted_qs = []

    for q in q_values:
        y = isf_data[q]
        y_fit_region = y[fit_mask]
        q_key = q_to_key(q)

        try:
            res = fit_nexp(time_ps, y, fit_mask, n)
        except Exception as e:
            print(f"Fit_failed_for_Q_{q:.4f}: {e}")
            continue

        fitted_qs.append(q)
        row = {"Q": q, "EISF_ISF": isf_lastpoint_map[q]}
        row.update(res["scalars"])
        results.append(row)

        csv_name = f"Q_{q_key}_fit.csv"
        gp_name = f"Q_{q_key}_fit.gp"
        write_q_diagnostic_csv(os.path.join(DIAG_DIR, csv_name), time_fit, y_fit_region, res, n)
        write_q_gnuplot(os.path.join(DIAG_DIR, gp_name), csv_name, q, DEFAULT_CUTOFF_PS, res, n)

        for k in range(len(time_fit)):
            crow = {
                "Q": q,
                "time_ps": time_fit[k],
                "isf_data": y_fit_region[k],
                "eisf": res["eisf_line"][k],
                "total_fit": res["fit_total"][k],
            }
            for name, curve in zip(exp_col_names(n), res["exp_curves"]):
                crow[name] = curve[k]
            combined_rows.append(crow)

    if not results:
        print("No_fits_succeeded.")
        return

    df_out = pd.DataFrame(results).sort_values("Q")

    print("\nFull_fit_results:\n")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
        print(df_out.round(6))

    df_out.to_csv(FULL_FIT_OUTPUT, index=False)
    print(f"\nSaved_full_fit_table_to_{FULL_FIT_OUTPUT}")

    summary_df = build_q_summary_table(df_out, n, isf_lastpoint_map)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    print("\nSummary_table:\n")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
        print(summary_df.round(6))

    print(f"\nSaved_summary_table_to_{SUMMARY_OUTPUT}")

    # Summary gnuplot scripts
    for i in range(1, n + 1):
        write_gnuplot_hwhm(gnuplot_hwhm_name(i), SUMMARY_OUTPUT, i)
        print(f"Saved_gnuplot_script_to_{gnuplot_hwhm_name(i)}")
    if n > 1:
        write_gnuplot_hwhm_all(GNUPLOT_HWHM_ALL, SUMMARY_OUTPUT, n)
        print(f"Saved_gnuplot_script_to_{GNUPLOT_HWHM_ALL}")

    write_gnuplot_eisf(GNUPLOT_EISF, SUMMARY_OUTPUT)
    print(f"Saved_gnuplot_script_to_{GNUPLOT_EISF}")

    # Combined diagnostics
    combined_csv = os.path.join(DIAG_DIR, "all_Q_totalfits.csv")
    write_combined_totalfits_csv(combined_csv, combined_rows, n)
    combined_gp = os.path.join(DIAG_DIR, "plot_all_Q_totalfits.gp")
    write_text(combined_gp, build_all_totalfits_gnuplot_content("all_Q_totalfits.csv", fitted_qs))

    combined_data_fit_csv = os.path.join(DIAG_DIR, "all_Q_data_and_fits.csv")
    write_combined_data_and_fits_csv(combined_data_fit_csv, combined_rows)
    combined_data_fit_gp = os.path.join(DIAG_DIR, "plot_all_Q_data_and_fits.gp")
    write_text(
        combined_data_fit_gp,
        build_all_data_and_fits_gnuplot_content("all_Q_data_and_fits.csv", fitted_qs),
    )

    print(f"Saved_combined_total_fit_table_to_{combined_csv}")
    print(f"Saved_combined_total_fit_gnuplot_to_{combined_gp}")
    print(f"Saved_combined_data_and_fit_table_to_{combined_data_fit_csv}")
    print(f"Saved_combined_data_and_fit_gnuplot_to_{combined_data_fit_gp}")

    print(f"\nDiagnostic_files_written_to_directory_{DIAG_DIR}")
    print("Per_Q_files_include:")
    print("  Q_<value>_fit.csv")
    print("  Q_<value>_fit.gp")
    print("with_<value>_using_underscores_and_p_for_decimal_points")

    print("\nRun_summary_plots_with:")
    for i in range(1, n + 1):
        print(f"  gnuplot {gnuplot_hwhm_name(i)}")
    if n > 1:
        print(f"  gnuplot {GNUPLOT_HWHM_ALL}")
    print(f"  gnuplot {GNUPLOT_EISF}")

    print("\nRun_diagnostic_plots_with_examples:")
    example_q = q_to_key(fitted_qs[0])
    print(f"  gnuplot {DIAG_DIR}/Q_{example_q}_fit.gp")
    print(f"  gnuplot {DIAG_DIR}/plot_all_Q_totalfits.gp")
    print(f"  gnuplot {DIAG_DIR}/plot_all_Q_data_and_fits.gp")

    print("\nDone.")


if __name__ == "__main__":
    main()
