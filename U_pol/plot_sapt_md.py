import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def plot_induction_parity(
    csv_file,
    output_png="sapt_vs_md_parity.png",
    title="SAPT vs MD Induction Parity Plot",
    chemical_accuracy_kj=1.0,
    fontsize=18,
):
    """
    Reads a CSV with columns:
        SAPT_Induction, MD_Induction, DimerID
    in kJ/mol units, then plots a parity scatter:
      x = SAPT_Induction (kJ/mol)
      y = MD_Induction   (kJ/mol)
    Colors each point by DimerID, and shows a small histogram
    of errors (MD - SAPT) in the inset.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file with columns [SAPT_Induction, MD_Induction, DimerID].
    output_png : str
        Output image filename, defaults to 'sapt_vs_md_parity.png'.
    title : str
        Main title of the plot.
    chemical_accuracy_kj : float
        For demonstration, we draw a ± band around y=x. Adjust or remove as needed.
    fontsize : int
        Base font size for labels and text.
    """
    # 1) Read the CSV
    df = pd.read_csv(csv_file)
    mask = df["DimerID"] > 50
    df = df[mask]
    x_sapt = df["SAPT_Induction"].values  # (kJ/mol)
    y_md = df["MD_Induction"].values  # (kJ/mol)
    dimer_id = df["DimerID"].values

    # 2) Compute errors
    errors = y_md - x_sapt
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    r2 = np.corrcoef(x_sapt, y_md)[0, 1] ** 2
    max_error = np.max(np.abs(errors))

    # 3) Setup plot style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "xtick.labelsize": fontsize * 0.8,
            "ytick.labelsize": fontsize * 0.8,
        }
    )

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    border_width = 1.5
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(border_width)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
        direction="in",
        length=6,
        width=border_width,
        which="major",
        top=True,
        right=True,
    )
    ax.tick_params(
        direction="in",
        length=3,
        width=border_width,
        which="minor",
        top=True,
        right=True,
    )

    # 4) Axis labels & title
    ax.set_xlabel("SAPT Induction (kJ/mol)", fontsize=fontsize)
    ax.set_ylabel("MD Induction (kJ/mol)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # 5) Plot scatter. Color by DimerID
    sc = ax.scatter(
        x_sapt, y_md, c=dimer_id, cmap="viridis", alpha=0.8, s=40, edgecolors="none"
    )

    # colorbar labeled "dimer distance (ID)"
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("dimer distance (ID)", fontsize=fontsize * 0.9)

    # 6) Diagonal y = x
    min_val = min(x_sapt.min(), y_md.min())
    max_val = max(x_sapt.max(), y_md.max())
    line_vals = np.linspace(min_val, max_val, 200)
    ax.plot(line_vals, line_vals, "k--", lw=2, alpha=0.7)

    # Optionally draw ± band around y=x
    lower_band = line_vals - chemical_accuracy_kj
    upper_band = line_vals + chemical_accuracy_kj
    ax.fill_between(
        line_vals,
        lower_band,
        upper_band,
        color="orange",
        alpha=0.2,
        label=f"±{chemical_accuracy_kj} kJ/mol",
    )

    # ax.set_xlim(-0.003,0.8*max_val)
    # ax.set_ylim(-0.003,0.8*max_val)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # 7) Show stats text
    text_x = 0.05
    text_y = 0.95
    stat_text = (
        f"MAE: {mae:.5f} kJ/mol\n"
        f"RMSE: {rmse:.5f} kJ/mol\n"
        f"R²: {r2:.5f}\n"
        f"Max error: {max_error:.5f} kJ/mol"
    )
    ax.text(
        text_x,
        text_y,
        stat_text,
        transform=ax.transAxes,
        fontsize=fontsize * 0.8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # 8) Inset: error histogram
    left, bottom, width, height = 0.45, 0.15, 0.3, 0.2
    ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset.hist(errors, bins=40, color="steelblue", alpha=0.7, edgecolor="k")
    ax_inset.set_title("Error Distribution (MD - SAPT)", fontsize=fontsize * 0.7)
    ax_inset.tick_params(axis="both", which="major", labelsize=fontsize * 0.6)
    # Center range around zero
    half_range = max(abs(errors.min()), abs(errors.max()), chemical_accuracy_kj * 3)
    ax_inset.set_xlim(-half_range, half_range)
    ax_inset.set_yticks([])
    # ax_inset.set_xscale('symlog', linthresh=1e-2)

    ax_inset.axvline(0.0, color="k", linestyle="--", linewidth=1.5)

    # 9) Save
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Saved parity plot to {output_png}")


# Example usage:
if __name__ == "__main__":
    # For demonstration, just one CSV.
    # Replace with your actual "induction_data.csv" file & desired output name.
    csv_file = "induction_data.csv"
    output_png = "parity_plot_induction.png"
    plot_induction_parity(
        csv_file,
        output_png=output_png,
        title="SAPT vs MD Induction Energy Parity (kJ/mol)",
        chemical_accuracy_kj=0.00001,
        fontsize=18,
    )
