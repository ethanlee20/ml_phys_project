
from math import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas
import skimage


def label_to_color(dc7, dc9, dc10):

    dc7_to_L = lambda dc7: float((dc7 + 0.5) * 60 + 40)
    dc9_to_a = lambda dc9: float(dc9 * 40)
    dc10_to_b = lambda dc10: float(dc10 * 40)

    return dc7_to_L(dc7), dc9_to_a(dc9), dc10_to_b(dc10)


def one_row_dataframe_to_series(dataframe):

    assert len(dataframe) == 1
    return dataframe.iloc[0]


if __name__ == "__main__":

    data = pandas.read_parquet("data/combined_processed.parquet")
    trials = [-1] + list(range(100_007, 100_010))
    var_names = {"q_squared": r"$q^2$ [GeV$^2$]", "cos_theta_mu": r"$\cos\theta_\mu$", "cos_theta_k":r"$\cos\theta_K$", "chi": r"$\chi$"}
    hist_intervals = {"q_squared":(0, 20), "cos_theta_mu":(-1, 1), "cos_theta_k":(-1, 1), "chi":(0, 2*pi)}
    n_bins = 10

    alpha = 1
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.dpi": 400, 
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern",
        "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
    })
    fig, axs = plt.subplots(2,2, layout="constrained")

    for var, ax in zip(["q_squared", "cos_theta_mu", "cos_theta_k", "chi"], axs.flat):
        
        legend_title = r"\begin{tabularx}{4cm}{ >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X } $\delta C_7$ & $\delta C_9$ & $\delta C_{10}$ \\ \end{tabularx}"
        ax.hist([], label=legend_title, alpha=0)

        for trial in trials:

            trial_data = data.loc[trial, :, "gen"]

            label = one_row_dataframe_to_series(trial_data[["dc7", "dc9", "dc10"]].drop_duplicates())
            lab_color = label_to_color(**label)
            rgb_color = skimage.color.lab2rgb(lab_color)

            hist_label = r"\begin{tabularx}{4cm}{ >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X } " +  f"{label["dc7"]:+10.2f} & {label["dc9"]:+10.2f} & {label["dc10"]:+10.2f} \\ " +  r"\end{tabularx}" 
            zorder = 0 if trial != -1 else 5
            ax.hist(trial_data[var], bins=n_bins, range=hist_intervals[var], color=rgb_color, histtype="step", linewidth=1.5, label=hist_label, alpha=alpha, zorder=zorder)

        ax.set_xlabel(var_names[var], fontsize=15)
        if var == "chi": ax.set_xticks([0, pi/2, pi, 3*pi/2,  2*pi], [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        
    axs.flat[3].legend(ncols=1)

    plt.savefig("data/plots/distribution.png", bbox_inches="tight")


