
from math import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas
import skimage


if __name__ == "__main__":

    data = pandas.read_parquet("data/combined_processed.parquet")
    trials = range(100_007, 100_010)

    alpha = 1
    plt.style.use("dark_background")

    plt.rcParams.update({
        "figure.dpi": 400, 
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern",
        "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
    })

    unique_labels = data.loc[trials[0]:trials[-1], :, "gen"][["dc7", "dc9", "dc10"]].drop_duplicates()
    print(unique_labels)

    dc7_to_L = lambda dc7: (dc7 + 0.5) * 60 + 40
    dc9_to_a = lambda dc9: dc9 * 40
    dc10_to_b = lambda dc10: dc10 * 40

    lab_colors = list(zip(dc7_to_L(unique_labels["dc7"]).astype(float), dc9_to_a(unique_labels["dc9"]).astype(float), dc10_to_b(unique_labels["dc10"]).astype(float)))
    print(lab_colors)
    rgb_colors = skimage.color.lab2rgb(lab_colors)
    print(rgb_colors)

    var_names = {"q_squared": r"$q^2$ [GeV$^2$]", "cos_theta_mu": r"$\cos\theta_\mu$", "cos_theta_k":r"$\cos\theta_K$", "chi": r"$\chi$"}

    fig, axs = plt.subplots(2,2, layout="constrained")

    n_bins = 10
    hist_intervals = {"q_squared":(0, 20), "cos_theta_mu":(-1, 1), "cos_theta_k":(-1, 1), "chi":(0, 2*pi)}

    # $\delta C_7$ & $\delta C_9$ & $\delta C_{10}$ \\

    for var, ax in zip(["q_squared", "cos_theta_mu", "cos_theta_k", "chi"], axs.flat):

        ax.hist([], label=r"\begin{tabularx}{4cm}{ >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X } $\delta C_7$ & $\delta C_9$ & $\delta C_{10}$ \\ \end{tabularx}", alpha=0)
        
        for trial, color, labels in zip(trials, rgb_colors, unique_labels.iterrows()):
            _, labels = labels
            hist_label = r"\begin{tabularx}{4cm}{ >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X } " +  f"{labels["dc7"]:+10.2f} & {labels["dc9"]:+10.2f} & {labels["dc10"]:+10.2f} \\ " +  r"\end{tabularx}" 
            # hist_label = table #f"{r"$\delta C_7$: ":50}" + f"{labels["dc7"]:>+5.2f}\n" + f"{r"$\delta C_7$: ":<20}" + f"{labels["dc9"]:>+5.2f}\n" + f"{r"$\delta C_{10}$: ":<20}" + f"{labels["dc10"]:>+5.2f}"
            ax.hist(data.loc[trial, :, "gen"][var], bins=10, range=hist_intervals[var], color=color, histtype="step", linewidth=1.5, label=hist_label, alpha=alpha)

        ax.set_xlabel(var_names[var], fontsize=15)

    axs.flat[3].legend(ncols=1)

    plt.savefig("data/plots/distribution.png", bbox_inches="tight")


