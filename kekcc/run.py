
### cd to this directory (kekcc) to run ###

import subprocess
from pathlib import Path
from itertools import product

from pandas import Series, read_parquet

### Helpers ###

def make_dec_file(e_or_mu, dc7, dc9, dc10, file_path):

    assert e_or_mu in ("e", "mu")

    content = f"""
    Alias MyB0 B0
    Alias MyAntiB0 anti-B0
    ChargeConj MyB0 MyAntiB0

    Alias MyK*0 K*0
    Alias MyAnti-K*0 anti-K*0
    ChargeConj MyK*0 MyAnti-K*0

    Decay Upsilon(4S)
    0.500  MyB0 anti-B0    VSS;
    0.500  B0 MyAntiB0    VSS;
    Enddecay

    Decay MyB0
    1.000 MyK*0 {e_or_mu}+ {e_or_mu}- BTOSLLNPR 0 0 {dc7} 0 1 {dc9} 0 2 {dc10} 0;
    Enddecay

    CDecay MyAntiB0

    Decay MyK*0
    1.000 K+ pi-   VSS;
    Enddecay

    CDecay MyAnti-K*0

    End
    """

    with open(file_path, "w") as f:
        f.write(content)


def make_file_paths(trial, sub_trial, dc7, dc9, dc10, path_to_output_dir):

    file_name_base = f"trial_{trial}_{sub_trial}_dc7_{dc7:.2f}_dc9_{dc9:.2f}_dc10_{dc10:.2f}"
    file_names = {
        "metadata" : f"{file_name_base}.json",
        "sim_output" : f"{file_name_base}.root",
        "recon_output" : f"{file_name_base}_re.root",
        "dec" : f"{file_name_base}.dec",
        "log" : f"{file_name_base}.log",
    }
    file_paths = {
        kind : path_to_output_dir.joinpath(name) 
        for kind, name in file_names.items()
    }
    return file_paths


def get_wilson_coefficients_series(sampled_wilson_coefficients_dataframe, trial):

    wilson_coefficients_series = sampled_wilson_coefficients_dataframe.loc[trial]
    return wilson_coefficients_series


if __name__ == "__main__":

    ### Parameters ###
    e_or_mu = "mu"
    trial_range = range(-1, 0) # each trial corresponds with a wilson coefficient sample
    sub_trial_range = range(0, 2) # split up large jobs (repeats per trial)
    events_per_sub_trial = 25_000
    # path_to_wilson_coefficient_samples = Path("../data/sampled_wilson_coefficients.parquet")
    path_to_output_dir = Path("../data/kekcc_output/")
    ###################

    # sampled_wilson_coefficients_dataframe = read_parquet(path_to_wilson_coefficient_samples)

    for trial, sub_trial in product(trial_range, sub_trial_range):

        metadata = Series({"dc7":0.0, "dc9":0.0, "dc10":0.0}) # get_wilson_coefficients_series(sampled_wilson_coefficients_dataframe, trial)
        metadata["trial"] = trial
        metadata["sub_trial"] = sub_trial
        metadata["channel"] = e_or_mu
        metadata["num_events"] = events_per_sub_trial

        file_paths = make_file_paths(
            trial=trial,
            sub_trial=sub_trial,
            dc7=metadata["dc7"], 
            dc9=metadata["dc9"], 
            dc10=metadata["dc10"], 
            path_to_output_dir=path_to_output_dir
        )

        metadata.to_json(file_paths["metadata"])
        make_dec_file(
            e_or_mu=e_or_mu,
            dc7=metadata["dc7"], 
            dc9=metadata["dc9"], 
            dc10=metadata["dc10"], 
            file_path=file_paths["dec"]
        )
        subprocess.run(
        f'bsub -q l "basf2 steer_sim.py -- {file_paths["dec"]} {file_paths["sim_output"]} {events_per_sub_trial} &>> {file_paths["log"]}'
            f' && basf2 steer_recon.py {e_or_mu} {file_paths["sim_output"]} {file_paths["recon_output"]} &>> {file_paths["log"]}'
            f' && rm {file_paths["sim_output"]}"',
            shell=True,
        )