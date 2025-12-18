
### cd to this directory (kekcc) to run ###

import subprocess

from pandas import read_parquet

from lib_sbi_btokstll.constants import path_to_data_dir


### Parameters ###

trial_range = range(0, 10) # each trial corresponds with a wilson coefficient sample
events_per_trial = 1_000
path_to_wilson_coefficient_samples = path_to_data_dir.joinpath("sampled_wilson_coefficients.parquet")
path_to_output_dir = path_to_


### Simulate ###

sampled_wilson_coefficients_dataframe = read_parquet(path_to_wilson_coefficient_samples)

for trial in trial_range:

    wilson_coefficients_series = sampled_wilson_coefficients_dataframe.loc[trial]
    dc7 = wilson_coefficients_series["dc7"]
    dc9 = wilson_coefficients_series["dc9"]
    dc10 = wilson_coefficients_series["dc10"]

    round_to = 2
    dc7_rounded = round(dc7, round_to)
    dc9_rounded = round(dc9, round_to)
    dc10_rounded = round(dc10, round_to)

    file_name_base = f"trial_{trial}_dc7_{dc7_rounded}_dc9_{dc9_rounded}_dc10_{dc10_rounded}"
    file_name_metadata = f"{file_name_base}.json"
    file_name_sim = f"{file_name_base}.root"
    file_name_recon = f"{file_name_base}_re.root"
    file_name_dec = f"{file_name_base}.dec"
    file_name_log  = f"{file_name_base}.log"

    decay = f"""
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
    1.000 MyK*0 mu+ mu- BTOSLLNPR 0 0 {dc7} 0 1 {dc9} 0 2 {dc10} 0;
    Enddecay

    CDecay MyAntiB0

    Decay MyK*0
    1.000 K+ pi-   VSS;
    Enddecay

    CDecay MyAnti-K*0

    End
    """

    wilson_coefficients_series.to_json(file_name_metadata)

    with open(file_name_dec, "w") as f:
        f.write(decay)

    subprocess.run(
	f'bsub -q l "basf2 steer_sim.py -- {file_name_dec} {file_name_sim} {events_per_trial} &>> {file_name_log}'
        f' && basf2 steer_recon.py {file_name_sim} {file_name_recon} &>> {file_name_log}'
        f' && rm {file_name_sim}"',
        shell=True,
    )
