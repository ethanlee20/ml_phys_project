
import pathlib

import tqdm
import pandas

from lib_sbi_btokstll.util import safer_convert_to_int
from lib_sbi_btokstll.data import get_split, open_output_root_file
from lib_sbi_btokstll.calc import calculate_B_to_K_star_ell_ell_variables


def root_to_parquet(path_to_root_file):
    
    path_to_root_file = pathlib.Path(path_to_root_file)
    if not path_to_root_file.is_file():
        raise FileNotFoundError(f"File not found: {path_to_root_file}")
    dataframe = open_output_root_file(path_to_root_file).drop(columns="__eventType__")
    save_path = path_to_root_file.with_suffix(".parquet")
    dataframe.to_parquet(save_path)


def combine_files(path_to_data_dir):

    path_to_data_dir = pathlib.Path(path_to_data_dir)
    metadata_file_paths = list(path_to_data_dir.rglob("*.json"))

    list_of_dataframes = []
    list_of_keys = []

    for path_to_metadata in (pbar:=tqdm.tqdm(metadata_file_paths, desc="Combining files")):

        pbar.set_postfix_str(path_to_metadata.name)

        metadata = pandas.read_json(path_to_metadata, typ="series")

        path_to_parquet_file = path_to_metadata.with_name(f"{path_to_metadata.stem}_re.parquet")
        if not path_to_parquet_file.is_file():
            path_to_root_file = path_to_parquet_file.with_suffix(".root")
            root_to_parquet(path_to_root_file)
        data = pandas.read_parquet(path_to_parquet_file)

        data = data.assign(**metadata.drop(labels=["trial", "sub_trial", "num_events"]))

        trial = safer_convert_to_int(metadata["trial"])
        sub_trial = safer_convert_to_int(metadata["sub_trial"])
        split = get_split(trial)
        keys = (trial, sub_trial, split)

        list_of_dataframes.append(data)
        list_of_keys.append(keys)

    data = pandas.concat(
        list_of_dataframes, 
        keys=list_of_keys, 
        names=["trial", "sub_trial", "split"], 
        verify_integrity=True
    )
    data = data.sort_index()
    return data


if __name__ == "__main__":

    path_to_data_dir = "data/kekcc_output/"
    path_to_save = "data/preprocessed.parquet"
    lepton_flavor = "mu"

    data = combine_files(path_to_data_dir)
    print("Calculating variables...")
    data = calculate_B_to_K_star_ell_ell_variables(data, ell=lepton_flavor)
    print(f"Saving data to {path_to_save}")
    data.to_parquet(path_to_save)
        

        