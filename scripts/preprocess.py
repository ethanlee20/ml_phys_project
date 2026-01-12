
import pathlib

import tqdm
import pandas

from lib_sbi_btokstll.util import safer_convert_to_int
from lib_sbi_btokstll.data import get_split, open_output_root_file
from lib_sbi_btokstll.calc import calculate_B_to_K_star_ell_ell_variables


def root_to_parquet(path_to_data_dir):
    
    data_dir = pathlib.Path(path_to_data_dir)
    list_of_data_file_paths = list(data_dir.rglob("*.root"))

    for path in (pbar:=tqdm.tqdm(list_of_data_file_paths, desc="Converting root to parquet")):
        pbar.set_postfix_str(path.name)
        dataframe = open_output_root_file(path).drop(columns="__eventType__")
        save_path = path.parent.joinpath(f"{path.stem}.parquet")
        dataframe.to_parquet(save_path)


def combine_files(path_to_data_dir):

    path_to_data_dir = pathlib.Path(path_to_data_dir)
    metadata_file_paths = list(path_to_data_dir.rglob("*.json"))

    list_of_dataframes = []
    list_of_keys = []

    for path_to_metadata in (pbar:=tqdm.tqdm(metadata_file_paths, desc="Combining parquet files")):

        pbar.set_postfix_str(path_to_metadata.name)

        metadata = pandas.read_json(path_to_metadata, typ="series")
        path_to_data_file = path_to_metadata.parent.joinpath(f"{path_to_metadata.stem}_re.parquet")
        data = pandas.read_parquet(path_to_data_file)

        data = data.assign(**metadata[["channel", "dc7", "dc9", "dc10"]])

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
    convert_root_to_parquet = False

    if convert_root_to_parquet:
        root_to_parquet(path_to_data_dir)
    data = combine_files(path_to_data_dir)
    print("Calculating variables...")
    data = calculate_B_to_K_star_ell_ell_variables(data, ell=lepton_flavor)
    print(f"Saving data to {path_to_save}")
    data.to_parquet(path_to_save)
        

        