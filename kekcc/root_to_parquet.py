
import pathlib

import pandas
import uproot
import tqdm


def open_output_root_file(path, tree_names=["gen", "det"]):
    
    """
    Open an output root file as a pandas dataframe.

    Each tree will be labeled by a 
    pandas multi-index.
    """

    with uproot.open(path) as file:
        list_of_dataframes = [
            file[name].arrays(library="pd") 
            for name in tree_names
        ]

    final_dataframe = pandas.concat(list_of_dataframes, keys=tree_names)
    return final_dataframe


if __name__ == "__main__":

    data_dir = pathlib.Path("data/kekcc_output/2025-12-23/")
    list_of_data_file_paths = list(data_dir.glob("*.root"))

    for path in (pbar:=tqdm.tqdm(list_of_data_file_paths)):
        pbar.set_postfix_str(path.name)
        dataframe = open_output_root_file(path).drop(columns="__eventType__")
        save_path = path.parent.joinpath(f"{path.stem}.parquet")
        dataframe.to_parquet(save_path)


