
import pathlib

import tqdm
import pandas


if __name__ == "__main__":

    data_dir = pathlib.Path("data/kekcc_output/")
    metadata_file_paths = list(data_dir.rglob("*.json"))

    list_of_dataframes = []
    list_of_trial_tuples = []

    for path_to_metadata in (pbar:=tqdm.tqdm(metadata_file_paths, desc="Loading files")):

        pbar.set_postfix_str(path_to_metadata.name)

        metadata = pandas.read_json(path_to_metadata, typ="series")
        path_to_data_file = path_to_metadata.parent.joinpath(f"{path_to_metadata.stem}_re.parquet")
        data = pandas.read_parquet(path_to_data_file)

        data = data.assign(**metadata[["channel", "dc7", "dc9", "dc10"]])
        trial_tuple = (metadata["trial"], metadata["sub_trial"])

        list_of_dataframes.append(data)
        list_of_trial_tuples.append(trial_tuple)

    data = pandas.concat(
        list_of_dataframes, 
        keys=list_of_trial_tuples, 
        names=["trial", "sub_trial"], 
        verify_integrity=True
    )

    data = data.sort_index()

    data.to_parquet("data/combined.parquet")
        
        

        