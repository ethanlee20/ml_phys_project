
from pathlib import Path

import pandas


path_to_data_dir = Path("data/kekcc_output/2026-01-02")
paths_to_metadata_files = list(path_to_data_dir.glob("*.json"))

for path_to_metadata in paths_to_metadata_files:

    metadata = pandas.read_json(path_to_metadata, typ="series")

    metadata["interval_dc7_lb"] = -0.5
    metadata["interval_dc7_ub"] = 0.5
    metadata["interval_dc9_lb"] = -2.0
    metadata["interval_dc9_ub"] = 1.0
    metadata["interval_dc10_lb"] = -1.0
    metadata["interval_dc10_ub"] = 1.0

    metadata.to_json(path_to_metadata)