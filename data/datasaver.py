"""
Small pipeline to save temp files easier instead of pandas build in solutions (.to_csv())
"""
# Imports
from pathlib import Path

from pandas import DataFrame

# -- METHODS --
def save_dataframe(dataframe: DataFrame, filename: str, destination: str = "output", **kwargs) -> Path:
    out_dir = Path(destination)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / filename
    dataframe.to_csv(file_path, **kwargs)
    #print(f"Saved to '{file_path}'")

    return file_path