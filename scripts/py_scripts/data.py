import pandas as pd
from pathlib import Path




ROOT_PATH: Path = Path(__file__).resolve().parents[2]

CACHED_DF_PATH: Path = ROOT_PATH / "datasets_dir"

CACHED_DF_PATH.mkdir(exist_ok = True)


def change_schema(
    df: pd.DataFrame, orig_col_names: list[str], col_names: list[str]
) -> None:
    if len(orig_col_names) != len(col_names):
        raise ValueError("orig_col_names and col_names must have the same length")
    if any(not name for name in orig_col_names):
        raise ValueError("orig_col_names cannot contain empty names")

    original_columns = [str(column) for column in df.columns]
    renamed_columns = list(df.columns)

    for orig_name, new_name in zip(orig_col_names, col_names):
        matching_indices = [
            index
            for index, column_name in enumerate(original_columns)
            if column_name.find(orig_name) != -1
        ]
        if not matching_indices:
            raise ValueError(f"No dataframe columns match {orig_name!r}")

        for index in matching_indices:
            renamed_columns[index] = new_name

    df.columns = renamed_columns



def preprocess(df) -> pd.DataFrame: 

    return (
        df
        .loc[:,["prompt","type"]]
        .dropna(subset = ["prompt"])
        .assign(prompt = lambda x: x["prompt"].str.strip())
        .drop_duplicates(subset = ["prompt", "type"])
        .reset_index(drop = True)
    )



def save_or_load_df_at_url(cache_file_name: str, url):

    file_path: Path = (CACHED_DF_PATH / cache_file_name)

    if file_path.exists():

        if file_path.suffix != ".csv":
            raise RuntimeError("Only .csv file extension is supported for caching dataframes.")
        
        df = pd.read_csv(file_path)

        change_schema(df, ["prompt", "type"], ["prompt", "type"])

        df.to_csv(file_path, index=False)

    else:
        
        df = pd.read_csv(url)

    

    return df



DF_XTEST_PROMPTS = save_or_load_df_at_url("xtest_df.csv", (
    "https://raw.githubusercontent.com/"
    "paul-rottger/xstest/main/xstest_prompts.csv"
))
    

df_name_config = {
    "xtest_df": DF_XTEST_PROMPTS
}



def prompts_at_type_and_category(df_name: str, prompt_type: str, number: int = None) -> list[str]:

    df = df_name_config[df_name]

    if type not in df.unique():
        raise RuntimeError(
            f"Unsupported prompt type. Choose one of "
            f"{df['type'].unique().tolist()}"
        )


    num_prompts_at_type = (df["type"] == type).sum()

    if number < 0 or number > num_prompts_at_type:
        raise ValueError(
            f"number must be between 0 and {num_prompts_at_type} "
            f"for prompt type {type!r}"
        )

    elif (number == None):

        number = num_prompts_at_type

    return (
        df.loc[DF_XTEST_PROMPTS["type"] == type, "prompt"]
        .iloc[:number]
        .tolist()
    )
