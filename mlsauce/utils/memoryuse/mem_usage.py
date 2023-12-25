import numpy as np
import copy
from tqdm import tqdm


# from https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65


def reduce_mem_usage(df, verbose=True, copy_input=True):
    if copy_input is True:
        df_res = copy.deepcopy(df)
    else:
        df_res = df

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df_res.memory_usage().sum() / 1024**2

    for idx, col in tqdm(enumerate(df_res.columns)):
        col_type = df_res[col].dtypes
        if col_type in numerics:
            c_min = df_res[col].min()
            c_max = df_res[col].max()
            if str(col_type)[:3] == "int":
                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    df_res[col] = df_res[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    df_res[col] = df_res[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    df_res[col] = df_res[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.int64).min
                    and c_max < np.iinfo(np.int64).max
                ):
                    df_res[col] = df_res[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df_res[col] = df_res[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df_res[col] = df_res[col].astype(np.float32)
                else:
                    df_res[col] = df_res[col].astype(np.float64)

    pbar.update(len(df_res.columns))

    end_mem = df_res.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df_res
