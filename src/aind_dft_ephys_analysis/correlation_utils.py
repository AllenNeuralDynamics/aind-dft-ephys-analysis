from typing import List, Tuple, Union, Optional
import pandas as pd

from typing import List, Tuple, Union, Optional
import pandas as pd

from typing import List, Tuple, Union, Optional
import pandas as pd

from typing import List, Tuple, Union, Optional
import pandas as pd

def select_units_by_significance(
    ds: pd.DataFrame,
    *,
    pval_col: str = "simple_LR-QLearning_L2F1_softmax-reward-g0-s0-d0-pval",
    coef_col: str = "simple_LR-QLearning_L2F1_softmax-reward-g0-s0-d0-coef",
    time_window: str = "0.3_2",
    alpha: float = 0.05,
    brain_areas: Union[str, List[str]] = "MD",
    coef_col_sign: Union[str, List[str]] = ("positive", "negative"),
    output_columns: Optional[List[str]] = None,
    output_time_window: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Tuple], List[dict], Optional[List[dict]]]:
    """
    Select units that pass p-value, time window, brain region, and coefficient sign filters.
    Additionally allows extracting different output columns from different time windows
    for the same (session_id, unit_index) pairs.

    Parameters
    ----------
    ds : pd.DataFrame
        Input DataFrame containing model results.

    pval_col : str
        Column name for p-values used to evaluate significance.
        Rows are retained when ds[pval_col] < alpha.

    coef_col : str
        Column name for the coefficient whose sign is used to filter rows.

    time_window : str
        Time window label used for selecting significant units.
        Only rows where ds["time_window"] == time_window are considered for selection.

    alpha : float
        P-value threshold for determining significance.
        Rows are retained when ds[pval_col] < alpha.

    brain_areas : str or list of str
        Brain region(s) to include. If a string is provided, it is automatically
        converted into a list. Only rows whose ds["brain_region"] is in this list
        are retained.

    coef_col_sign : str or list of str
        Allowed coefficient signs for filtering rows:
            - "positive": retain rows with ds[coef_col] > 0
            - "negative": retain rows with ds[coef_col] < 0
            - ("positive", "negative"): retain all rows where ds[coef_col] != 0

        If a single string is provided, it is converted to a list internally.

    output_columns : list of str, optional
        List of column names to extract for the selected units.
        These columns may come from different time windows specified in
        `output_time_window`.

        If None, `output_dicts` will be returned as None.

    output_time_window : list of str, optional
        List of time window labels corresponding to `output_columns`.
        Must have the same length as `output_columns`.

        Interpretation:
            output_columns[i] is extracted from rows where
                ds["time_window"] == output_time_window[i]

        For each selected (session_id, unit_index), this function retrieves
        the value of each requested column from its specified time window.

    Returns
    -------
    selected : pd.DataFrame
        DataFrame containing selected rows with columns ["session_id", "unit_index"].
        These rows define the units that pass all significance filters.

    result_tuples : list of tuple
        List of (session_id, unit_index) tuples for the selected units.

    result_dicts : list of dict
        List of dictionaries, each containing:
            {
                "session_id": ...,
                "unit_index": ...
            }

    output_dicts : list of dict or None
        If `output_columns` is provided:
            A list of dictionaries, where each dictionary corresponds to one selected unit
            and contains:
                {
                    "session_id": ...,
                    "unit_index": ...,
                    output_columns[0]: value_from_output_time_window[0],
                    output_columns[1]: value_from_output_time_window[1],
                    ...
                }
        If `output_columns` is None:
            Returns None.
    """
    # Normalize brain_areas
    if isinstance(brain_areas, str):
        brain_areas = [brain_areas]

    # Normalize coef_col_sign
    if isinstance(coef_col_sign, str):
        coef_col_sign = [coef_col_sign]

    # ---------------------------------------------
    # Coefficient sign mask
    # ---------------------------------------------
    if "positive" in coef_col_sign and "negative" not in coef_col_sign:
        sign_mask = ds[coef_col] > 0
    elif "negative" in coef_col_sign and "positive" not in coef_col_sign:
        sign_mask = ds[coef_col] < 0
    elif "positive" in coef_col_sign and "negative" in coef_col_sign:
        sign_mask = (ds[coef_col] > 0) | (ds[coef_col] < 0)
    else:
        raise ValueError("coef_col_sign must contain 'positive', 'negative', or both.")

    # ---------------------------------------------
    # Selection mask
    # ---------------------------------------------
    mask = (
        (ds[pval_col] < alpha) &
        (ds["time_window"] == time_window) &
        (ds["brain_region"].isin(brain_areas)) &
        (sign_mask)
    )

    selected = ds.loc[mask, ["session_id", "unit_index"]].drop_duplicates()
    result_tuples = list(selected.itertuples(index=False, name=None))
    result_dicts = selected.to_dict(orient="records")

    # ---------------------------------------------
    # No additional output
    # ---------------------------------------------
    if output_columns is None:
        return selected, result_tuples, result_dicts, None

    # input checks
    if output_time_window is None:
        raise ValueError("output_time_window must be provided when output_columns is provided.")
    if len(output_columns) != len(output_time_window):
        raise ValueError("output_time_window length must match output_columns length.")

    # verify columns
    missing = [c for c in output_columns if c not in ds.columns]
    if missing:
        raise KeyError(f"The following columns are missing from DataFrame: {missing}")

    # Prepare output records
    out_records = { (sid, uid): {"session_id": sid, "unit_index": uid}
                    for sid, uid in result_tuples }

    # For each column fetch values from its specific time window
    for col, tw in zip(output_columns, output_time_window):
        temp = ds.loc[
            ds["time_window"] == tw,
            ["session_id", "unit_index", col]
        ]
        for _, row in temp.iterrows():
            key = (row["session_id"], row["unit_index"])
            if key in out_records:
                out_records[key][col] = row[col]

    output_dicts = list(out_records.values())

    return selected, result_tuples, result_dicts, output_dicts


def get_column_names(ds: pd.DataFrame) -> List[str]:
    """
    Return the column names of the input DataFrame.

    Parameters
    ----------
    ds : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list of str
        List of column names in the DataFrame.
    """
    return list(ds.columns)
