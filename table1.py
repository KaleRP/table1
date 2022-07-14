import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import pydaisi as pyd


def run_table1(fpath: str=None, columns: List[str]=None, generate_data=False, num_rows: int = 10):
    """
    Runs the table1 generator and creates a dataframe for visualization
    :param fpath: Path to CSV containing data
    :param columns: Relevant columns specified, or defaulted to all columns.
    :return: a Pandas Dataframe with summary statistics
    """
    fake_data_generator = pyd.Daisi('rpkale/faker_health_data')
    df = None
    if generate_data is False:
        if fpath is not None and len(fpath) > 0:
            df = pd.read_csv(fpath)
    else:
        df = fake_data_generator.gen_data(num_rows, columns)
    if columns is not None and isinstance(columns, str) and len(columns) > 0:
        columns = eval(columns)
    data = generate_table1_data(df, columns)
    df = dispay_table1(data)
    return df


def generate_table1_data(df: pd.DataFrame=None, columns: List[str]=None) -> List[Dict]:
    """
    This function generates the data for table 1 by categorizing relevant columns as numerical, binary, datetime, and
    categorical. Presently, only numerical and binary are used.
    :param df: A Pandas dataframe
    :param columns: Relevant columns specified, or defaulted to all columns.
    :return: List[Dict]
    """
    if df is None:
        df = pd.read_csv('study_data.csv')

    if columns is None or len(columns) == 0:
        columns = df.columns

    if len(columns) == 1:
        df = pd.DataFrame(df[columns]).reset_index()
    else:
        df = df[columns]
    # df = df.set_index(df.columns[0])


    # Determine datetime columns and convert
    mask = df.astype(str).apply(lambda x: x.str.match(r'\d{4}-\d{2}-\d{2}').all())
    if not mask.empty:
        df.loc[:, mask] = df.loc[:, mask].apply(pd.to_datetime)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    binary_cols = [col for col in df.columns if len(df[col].unique()) == 2]
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # define categorical columns as unused columns with at most 75% unique values. Aka, don't include "Name" column
    rest_cols = categorical_cols = set(df.columns) - set(numeric_cols) - set(binary_cols) - set(datetime_cols)
    categorical_cols = [col for col in rest_cols if len(df[col].unique()) < (len(df) - (len(df) * 0.25))]

    sample_size = len(df)
    totals_dict = {"Totals": {'sample_size': sample_size}}

    numerical_mean_std_dict = {'Numerical': {col: {'mean': df[col].mean(), 'std': df[col].std()}
                                             for col in numeric_cols}}

    # Figure out which binary datapoint is seen most often for binary cols and get mean/std
    binary_mode_count_pct_dict = {}
    for col in binary_cols:
        binary_mode = df[col].mode().values[0]
        binary_mode_count = len(df[df[col] == binary_mode])
        binary_mode_pct = binary_mode_count / len(df)

        binary_mode_count_pct_dict[col] = {'mode': binary_mode, 'count': binary_mode_count, 'pct': binary_mode_pct}
        binary_mode_count_pct_dict = {'Binary': binary_mode_count_pct_dict}

    data = [totals_dict, numerical_mean_std_dict, binary_mode_count_pct_dict]
    return data

def dispay_table1(data: List[Dict]):
    """

    :param data:
    :return:
    """
    df = pd.DataFrame(columns=['Feature', 'Overall', 'Distribution'])
    for data_dict in data:
        for data_type, features_dict in data_dict.items():
            if data_type == 'Totals':
                feature = 'Sample Size: n'
                overall = features_dict['sample_size']
                distribution = None
                df = _concat_data(df, feature, overall, distribution)

            if data_type == 'Numerical':
                for col, values_dict in features_dict.items():
                    feature = f'{col}: mean (std)'
                    overall = values_dict['mean']
                    distribution = values_dict['std']
                    df = _concat_data(df, feature, overall, distribution)

            if data_type == 'Binary':
                for col, values_dict in features_dict.items():
                    feature = f'{col} =  "{values_dict["mode"]}": count (%)'
                    overall = values_dict["count"]
                    distribution = values_dict['pct']
                    df = _concat_data(df, feature, overall, distribution)
    return df

def _concat_data(df: pd.DataFrame, feature: str, overall: float, distribution: Optional[float]):
    df_data = {}
    df_data['Feature'] = [feature]
    df_data['Overall'] = [overall]
    df_data['Distribution'] = [distribution]
    df2 = pd.DataFrame(df_data)
    return pd.concat([df, df2])

if __name__ == '__main__':
    df = run_table1(None, ['Age', 'BMI'])
    print(df)