import pandas as pd
from pathlib import Path
file_to_load = Path.cwd().parent.joinpath('data').joinpath('chipotle.tsv')
food = pd.read_csv("../../data/chipotle.tsv",sep='\t')

def change_price_to_float(input_df):
    input_df['item_price'] = input_df['item_price'].str.replace('$', '').astype(float)
    return input_df

def number_of_observations(input_df):
    new_df = input_df.copy()
    return len(new_df)

def items_and_prices(input_df):
    input_df = input_df[['item_name', 'item_price']]
    return input_df


def sorted_by_price(item_price):
    new_df = item_price.sort_values(by='item_price', ascending=False)
    return new_df

def avg_price(input_df):
    new_df = input_df.copy()
    return new_df['item_price'].mean()


def unique_items_over_ten_dollars(input_df):
    new_df = input_df.copy()
    high_cost_items = new_df[new_df['item_price'] > 10]
    return high_cost_items[['item_name', 'choice_description', 'item_price']].drop_duplicates()


def items_starting_with_s(input_df):
    new_df = input_df['item_name'][input_df['item_name'].str.startswith('S')]
    return new_df

def first_three_columns(input_df):
    new_df = input_df.iloc[:, :3]
    return new_df


def every_column_except_last_two(input_df):
    new_df =input_df.iloc[:, :-2]
    return new_df


def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    filtered_df = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    return filtered_df[columns_to_keep]


def generate_quartile(input_df):
    def quartile(pre):
        if 30 < pre:
            return "premium"
        elif 20 <= pre <= 29.99:
            return "high-cost"
        elif 10 <= pre <= 19.99:
            return "medium-cost"
        elif 0 <= pre <=  9.99:
            return 'low-cost'
    input_df['Quartile'] = input_df['item_price'].apply(quartile)
    return input_df


def average_price_in_quartiles(input_df):
    return input_df.groupby('Quartile')['item_price'].mean().reset_index()


def minmaxmean_price_in_quartile(input_df):
    return input_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean']).reset_index()


import random
def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory, seed=42):
    random.seed(seed)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = [distribution() for _ in range(length_of_trajectory)]
        cumulative_average = [sum(trajectory[:i + 1]) / (i + 1) for i in range(length_of_trajectory)]
        result.append(cumulative_average)
    return result


def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_mean = 0.0
        for _ in range(length_of_trajectory):
            value = distribution.gen_random()
            cumulative_mean += value
            trajectory.append(cumulative_mean / (len(trajectory) + 1))
        result.append(trajectory)
    return result

def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_mean = 0.0
        for _ in range(length_of_trajectory):
            value = distribution.generate_random()
            cumulative_mean += value
            trajectory.append(cumulative_mean / (len(trajectory) + 1))
        result.append(trajectory)
    return result

def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_mean = 0.0
        for _ in range(length_of_trajectory):
            value = distribution.gen_rand()
            cumulative_mean += value
            trajectory.append(cumulative_mean / (len(trajectory) + 1))
        result.append(trajectory)
    return result