
import pandas as pd
import random
import matplotlib.pyplot as plt
euro12 = pd.read_csv("data/Euro_2012_stats_TEAM.csv")


def number_of_participants(input_df):
    new_df = input_df.copy()
    return len(new_df)


def goals(input_df):
    new_df = input_df.copy()
    return new_df[['Team', 'Goals']]


def sorted_by_goal(input_df):
    new_df = input_df.copy()
    return new_df.sort_values(by='Goals', ascending=False)


def avg_goal(input_df):
    new_df = input_df.copy()
    return new_df['Goals'].mean()


def countries_over_five(input_df):
    new_df = input_df.copy()
    return new_df[new_df['Goals'] >= 6]


def countries_starting_with_g(input_df):
    return input_df[input_df['Team'].str.startswith('G')]


def first_seven_columns(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:, :7]


def every_column_except_last_three(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:, :-3]


def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    filtered_df = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    return filtered_df[columns_to_keep]


def generate_quartile(input_df):
    new_df = input_df.copy()

    def quartile(goal):
        if 6 <= goal <= 12:
            return 1
        elif goal == 5:
            return 2
        elif 3 <= goal <= 4:
            return 3
        else:
            return 4

    new_df['Quartile'] = new_df['Goals'].apply(quartile)
    return new_df


def average_yellow_in_quartiles(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['Yellow Cards'].mean()


def minmax_block_in_quartile(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['Blocks'].agg(['min', 'max'])


def scatter_goals_shots(input_df):
    new_df = input_df.copy()
    plt.figure()
    plt.scatter(new_df['Goals'], new_df['Shots on target'])
    plt.title('Goals and Shot on target')
    plt.xlabel('Goals')
    plt.ylabel('Shots on target')
    return plt


def scatter_goals_shots_by_quartile(input_df):
    new_df = input_df.copy()
    plt.figure()
    for quartile, data in new_df.groupby('Quartile'):
        plt.scatter(data['Goals'], data['Shots on target'], label=f'Quartile {quartile}')
    plt.title('Goals and Shot on target')
    plt.xlabel('Goals')
    plt.ylabel('Shots on target')
    plt.legend(title='Quartiles')
    return plt


def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for _ in range(number_of_trajectories):
        trajectory = [pareto_distribution(1, 1) for _ in range(length_of_trajectory)]
        cumulative_means = [sum(trajectory[:i + 1]) / (i + 1) for i in range(length_of_trajectory)]
        trajectories.append(cumulative_means)
    return trajectories
