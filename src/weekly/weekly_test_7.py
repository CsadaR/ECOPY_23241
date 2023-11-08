import pandas as pd
sp500_df = pd.read_parquet('../../data/sp500.parquet', engine='fastparquet')
ff_factors_df = pd.read_parquet('../../data/ff_factors.parquet', engine='fastparquet')
merged_df = pd.merge(sp500_df, ff_factors_df, on='Date', how='left')
merged_df['Excess Return'] = merged_df['Monthly Returns'] - merged_df['RF']
merged_df = merged_df.sort_values(by='Date')
merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)
merged_df = merged_df.dropna(subset=['ex_ret_1'])
merged_df = merged_df.dropna(subset=['HML'])

