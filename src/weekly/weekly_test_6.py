import pandas as pd
sp500_data = pd.read_parquet('../../data/sp500.parquet', engine='fastparquet')
ff_factors_data = pd.read_parquet('../../data/ff_factors.parquet', engine='fastparquet')

merged_data = sp500_data.merge(ff_factors_data, on='Date', how='left')
merged_data['Excess Return'] = merged_data['Monthly Returns'] - merged_data['RF']
merged_data.sort_values(by=['Date'], inplace=True)
merged_data['ex_ret_1'] = merged_data.groupby('Symbol')['Excess Return'].shift(-1)
merged_data = merged_data.dropna(subset=['ex_ret_1'])
merged_data = merged_data.dropna(subset=['HML'])
amazon_data = merged_data[merged_data['Symbol'] == 'AMZN'].copy()
amazon_data.drop(columns=['Symbol'], inplace=True)
