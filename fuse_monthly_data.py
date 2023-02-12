#Python script for processing Bureau of Transportation Statistics Marketing Carrier On-Time Performance data.
#Adapted from Jupyter notebook with the same purpose.
#Link: https://www.transtats.bts.gov/
import os
import pandas as pd
import numpy as np
import re

#Dictionary to remap cancellation codes to single cancellation column that encodes if flight is cancelled along with cancellation reason
cancel_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, np.nan: 0}

#Get all files matching naming parameters
filenames = [_ for _ in os.listdir() if 'T_ONTIME_MARKETING' in _ and _.endswith('.csv')]
#Create a list of dataframes
dfs = []
for file in filenames:
    #For each monthly data file, read the data
    temp_df = pd.read_csv(file)
    #Ensure there are no non-cancelled flights with missing flight data
    temp_df = temp_df[~((temp_df['CANCELLED'] == 0) & (temp_df['AIR_TIME'].isna()))]
    #Apply cancellation mapping
    temp_df['CANCELLED'] = temp_df['CANCELLATION_CODE'].map(cancel_dict)
    #Remove now redundant cancellation code column
    temp_df.drop(['CANCELLATION_CODE'], axis=1, inplace=True)
    temp_df['TAIL_NUM'] = temp_df['TAIL_NUM'].str.replace(r'^(\d{3}NV)', r'N\1', regex=True)
    #Fill delay columns with 0
    temp_df['CARRIER_DELAY'].fillna(0, inplace=True)
    temp_df['WEATHER_DELAY'].fillna(0, inplace=True)
    temp_df['NAS_DELAY'].fillna(0, inplace=True)
    temp_df['SECURITY_DELAY'].fillna(0, inplace=True)
    temp_df['LATE_AIRCRAFT_DELAY'].fillna(0, inplace=True)
    #Fill any other np.nan values
    temp_df.fillna(0, axis=1, inplace=True)
    #Append cleaned dataframe to list
    dfs.append(temp_df)

#Concatenate the list of dataframes to create a single, large dataframe with all monthly data
output = pd.concat(dfs)
#Sort values by date, ascending
output.sort_values(['YEAR', 'MONTH', 'DAY_OF_MONTH'], inplace=True)

#Output the file in various formats
output.to_csv('AirlineData.csv', index=False)
output.to_pickle('AirlineData.pkl')
output.to_parquet('AirlineData.parquet.gzip', compression='gzip')