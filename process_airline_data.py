#Python script for processing Bureau of Transportation Statistics Marketing Carrier On-Time Performance data.
#Adapted from Jupyter notebook with the same purpose.
#Use fields.txt for list of fields selected.
#Link: https://www.transtats.bts.gov/
import pandas as pd
import numpy as np
import pickle
import datetime

#Read data from csv, parse flight date column as np.datetime
df1 = pd.read_csv('./data/2022Data.csv', parse_dates=['FL_DATE'])
print('Data Read.')

#Set "hhmm" data as str type for parsing, others as int
df1['CRS_DEP_TIME'] = df1['CRS_DEP_TIME'].astype(int).astype(str)
df1['DEP_TIME'] = df1['DEP_TIME'].astype(int).astype(str)
df1['DEP_DELAY'] = df1['DEP_DELAY'].astype(int)
df1['WHEELS_OFF'] = df1['WHEELS_OFF'].astype(int).astype(str)
df1['TAXI_OUT'] = df1['TAXI_OUT'].astype(int)

df1['CRS_ARR_TIME'] = df1['CRS_ARR_TIME'].astype(int).astype(str)
df1['ARR_TIME'] = df1['ARR_TIME'].astype(int).astype(str)
df1['ARR_DELAY'] = df1['ARR_DELAY'].astype(int)
df1['WHEELS_ON'] = df1['WHEELS_ON'].astype(int).astype(str)
df1['TAXI_IN'] = df1['TAXI_IN'].astype(int)

df1['CRS_ELAPSED_TIME'] = df1['CRS_ELAPSED_TIME'].astype(int)
df1['ACTUAL_ELAPSED_TIME'] = df1['ACTUAL_ELAPSED_TIME'].astype(int)
df1['AIR_TIME'] = df1['AIR_TIME'].astype(int)
df1['FLIGHTS'] = df1['FLIGHTS'].astype(int)
df1['DISTANCE'] = df1['DISTANCE'].astype(int)

df1['DIVERTED'] = df1['DIVERTED'].astype(int)
df1['CARRIER_DELAY'] = df1['CARRIER_DELAY'].astype(int)
df1['WEATHER_DELAY'] = df1['WEATHER_DELAY'].astype(int)
df1['NAS_DELAY'] = df1['NAS_DELAY'].astype(int)
df1['SECURITY_DELAY'] = df1['SECURITY_DELAY'].astype(int)
df1['LATE_AIRCRAFT_DELAY'] = df1['LATE_AIRCRAFT_DELAY'].astype(int)

#Rearrange columns
df1 = df1[['DAY_OF_WEEK', 'FL_DATE',
       'MKT_UNIQUE_CARRIER', 'MKT_CARRIER_FL_NUM', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_FL_NUM',
       'TAIL_NUM', 'ORIGIN', 'DEST',
       'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF',
       'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'TAXI_IN', 'WHEELS_ON',
       'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'FLIGHTS',
       'DIVERTED','CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]
print('Types Converted.')

def parse_str_time(input_str):
    """Convert str of hhmm data to list of hours and minutes as int.

    Args:
        input_str (str): hhmm data.

    Returns:
        list of int: List of hours and minutes as int
    """
    if len(input_str) == 1:
        return ['0', input_str]
    elif len(input_str) == 2:
        return ['0', input_str]
    elif len(input_str) == 3:
        return [input_str[0], input_str[1:]]
    elif len(input_str) == 4:
        return [input_str[0:2], input_str[2:]]
    
def np_conv_time(row):
    """Convert hhmm data along with flight date data to create datetime64 data for departure/arrival times.

    Args:
        row (pandas Series): Row from dataframe

    Returns:
        pandas Series: Altered row from dataframe
    """
    date = row['FL_DATE']
    cols = ['CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF', 'CRS_ARR_TIME', 'ARR_TIME', 'WHEELS_ON']
    for col in cols:
        time_parts = parse_str_time(row[col])
        row[col] = date + np.timedelta64(time_parts[0], 'h') + np.timedelta64(time_parts[1], 'm')
    return row

#Apply the time conversion function to the dataframe
df1 = df1.apply(lambda x: np_conv_time(x), axis=1)
print('Dates Parsed.')

#Save data to csv
df1.to_csv('AirlineData.csv', index=False)
print('CSV Saved.')

#Pickle dataframe for faster loading
with open('./data/2022Data.pkl', 'wb') as f:
    pickle.dump(df1, f)
print('Pickle Saved.')

print('Processing Complete!')