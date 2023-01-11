#Python script for processing Bureau of Transportation Statistics Marketing Carrier On-Time Performance data.
#Adapted from Jupyter notebook with the same purpose.
#Link: https://www.transtats.bts.gov/
import pandas as pd
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
import html5lib
import re

#Read data from csv, using only the Tail Number column
df1 = pd.read_csv('./data/2022Data.csv', usecols=['TAIL_NUM'])
print('Data Read.')

#Get unique list of aircraft by N-number
tails = set(df1['TAIL_NUM'].unique())

#Create dicts for aircraft year, model, manufacturer, engine, initialize to '0'
years = dict.fromkeys(tails, 0)
models = dict.fromkeys(tails, 0)
manufacturers = dict.fromkeys(tails, 0)
engines = dict.fromkeys(tails, 0)
print('Dicts Prepared.')

#Iterate through each aircraft, make a request to the FAA registry page for that aircraft
for tail in tails:
    soup = BeautifulSoup(requests.get(f'https://registry.faa.gov/AircraftInquiry/Search/NNumberResult?nNumberTxt={tail}').text, 'html5lib')
        
    #Check if the page is valid. Some pages are invalid (Registration moved, parked aircraft w/ registration held, exported aircraft etc)
    check_valid = soup.find('td', attrs={'data-label': 'Status'})
    
    if check_valid is not None:
        #If valid page, read the year, manufacturer, model and engine from the page, save them to the dict using the N-number as the key
        if check_valid.text == 'Valid':
            years[tail] = soup.find('td', attrs={'data-label': 'Mfr Year'}).text.strip()
            manufacturers[tail] = soup.find('td', attrs={'data-label': 'Manufacturer Name'}).text.strip()
            models[tail] = soup.find('td', attrs={'data-label': 'Model'}).text.strip()
            engines[tail] = soup.find('td', attrs={'data-label': 'Engine Manufacturer'}).text.strip() + ' ' + \
                            soup.find('td', attrs={'data-label': 'Engine Model'}).text.strip()
    else:
        #If not a valid page, store as error for manual processing
        years[tail] = 0
        manufacturers[tail] = 'ERROR FINDING REGISTRATION'
        models[tail] = 'ERROR'
        engines[tail] = 'ERROR'
print('Registrations Checked.')

#Create series from each dict
s0 = pd.Series(years)
s1 = pd.Series(manufacturers)
s2 = pd.Series(models)
s3 = pd.Series(engines)

#Create dataframe using the series, with tail as the index
frame = {'Year': s0, 'Manufacturer': s1, 'Model': s2, 'Engine': s3}
df2 = pd.DataFrame(frame)
df2.index.names = ['Tail']
df2.reset_index(inplace=True)
print('DataFrame Created.')

#Simply manufacturers to have maximum 5 unique values rather than 10+
simplified_manufacturer = {
 'AIRBUS': 'Airbus',
 'AIRBUS SAS': 'Airbus',
 'AIRBUS INDUSTRIE': 'Airbus',
 'AIRBUS CANADA LP': 'Airbus',
 'AIRBUS CANADA LTD PTNRSP': 'Airbus',
 'C SERIES AIRCRAFT LTD PTNRSP': 'Airbus',
 'BOMBARDIER INC': 'Bombardier',
 'BOEING': 'Boeing',
 'EMBRAER': 'Embraer',
 'EMBRAER S A': 'Embraer',
 'EMBRAER-EMPRESA BRASILEIRA DE': 'Embraer',
 'EMPRESA BRASILEIRA DE AERO S A': 'Embraer',
 'YABORA INDUSTRIA AERONAUTICA S': 'Embraer',
 'MCDONNELL DOUGLAS CORPORATION' : 'McDonnell Douglas',
 0: 'ERROR'
}
df2['Manufacturer'] = df2.apply(lambda x: simplified_manufacturer[x.Manufacturer] if x.Manufacturer in simplified_manufacturer else 'ERROR', axis=1)
print('Manufacturers Simplified.')

#Convert FAA model types to ICAO model types
#ICAO types as IATA includes types specific to winglet types
#Airbus: AXXX-(Series ID)(Engine ID)(Engine Version)(Neo/Freighter/Combi) Engine ID 5/7 for A320 family neo
#Boeing: https://en.wikipedia.org/wiki/List_of_Boeing_customer_codes
def convert_model_ICAO_type(row):
    """Convert FAA model type to ICAO model type.

    Args:
        row (pandas Series): Row from dataframe

    Returns:
        pandas Series: Altered row from dataframe
    """
    #Remove any extra whitespace
    model = str(row['Model']).strip()
    manufacturer = str(row['Manufacturer']).strip()
    #Convert based on known FAA model types to appropriate ICAO model type
    #Dash 8D, CRJs
    if model == 'DHC-8-402':
        ac_type = 'DH8D'
    elif model == 'CL-600-2B19':
        ac_type = 'CRJ2'
    elif model == 'CL-600-2C10':
        ac_type = 'CRJ7'
    elif model == 'CL-600-2C11':
        ac_type = 'CRJ7'
    elif model == 'CL-600-2D24':
        ac_type = 'CRJ9'
    elif model == 'CL-600-2E25':
        ac_type = 'CRJX'
    elif model in ['ERJ 170-200 LR', 'ERJ 170-200 STD']: #Specific E175s
        ac_type = 'E75L'
    elif manufacturer == 'Bombardier':
        bombardier_conv_dict = {
            'BD-500-1A10': 'BCS1',
            'BD-500-1A11': 'BCS3',
        }
        if model in bombardier_conv_dict: #C-Series
            ac_type = bombardier_conv_dict[model]
        else:
            ac_type = 'DHC' + chr(ord(model[6]) + 16) #Dash 8s
    elif manufacturer == 'Boeing':
        boeing_conv_dict = { #737 MAX, 777F and 787-10
            '737-7': 'B37M',
            '737-8': 'B38M',
            '737-9': 'B39M',
            '737-10': 'B3XM',
            '777F': 'B77L',
            '787-10': 'B78X'
        }
        if model in boeing_conv_dict: #If in list, convert
            ac_type = boeing_conv_dict[model]
        elif model[0:5] == '777-2' and model[-2:] == 'LR': #Other special type
            ac_type = 'B77L'
        elif model[0:5] == '777-F' or model[0:4] == '777F': #Other special type
            ac_type = 'B77L'
        elif model[0:5] == '777-3' and model[-2:] == 'ER': #Other special type
            ac_type = 'B77W'
        else:
            ac_type = 'B7' + model[1] + model[4] #Remainder of types can be converted using specific digits of model
    elif manufacturer == 'Embraer':
        ac_type = 'E' + model[4:7] #Convert using specific digits of model
        if ac_type == 'E140': #Unique type
            ac_type = 'E135'
    elif manufacturer == 'Airbus':
        if ('N' in model) and (model[6] in ['5','7']): #A320 family/A330 neo
            ac_type = 'A' + model[2:4] + 'N'
        elif 'A350-1' in model: #Unique type
            ac_type = 'A35K'
        elif 'A330B' in model: #Unique type
            ac_type = 'A30B'
        elif 'A330C' in model: #Unique type
            ac_type = 'A30B'
        elif model[0:4] in ['A318', 'A319', 'A320', 'A321']:
            ac_type = model[0:4]
        else:
            ac_type = model[0:3] + model[5] #Convert using specific digits of model
    elif manufacturer == 'McDonnell Douglas':
        ac_type = 'MD' + ''.join(re.findall(r'\d', model)) #Convert using specific digits of model
    else:
        ac_type = 'NOT FOUND'
    
    #List of acceptable, valid types
    allowed_types = ['DH8A', 'DH8B', 'DH8C', 'DH8D', #Dash 8
                     'CRJ2', 'CRJ7', 'CRJ9', 'E135', 'E145', 'E170', 'E75S', 'E75L', 'E190', 'E195', 'E290', 'E295', 'BCS1', 'BCS3', #Regional jet
                     'MD11', 'MD81', 'MD82', 'MD83', 'MD87', 'MD88', 'MD90', #MD
                     'B712', 'B721', 'B722', 'B732', 'B733', 'B734', 'B735', 'B736', 'B737', 'B738', 'B739', #Boeing narrow body
                     'B37M', 'B38M', 'B39M', 'B3XM', 'B752', 'B753', #Boeing narrow body cont
                     'B741', 'B742', 'B743', 'B744', 'B748', #Boeing wide body
                     'B762', 'B763', 'B764', 'B772', 'B77L', 'B773', 'B77W', 'B778', 'B779', 'B788', 'B789', 'B78X', #Boeing wide body cont
                     'A318', 'A319', 'A19N', 'A320', 'A20N', 'A321', 'A21N', #Airbus narrow body
                     'A306', 'A30B', 'A332', 'A333', 'A338', 'A339', 'A342', 'A343', 'A345', 'A346', 'A359', 'A35K', 'A388'] #Airbus wide body
    
    #List of narrowbody aircraft
    narrow_bodies = ['DH8A', 'DH8B', 'DH8C', 'DH8D','CRJ2', 'CRJ7', 'CRJ9', 'E135', 'E145', 'E170', 'E75S', 'E75L', 'E190', 'E195', 'E290', 'E295',
                     'BCS1', 'BCS3', 'MD81', 'MD82', 'MD83', 'MD87', 'MD88', 'MD90',
                     'B712', 'B721', 'B722', 'B732', 'B733', 'B734', 'B735', 'B736', 'B737', 'B738', 'B739',
                     'B37M', 'B38M', 'B39M', 'B3XM', 'B752', 'B753',
                     'A318', 'A319', 'A19N', 'A320', 'A20N', 'A321', 'A21N']
    #List of widebody aircraft
    wide_bodies = ['MD11', 'B741', 'B742', 'B743', 'B744', 'B748',
                   'B762', 'B763', 'B764', 'B772', 'B77L', 'B773', 'B77W', 'B778', 'B779', 'B788', 'B789', 'B78X',
                   'A306', 'A30B', 'A332', 'A333', 'A338', 'A339', 'A342', 'A343', 'A345', 'A346', 'A359', 'A35K', 'A388']
    #Determine if aircraft is narrowbody is widebody
    ac_is_narrow = 1 if ac_type in narrow_bodies else 0
    ac_is_wide = 1 if ac_type in wide_bodies else 0
    
    #List of short-range aircraft (Range < 2300 NM, approx based on common engine type)
    short_range_ac = ['DH8A', 'DH8B', 'DH8C', 'DH8D',
                      'CRJ2', 'CRJ7', 'CRJ9', 'E135', 'E145', 'E170', 'E75S', 'E75L', 'E190', 'E195', 
                      'MD81', 'MD82',
                      'B712', 'B721', 'B722', 'B733',
                      ]
    #List of medium-range aircraft (2300 NM < Range < 4000 NM, approx based on common engine type)
    medium_range_ac = ['E290', 'E295', 'BCS1', 'BCS3',
                       'MD83', 'MD87', 'MD88', 'MD90',
                       'B732', 'B734', 'B735', 'B736', 'B737', 'B738', 'B739', 'B37M', 'B38M', 'B39M', 'B3XM', 'B752', 'B753',
                       'A318', 'A319', 'A19N', 'A320', 'A20N', 'A321', 'A21N',
                       'A30B'
                       ]
    #List of long-range aircraft (Range > 4000 NM, approx based on common engine type)
    long_range_ac = ['MD11',
                     'B741', 'B742', 'B743', 'B744', 'B748',
                     'B762', 'B763', 'B764', 'B772', 'B77L', 'B773', 'B77W', 'B778', 'B779', 'B788', 'B789', 'B78X',
                     'A306', 'A332', 'A333', 'A338', 'A339', 'A342', 'A343', 'A345', 'A346', 'A359', 'A35K', 'A388'
                     ]
    #Determine if aircraft is short, medium, or long range
    ac_is_sr = 1 if ac_type in short_range_ac else 0
    ac_is_mr = 1 if ac_type in medium_range_ac else 0
    ac_is_lr = 1 if ac_type in long_range_ac else 0
    
    #Return row, or error if type is invalid
    if ac_type in allowed_types:
        return ac_type, ac_is_narrow, ac_is_wide, ac_is_sr, ac_is_mr, ac_is_lr
    return 'NOT ALLOWED',0,0,0,0,0

df2[['ICAO Type', 'Narrow-body', 'Wide-body', 'Short Range', 'Medium Range', 'Long Range']] = df2.apply(lambda x: convert_model_ICAO_type(x), axis=1, result_type='expand')
print('ICAO Type Converted.')

#"General" aircraft type dictionary using ICAO type
def convert_ICAO_general_type(ICAO_Type):
    if ICAO_Type in ['B712']:
        return 'Boeing 717'
    elif ICAO_Type in ['B732', 'B733', 'B734', 'B735']:
        return 'Boeing 737 Original/Classic'
    elif ICAO_Type in ['B736', 'B737', 'B738', 'B739', ]:
        return 'Boeing 737 NG'
    elif ICAO_Type in ['B37M', 'B38M', 'B39M', 'B3XM']:
        return 'Boeing 737 MAX'
    elif ICAO_Type in ['B752', 'B753']:
        return 'Boeing 757'
    elif ICAO_Type in ['B762', 'B763', 'B764']:
        return 'Boeing 767'
    elif ICAO_Type in ['B772', 'B77L', 'B773', 'B77W', 'B778', 'B779']:
        return 'Boeing 777'
    elif ICAO_Type in ['B788', 'B789', 'B78X']:
        return 'Boeing 787'
    elif ICAO_Type in ['BCS1', 'BCS3']:
        return 'Airbus A220'
    elif ICAO_Type in ['A318']:
        return 'Airbus A319'
    elif ICAO_Type in ['A19N', 'A319']:
        return 'Airbus A319'
    elif ICAO_Type in ['A20N', 'A320']:
        return 'Airbus A320'
    elif ICAO_Type in ['A21N', 'A321']:
        return 'Airbus A321'
    elif ICAO_Type in ['A332', 'A333', 'A338', 'A339']:
        return 'Airbus A330'
    elif ICAO_Type in ['A359', 'A35K']:
        return 'Airbus A350'
    elif ICAO_Type in ['DH8A', 'DH8B', 'DH8C', 'DH8D']:
        return 'Dash 8'
    elif ICAO_Type in ['CRJ2']:
        return 'Canadair Regional Jet 100/200'
    elif ICAO_Type in ['CRJ7', 'CRJ9']:
        return 'Canadair Regional Jet 550/700/900/1000'
    elif ICAO_Type in ['E135', 'E145']:
        return 'Embraer Regional Jet 135/140/145'
    elif ICAO_Type in ['E170', 'E75S', 'E75L', 'E190', 'E195']:
        return 'Embraer E-Jet'
    else:
        return 'ERROR'
df2['General Type'] = df2['ICAO Type'].apply(convert_ICAO_general_type)
#Rearrange columns
df2 = df2[['Tail', 'Year', 'Manufacturer', 'Model', 'ICAO Type', 'General Type', 'Narrow-body', 'Wide-body', 'Short Range', 'Medium Range', 'Long Range']]
print('General Type Converted.')

#Sort the dataframe and reset the index
df2.sort_values(by=['Manufacturer', 'Model', 'Year', 'Tail'], ascending=True, inplace=True)
df2.reset_index(inplace=True, drop=True)

#Save data to csv
df2.to_csv('AircraftData.csv', index=False)
print('CSV Saved.')

#Pickle dataframe for faster loading
with open('./data/AircraftData.pkl', 'wb') as f:
    pickle.dump(df2, f)
print('Pickle Saved.')

print('Processing Complete!')