#Import packages for data & data manipulation
import pandas as pd
import numpy as np
import datetime

#Import packages for building the dashbaord
from dash import Dash, dash_table, html, dcc, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp

#Read data from S3
df1 = pd.read_parquet('https://jackyluo.s3.amazonaws.com/AirlineDataSmall.parquet.gzip')
df2 = pd.read_csv('https://jackyluo.s3.amazonaws.com/AircraftData.csv',
                  usecols=['Tail', 'Year', 'Manufacturer', 'ICAO Type', 'General Type',
                            'Narrow-body', 'Wide-body', 'Short Range', 'Medium Range', 'Long Range'],
                  dtype={'Year': np.int16,'Narrow-body': np.int8, 'Wide-body': np.int8,
                         'Short Range': np.int8, 'Medium Range': np.int8, 'Long Range': np.int8,
                         'Manufacturer': 'category','ICAO Type': 'category', 'General Type': 'category'})
df3 = pd.read_csv('https://jackyluo.s3.amazonaws.com/Stations.csv',
                  usecols=['ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM', 'NAME',
                           'LATITUDE', 'LONGITUDE', 'ELEVATION', 'ICAO', 'IATA', 'FAA'],
                  dtype={'LATITUDE': np.float32, 'LONGITUDE': np.float32, 'ELEVATION': np.float32})
df4 = pd.read_csv('https://jackyluo.s3.amazonaws.com/Carriers.csv', index_col='Code')
    
#Merge airline data with aircraft data to add aircraft information
data = pd.merge(df1, df2, how='left', left_on='TAIL_NUM', right_on='Tail')

#Create methods for conversion between unique carrier codes and carrier names for readability
carrier_code_to_name_converter = df4.to_dict()['Description']
carrier_name_to_code_converter = {value: key for key,value in carrier_code_to_name_converter.items()}
carrier_types = ['MKT_UNIQUE_CARRIER', 'OP_UNIQUE_CARRIER']
carrier_type = carrier_types[0]

#Create alphabetical list of carriers for use in dropdown
op_carriers = list(df1[carrier_type].unique())
op_carriers = [carrier_code_to_name_converter[x] + ' (' + x + ')' for x in op_carriers]
op_carriers = sorted(op_carriers)

#Function to convert minutes to day(s)/hour(s)/minute(s) format
def convert_time(minutes):
    return  f'{minutes//24//60:.0f}d {minutes//60%24:.0f}h {minutes%60:.0f}m'

#Stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Set app
app = Dash(__name__, update_title='Loading...', external_stylesheets=external_stylesheets)
application = app.server

#TODO: Add dcc.loading element to complement title change
#Set initial layout, including input elements and placeholder divs
app.layout = html.Div([
    html.Div(id='title-changer'),
    html.Div([
        html.H1('2022 Bureau of Transportation Statistics Airline Data Visualization', id='header'),
        html.Div([
            html.Div([
                dcc.Dropdown(op_carriers, placeholder='Select a Carrier', id='main-dropdown'),
            ]),
            html.Br(),
            html.Div([
                html.P('Select a Date Range (by month)'),
                dcc.RangeSlider(1, 11, 1, value=[1,11], id='main-date-selector')
            ]),
            html.Br(),
            html.Div([
                html.P('Select a Type Designator Operating Mode (IATA Type Designators Currently Unavailable)'),
                dcc.RadioItems(['Aircraft use General Type Designators', 'Aircraft use ICAO Type Designators'],
                               'Aircraft use General Type Designators', inline=True, id='main-type-selector')
            ]),
            html.Br(),
            html.Div([
                dcc.Tabs(id='main-tab-selector', children=[
                    dcc.Tab(label='Aircraft Data', value='Aircraft Data', id='tab0', disabled=True),
                    dcc.Tab(label='Network Data', value='Network Data', id='tab1', disabled=True),
                    dcc.Tab(label='Performance Data', value='Performance Data', id='tab2', disabled=True),
                ]),
            ]),
        ]),
        html.Div(id='output-wrapper'),
    ], style={'display':'block'})
])

#TODO: Add/edit callback for loading

#Callback 1 for disabling tabs until carrier has been selected
@app.callback(
    Output('tab0', 'disabled'),
    Output('tab1', 'disabled'),
    Output('tab2', 'disabled'),
    Input('main-dropdown', 'value')
)
def enable_tabs(airline):
    """Disable tabs until a carrier has been selected in the dropdown.

    Args:
        airline (str): Carrier selected by the dropdown.

    Returns:
        tuple (boolean, boolean, boolean): Enable/disable boolean for aircraft/network/performance data tab respectively
    """
    if airline:
        return False, False, False
    return True, True, True

#Callback 2 for generating output based on tab selected
@app.callback(
    Output('output-wrapper', 'children'),
    Input('main-dropdown', 'value'),
    Input('main-date-selector', 'value'),
    Input('main-tab-selector', 'value'),
    Input('main-type-selector', 'value')
)
def update_airline(airline, date_range, tab_selected, selected_type):
    """Generates the output dash html elements including graphs, tables and text based on selected carrier, date range, tab and operating mode.

    Args:
        airline (str): Selected carrier from dropdown input.
        date_range (list of int): Selected date range in months, min and max val from rangeslider input. Default [1,12].
        tab_selected (str): Selected type of data to be shown from tabs input. 
        selected_type (str): Selected operating mode for aircraft type from radioitems input. Default "General Type".

    Returns:
        dash.Html.Div: Wrapper for output. Contains graphs, tables and text.
    """
    #If a carrier is selected, get its unique carrier code
    if airline:
        airline = carrier_name_to_code_converter[airline[:-5]]
    #Filter data to be only from the selected carrier
    df = data[data[carrier_type] == airline]
    #Create date range based on range selected by the slider
    date_range[0] = '0' + str(date_range[0]) if date_range[0] < 10 else str(date_range[0])
    date_range[1] = '0' + str(date_range[1] + 1) if date_range[1] < 9 else str(date_range[1] + 1)
    date_range[0] = '2022-' + date_range[0]
    date_range[1] = '2022-' + date_range[1] if date_range[1] != '13' else '2023-01'
    dates = np.arange(date_range[0], date_range[1], dtype='datetime64[D]')
    #Get min date and max date in numpy datetime64 type
    start_date = dates[0]
    end_date = dates[len(dates) - 1]
    #Filter data to be only within date range
    df = df[(df['FL_DATE'] >= start_date) & (df['FL_DATE'] <= end_date)]
    #Initialize output as empty div
    output = html.Div()
    
    #Output relevant aircraft data if aircraft data tab is selected
    if tab_selected == 'Aircraft Data':
        #Set operating type based on selected operating type
        op_types = ['General Type', 'ICAO Type']
        op_type = op_types[0] if selected_type == 'Aircraft use General Type Designators' else op_types[1]
        #Filter by operating carrier rather than marketing carrier to prevent overlapping fleet information from regional carriers.
        #This also allows the fleet data to be compared to other fleet tracking websites for consistency.
        df = df[df['OP_UNIQUE_CARRIER'] == airline]
        #Remove cancelled flights
        df = df[df['CANCELLED'] == 0]
        #Create a dataframe of unique N-numbers, which uniquely identify aircraft
        df = df[~df['Tail'].duplicated(keep='first')]
        df['Range'] = None
        #Create a single column for range of aircraft
        df['Range'] = df.apply(lambda x: 'Short Range' if x['Short Range'] == 1 else x['Range'], axis=1)
        df['Range'] = df.apply(lambda x: 'Medium Range' if x['Medium Range'] == 1 else x['Range'], axis=1)
        df['Range'] = df.apply(lambda x: 'Long Range' if x['Long Range'] == 1 else x['Range'], axis=1)
        #Create a single column for wide or narrowbody aircraft
        df['Width'] = df.apply(lambda x: 'Wide-body' if x['Wide-body'] == 1 else 'Narrow-body', axis=1)
        df = df[['Range', 'Width', 'Manufacturer', 'ICAO Type', 'General Type', 'Tail', 'Year']]
        #Create a counter column for building fleet size
        df['Count'] = 1

        #Group aircraft by type to determine number of aircraft of each type as a dataframe, rename the appropriate column and reset index for merge
        df_fleet_0 = df.groupby(op_type)[op_type].count().to_frame().rename({op_type: 'Number of Aircraft'}, axis=1).reset_index()
        #Group aircraft by type to determine average age of aircraft of each type as a dataframe, rename the appropriate column and reset index for merge
        df_fleet_1 = df.groupby(op_type)['Year'].mean().to_frame().rename({'Year': 'Average Age'}, axis=1).reset_index()
        #Calculate the age using current datetime, fix float precision
        df_fleet_1['Average Age'] = float(datetime.datetime.now().year) - df_fleet_1['Average Age']
        df_fleet_1['Average Age'] = df_fleet_1['Average Age'].round(2)
        #Perform the merge to have a single dataframe with aircraft type, number of aircraft of given type, and average age of given type
        df_fleet = pd.merge(df_fleet_0, df_fleet_1, how='inner', left_on=op_type, right_on=op_type)
        #Initially sort by descending number of aircraft to have most common aircraft at the top and least common aircraft at the bottom
        df_fleet = df_fleet.sort_values(by=['Number of Aircraft', op_type], ascending=False).dropna()
        #Get total fleet size by summing number of aircraft of each type
        total_fleet_size = df_fleet['Number of Aircraft'].sum()
        
        #Remerge data
        df = pd.merge(df, df_fleet, how='left', left_on=op_type, right_on=op_type)
        
        #Create output element, including sortable datatable and plots. Use class names to fix element widths based on stylesheet
        output = html.Div([
            html.Div([
                html.Div([
                    html.H5(f'Total Fleet Size: {total_fleet_size}'),
                    dash_table.DataTable(
                        id='fleet-table',
                        data=df_fleet.to_dict('records'),
                        columns=[{'name': i, 'id': i, 'deletable': False} for i in df_fleet.columns if i != 'id'],
                        sort_action='native',
                        sort_mode='single',
                        style_header={'backgroundColor':'rgb(105,105,105)',
                                        'color':'white',
                                        'font_size':'18px'},
                        page_size=10
                    )
                ], className='four columns'),
                html.Div([
                    dcc.Graph(figure=px.bar(df,x=op_type, y='Number of Aircraft', barmode='overlay', hover_name='General Type', hover_data=['Range', 'Width']).update_xaxes(categoryorder='total ascending')),
                ], className='eight columns'),
            ]),
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(figure=px.pie(df, names='Width', values='Count'))
                    ], className='four columns'),
                    html.Div([
                        dcc.Graph(figure=px.pie(df, names='Range', values='Count'))
                    ], className='four columns'),
                    html.Div([
                        dcc.Graph(figure=px.pie(df, names='Manufacturer', values='Count'))
                    ], className='four columns'),
                ]),
            ]),
        ])
    #Output relevant network data if network data tab is selected
    elif tab_selected == 'Network Data':
        #Remove cancelled flights
        df = df[df['CANCELLED'] == 0]
        #Reindex data to make data easier to work with
        df = df.reset_index()
        #Get list of unique operators, convert to names instead of unique codes
        operators = df['OP_UNIQUE_CARRIER'].unique()
        operators = [carrier_code_to_name_converter[x] for x in operators]
        
        #Get number of flights by each origin airport location
        num_flights = df.groupby('ORIGIN')['ORIGIN'].count().to_frame().rename({'ORIGIN': 'Number of Flights'}, axis=1).reset_index()
        #Find the distance of longest flight, get index of longest flight to get other data such as airtime, origin and destination
        longest_flight = df['DISTANCE'].max()
        longest_flight_idx = df['DISTANCE'].idxmax()
        longest_flight_airtime = int(df['AIR_TIME'].iloc[longest_flight_idx])
        longest_flight_pair = (df['ORIGIN'].iloc[longest_flight_idx], df['DEST'].iloc[longest_flight_idx])
        #Find the mean distance and airtime as comparison for shortest/longest flights
        mean_distance = df['DISTANCE'].mean()
        mean_airtime = df['AIR_TIME'].mean()
        #Find the distance of shortest flight, get index of shortest flight to get other data such as airtime, origin and destination
        shortest_flight = df['DISTANCE'].min()
        shortest_flight_idx = df['DISTANCE'].idxmin()
        shortest_flight_airtime = int(df['AIR_TIME'].iloc[shortest_flight_idx])
        shortest_flight_pair = (df['ORIGIN'].iloc[shortest_flight_idx], df['DEST'].iloc[shortest_flight_idx])
        #Get number of total unique destinations
        total_destinations = df['DEST'].nunique()
        #Filter data by unique origin/destination pairs for mapping
        df = df[~df[['ORIGIN', 'DEST']].duplicated(keep='first')]
        #Merge origin/destination pair data with airport names along with longitude/latitude/elevation data
        df = pd.merge(df, df3, how='inner', left_on='ORIGIN', right_on='ORIGIN')
        #Merge origin/destination pair data with number of flights from each destination calculated earlier
        df = pd.merge(df, num_flights, how='inner', left_on='ORIGIN', right_on='ORIGIN')
        #Get top 10 airports by number of outgoing flights, should represent hub airports
        hubs = df[~df['ORIGIN'].duplicated(keep='first')].sort_values('Number of Flights').tail(10)['ORIGIN']
        
        #TODO: Reformat text elements to be table-like with descriptive text as small header, numerical value as large table element for cleaner look
        #TODO: Figure out px.line_mapbox to create route map that includes lines between each origin/destination pair. Currently not clean, incompatible with px.scatter_mapbox
        #Create output element, including text and map. Use class names to fix element widths based on stylesheet
        output = html.Div([
            html.Div([
                html.Div([
                    html.H3(f'Total Domestic Destinations: {total_destinations}'),
                    html.H5(f'Flights Operated By: {", ".join([str(x).replace(".","") for x in [*operators]])}'),
                    html.Hr(),
                    html.H5(f'Top 10 Serviced Airports: {hubs.iloc[9]}, {hubs.iloc[8]}, {hubs.iloc[7]}, {hubs.iloc[6]}, {hubs.iloc[5]}, {hubs.iloc[4]}, {hubs.iloc[3]}, {hubs.iloc[2]}, {hubs.iloc[1]}, {hubs.iloc[0]}'),
                ], className='five columns'),
                html.Div([
                    html.H6(f'Longest Domestic Route: {longest_flight_pair[0]} - {longest_flight_pair[1]}'),
                    html.H6(f'Longest Domestic Route Distance: {longest_flight} mi'),
                    html.H6(f'Longest Domestic Route Flight Time: {longest_flight_airtime} min'),
                    html.H6(f'Shortest Domestic Route: {shortest_flight_pair[0]} - {shortest_flight_pair[1]}'),
                    html.H6(f'Shortest Domestic Route Distance: {shortest_flight} mi'),
                    html.H6(f'Shortest Domestic Route Flight Time: {shortest_flight_airtime} min'),
                    html.H6(f'Mean Distance: {mean_distance:.1f} mi'),
                    html.H6(f'Mean Flight Time: {mean_airtime:.1f} min'),
                ], className='five columns'),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(figure=px.scatter_mapbox(df, lat='LATITUDE', lon='LONGITUDE', size='Number of Flights',
                                                        hover_name='NAME', hover_data=['ELEVATION', 'ICAO', 'FAA', 'IATA'], size_max=75,
                                                        zoom=4, mapbox_style='carto-positron'),style={'width': '100%', 'height': '90vh'})
                ]),
            ], className='twelve columns'),
        ])
    #Output relevant performance data if performance data tab is selected
    elif tab_selected == 'Performance Data':
        #Count number of cancelled flights, remove cancelled flights from dataframe
        cancellations = df[df['CANCELLED'] > 0]['CANCELLED'].count()
        df = df[df['CANCELLED'] == 0]
        
        #Create a dataframe with the average arrival delay and departure delay by day of year
        daily_avg_delay_0 = df.groupby('FL_DATE')['ARR_DELAY'].mean().to_frame().reset_index()
        daily_avg_delay_1 = df.groupby('FL_DATE')['DEP_DELAY'].mean().to_frame().reset_index()
        daily_avg_delay = pd.merge(daily_avg_delay_0, daily_avg_delay_1, how='inner', left_on='FL_DATE', right_on='FL_DATE')
        #Get average arrival delay and departure delay for all routes by the selected carrier
        dep_delay = df['DEP_DELAY'].mean()
        arr_delay = df['ARR_DELAY'].mean()
        #Get percentage of flights where flights are on time or early (delay <= 15)
        ontime_arr = df[df['ARR_DELAY'] <= 15].shape[0] / df.shape[0] * 100
        ontime_dep = df[df['DEP_DELAY'] <= 15].shape[0] / df.shape[0] * 100
        #Create a list of delay causes
        delays = []
        #Sum number of minutes of delay by each cause
        delay0 = df['CARRIER_DELAY'].sum()
        delay1 = df['WEATHER_DELAY'].sum()
        delay2 = df['NAS_DELAY'].sum()
        delay3 = df['SECURITY_DELAY'].sum()
        delay4 = df['LATE_AIRCRAFT_DELAY'].sum()
        #Sum total delay by carrier
        total_delay = delay0 + delay1 + delay2 + delay3 + delay4
        #Append each cause to list as tuple along with number of minutes of delay, and percentage of total delay
        delays.append((delay0, delay0 / total_delay * 100, 'Carrier'))
        delays.append((delay1, delay1 / total_delay * 100, 'Weather'))
        delays.append((delay2, delay2 / total_delay * 100, 'National Air System'))
        delays.append((delay3, delay3 / total_delay * 100, 'Security'))
        delays.append((delay4, delay4 / total_delay * 100, 'Late Aircraft'))
        #Sort list by number of minutes of delay, descending
        delays.sort(key=lambda x: x[0], reverse=True)
        
        #TODO: Reformat text elements to be table-like with descriptive text as small header, numerical value as large table element for cleaner look
        #Create output element, including text and plots. Use class names to fix element widths based on stylesheet
        output = html.Div([
            html.Div([
                html.Div([
                    html.H5(f'Number of Flights: {df.shape[0]}'),
                    html.H5(f'Number of Cancelled Flights: {cancellations} ({cancellations/df.shape[0]*100:.2f}%)'),
                    html.H5(f'Average Arrival Delay: {arr_delay:.0f} min'),
                    html.H5(f'Average Departure Delay: {dep_delay:.0f} min'),
                    html.H5(f'Percentage of Flights Arriving Early/On-Time: {ontime_arr:.2f}%'),
                    html.H5(f'Percentage of Flights Departing Early/On-Time: {ontime_dep:.2f}%'),
                ],className='six columns'),
                html.Div([
                    html.H5(f'Total Delay: {convert_time(total_delay)}'),
                    html.H5(f'Total Delay Caused by {delays[0][2]}: {convert_time(delays[0][0])} ({delays[0][1]:.2f}% of delays)'),
                    html.H5(f'Total Delay Caused by {delays[1][2]}: {convert_time(delays[1][0])} ({delays[1][1]:.2f}% of delays)'),
                    html.H5(f'Total Delay Caused by {delays[2][2]}: {convert_time(delays[2][0])} ({delays[2][1]:.2f}% of delays)'),
                    html.H5(f'Total Delay Caused by {delays[3][2]}: {convert_time(delays[3][0])} ({delays[3][1]:.2f}% of delays)'),
                    html.H5(f'Total Delay Caused by {delays[4][2]}: {convert_time(delays[4][0])} ({delays[4][1]:.2f}% of delays)'),
                ], className='six columns'),
            ], className='twelve columns'),
            html.Div([
                html.Div([
                    dcc.Graph(figure=px.line(daily_avg_delay, x='FL_DATE', y=['ARR_DELAY', 'DEP_DELAY'], markers=True).add_hline(y=15)),
                    html.P('A flight is considered on-time when it arrives less than 15 minutes after its published arrival time.')
                ]),
            ], className='twelve columns')
        ])
    #Return the output div
    return output

#(Clientside) Callback 3 for changing the title of the webpage based on if a tab is selected.
# If a tab is not selected, use a descriptive title. If a tab is selected, use the name of the tab as the title. Uses "Loading..." as title while processing
app.clientside_callback(
    '''
    function(tab_value) {
        if (tab_value == "tab-1"){
            document.title = "2022 Bureau of Transportation Statistics Airline Data Visualization";
        }else{
            document.title = tab_value;
        }
        return null;
    }
    ''',
    Output('title-changer', 'children'),
    Input('main-tab-selector', 'value')
)

#Run on port 8080
if __name__ == '__main__':
    application.run_server(debug=False, port=8080)