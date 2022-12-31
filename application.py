#Import packages for data & data manipulation
import pandas as pd
import numpy as np
import pickle
import datetime
import boto3

#Import packages for dashboarding
from dash import Dash, dash_table, html, dcc, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp

s3 = boto3.resource('s3')

#df1 = pickle.loads(s3.Bucket('jackyluo').Object('2022Data.pkl').get()['Body'].read())
#df2 = pickle.loads(s3.Bucket('jackyluo').Object('AircraftData.pkl').get()['Body'].read())
#df3 = pickle.loads(s3.Bucket('jackyluo').Object('Stations.pkl').get()['Body'].read())
#df4 = pickle.loads(s3.Bucket('jackyluo').Object('Carriers.pkl').get()['Body'].read())
with open('./data/2022Data.pkl', 'rb') as f:
    df1 = pickle.load(f)
with open('./data/AircraftData.pkl', 'rb') as f:
    df2 = pickle.load(f)
with open('./data/Stations.pkl', 'rb') as f:
    df3 = pickle.load(f)
with open('./data/Carriers.pkl', 'rb') as f:
    df4 = pickle.load(f)
    
data = pd.merge(df1, df2, how='left', left_on='TAIL_NUM', right_on='Tail')

carrier_code_to_name_converter = df4.to_dict()['Description']
carrier_name_to_code_converter = {value: key for key,value in carrier_code_to_name_converter.items()}
carrier_types = ['MKT_UNIQUE_CARRIER', 'OP_UNIQUE_CARRIER']
carrier_type = carrier_types[0]

op_carriers = list(df1[carrier_type].unique())
op_carriers = [carrier_code_to_name_converter[x] + ' (' + x + ')' for x in op_carriers]
op_carriers = sorted(op_carriers)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, update_title='Loading...', external_stylesheets=external_stylesheets)
#app.title = '2022 Bureau of Transportation Statistics Airline Data Visualization'

application = app.server

#Clear the layout and do not display exception till callback gets executed

app.layout = html.Div([
    html.Div(id='title-changer'),
    html.Div([
        html.H1('2022 Bureau of Transportation Statistics Airline Data Visualization', id='header'),
        html.Div([
            html.Div([
                html.P('Select a Carrier'),
                dcc.Dropdown(op_carriers, placeholder='Select a Carrier', id='main-dropdown'),
            ]),
            html.Br(),
            html.Div([
                html.P('Select a Date Range (by month)'),
                dcc.RangeSlider(1, 9, 1, value=[1,9], id='main-date-selector')
            ]),
            html.Br(),
            html.Div([
                html.P('Select a Type Designator Operating Mode (IATA Type Designators Currently Unavailable)'),
                dcc.RadioItems(['Aircraft use General Type Designators', 'Aircraft use ICAO Type Designators'],
                               'Aircraft use General Type Designators', inline=True, id='main-type-selector')
            ]),
            html.Br(),
            html.Div([
                html.P('Select the data you would like to see'),
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

@app.callback(
    Output('tab0', 'disabled'),
    Output('tab1', 'disabled'),
    Output('tab2', 'disabled'),
    Input('main-dropdown', 'value')
)
def enable_tabs(airline):
    if airline:
        return False, False, False
    return True, True, True

@app.callback(
    Output('output-wrapper', 'children'),
    Input('main-dropdown', 'value'),
    Input('main-date-selector', 'value'),
    Input('main-tab-selector', 'value'),
    Input('main-type-selector', 'value')
)
def update_airline(airline, date_range, tab_selected, selected_type):
    if airline:
        airline = carrier_name_to_code_converter[airline[:-5]]
    df = data[data[carrier_type] == airline]
    date_range[0] = '0' + str(date_range[0]) if date_range[0] < 10 else str(date_range[0])
    date_range[1] = '0' + str(date_range[1] + 1) if date_range[1] < 9 else str(date_range[1] + 1)
    date_range[0] = '2022-' + date_range[0]
    date_range[1] = '2022-' + date_range[1] if date_range[1] != '13' else '2023-01'
    dates = np.arange(date_range[0], date_range[1], dtype='datetime64[D]')
    start_date = dates[0]
    end_date = dates[len(dates) - 1]
    df = df[(df['FL_DATE'] >= start_date) & (df['FL_DATE'] <= end_date)]
    output = html.Div()
    
    if tab_selected == 'Aircraft Data':
        op_types = ['General Type', 'ICAO Type']
        op_type = op_types[0] if selected_type == 'Aircraft use General Type Designators' else op_types[1]
        df_aircraft = df.copy()
        df_aircraft = df_aircraft[df_aircraft['OP_UNIQUE_CARRIER'] == 'airline']
        df_aircraft = df_aircraft[~df_aircraft['Tail'].duplicated(keep='first')]
        df_aircraft['Range'] = None
        df_aircraft['Range'] = df_aircraft.apply(lambda x: 'Short Range' if x['Short Range'] == 1 else x['Range'], axis=1)
        df_aircraft['Range'] = df_aircraft.apply(lambda x: 'Medium Range' if x['Medium Range'] == 1 else x['Range'], axis=1)
        df_aircraft['Range'] = df_aircraft.apply(lambda x: 'Long Range' if x['Long Range'] == 1 else x['Range'], axis=1)
        df_aircraft['Width'] = df_aircraft.apply(lambda x: 'Wide-body' if x['Wide-body'] == 1 else 'Narrow-body', axis=1)
        df_aircraft = df_aircraft[['Range', 'Width', 'Manufacturer', 'ICAO Type', 'General Type', 'Tail', 'Year']]
        df_aircraft['Count'] = 1

        df_fleet_0 = df_aircraft.groupby(op_type)[op_type].count().to_frame().rename({op_type: 'Number of Aircraft'}, axis=1).reset_index()
        df_fleet_1 = df_aircraft.groupby(op_type)['Year'].mean().to_frame().rename({'Year': 'Average Age'}, axis=1).reset_index()
        df_fleet_1['Average Age'] = float(datetime.datetime.now().year) - df_fleet_1['Average Age']
        df_fleet_1['Average Age'] = df_fleet_1['Average Age'].round(2)
        df_fleet = pd.merge(df_fleet_0, df_fleet_1, how='inner', left_on=op_type, right_on=op_type)
        df_fleet = df_fleet.sort_values(by=['Number of Aircraft', op_type], ascending=False)
        total_fleet_size = df_fleet['Number of Aircraft'].sum()
        
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
                    dcc.Graph(figure=px.histogram(df_aircraft,x=op_type).update_xaxes(categoryorder='total ascending')),
                ], className='eight columns'),
            ]),
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(figure=px.pie(df_aircraft, names='Width', values='Count'))
                    ], className='four columns'),
                    html.Div([
                        dcc.Graph(figure=px.pie(df_aircraft, names='Range', values='Count'))
                    ], className='four columns'),
                    html.Div([
                        dcc.Graph(figure=px.pie(df_aircraft, names='Manufacturer', values='Count'))
                    ], className='four columns'),
                ]),
            ]),
        ])
    elif tab_selected == 'Network Data':
        df_network = df.copy()
        df_network = df_network.reset_index()
        operators = df_network['OP_UNIQUE_CARRIER'].unique()
        operators = [carrier_code_to_name_converter[x] for x in operators]
        
        num_flights = df_network.groupby('ORIGIN')['ORIGIN'].count().to_frame().rename({'ORIGIN': 'Number of Flights'}, axis=1).reset_index()
        longest_flight = df_network['DISTANCE'].max()
        longest_flight_idx = df_network['DISTANCE'].idxmax()
        longest_flight_airtime = int(df_network['AIR_TIME'].iloc[longest_flight_idx])
        longest_flight_pair = (df_network['ORIGIN'].iloc[longest_flight_idx], df_network['DEST'].iloc[longest_flight_idx])
        mean_distance = df_network['DISTANCE'].mean()
        mean_airtime = df_network['AIR_TIME'].mean()
        shortest_flight = df_network['DISTANCE'].min()
        shortest_flight_idx = df_network['DISTANCE'].idxmin()
        shortest_flight_airtime = int(df_network['AIR_TIME'].iloc[shortest_flight_idx])
        shortest_flight_pair = (df_network['ORIGIN'].iloc[shortest_flight_idx], df_network['DEST'].iloc[shortest_flight_idx])
        total_destinations = df_network['DEST'].nunique()
        df_network = df_network[~df_network[['ORIGIN', 'DEST']].duplicated(keep='first')]
        df_network = pd.merge(df_network, df3, how='inner', left_on='ORIGIN', right_on='ORIGIN')
        df_network = pd.merge(df_network, num_flights, how='inner', left_on='ORIGIN', right_on='ORIGIN')
        hubs = df_network[~df_network['ORIGIN'].duplicated(keep='first')].sort_values('Number of Flights').tail()['ORIGIN']
        
        output = html.Div([
            html.Div([
                html.Div([
                    html.H4(f'Total Domestic Destinations: {total_destinations}'),
                    html.H5(f'Flights Operated By: {", ".join([str(x) for x in [*operators]])}'),
                    html.Br(),
                    html.H5(f'Top 5 Serviced Airports: {hubs.iloc[4]}, {hubs.iloc[3]}, {hubs.iloc[2]}, {hubs.iloc[1]}, {hubs.iloc[0]}'),
                ], className='five columns'),
                html.Div([
                    html.H5(f'Longest Domestic Route: {longest_flight_pair[0]} - {longest_flight_pair[1]}'),
                    html.H5(f'Longest Domestic Route Distance: {longest_flight} mi'),
                    html.H5(f'Longest Domestic Route Flight Time: {longest_flight_airtime} min'),
                    html.Hr(),
                    html.H5(f'Shortest Domestic Route: {shortest_flight_pair[0]} - {shortest_flight_pair[1]}'),
                    html.H5(f'Shortest Domestic Route Distance: {shortest_flight} mi'),
                    html.H5(f'Shortest Domestic Route Flight Time: {shortest_flight_airtime} min'),
                    html.Hr(),
                    html.H5(f'Mean Distance: {mean_distance:.1f} mi'),
                    html.H5(f'Mean Flight Time: {mean_airtime:.1f} min'),
                ], className='five columns'),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(figure=px.scatter_mapbox(df_network, lat='LATITUDE', lon='LONGITUDE', size='Number of Flights',
                                                        hover_name='NAME', hover_data=['ELEVATION', 'ICAO', 'FAA', 'IATA'], size_max=75,
                                                        zoom=4, mapbox_style='carto-positron'),style={'width': '100%', 'height': '90vh'})
                ]),
            ], className='twelve columns'),
        ])
    elif tab_selected == 'Performance Data':
        df_routes = df.copy()
        df_routes = pd.merge(df_routes, df3, how='inner', left_on='ORIGIN', right_on='ORIGIN')
        days_of_week = df_routes.groupby('DAY_OF_WEEK')['ORIGIN'].count().to_frame().reset_index()
        days_of_week['Day Name'] = days_of_week['DAY_OF_WEEK']
        days_of_week.replace({'Day Name': {1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday',7:'Sunday',9:'Unknown'}}, inplace=True)
        days_of_week.rename({'DAY_OF_WEEK': 'Day of Week', 'ORIGIN': 'Number of Flights'}, axis=1, inplace=True)
        daily_avg_delay_0 = df_network.groupby('FL_DATE')['ARR_DELAY'].mean().to_frame().reset_index()
        daily_avg_delay_1 = df_network.groupby('FL_DATE')['DEP_DELAY'].mean().to_frame().reset_index()
        daily_avg_delay = pd.merge(daily_avg_delay_0, daily_avg_delay_1, how='inner', left_on='FL_DATE', right_on='FL_DATE')
        dep_delay = df_routes['DEP_DELAY'].mean()
        arr_delay = df_routes['ARR_DELAY'].mean()
        ontime_arr = df_routes[df_routes['ARR_DELAY'] <= 0].shape[0] / df_routes.shape[0] * 100
        ontime_dep = df_routes[df_routes['DEP_DELAY'] <= 0].shape[0] / df_routes.shape[0] * 100
        ontime_conv = df_routes[(df_routes['DEP_DELAY'] > 0) & (df_routes['ARR_DELAY'] <= 0)].shape[0] / df_routes[df_routes['DEP_DELAY'] <= 0].shape[0] * 100
        delays = []
        delay0 = df_routes['CARRIER_DELAY'].sum()
        delay1 = df_routes['WEATHER_DELAY'].sum()
        delay2 = df_routes['NAS_DELAY'].sum()
        delay3 = df_routes['SECURITY_DELAY'].sum()
        delay4 = df_routes['LATE_AIRCRAFT_DELAY'].sum()
        total_delay = delay0 + delay1 + delay2 + delay3 + delay4
        delays.append((delay0, delay0 / total_delay * 100, 'Carrier'))
        delays.append((delay1, delay1 / total_delay * 100, 'Weather'))
        delays.append((delay2, delay2 / total_delay * 100, 'National Air System'))
        delays.append((delay3, delay3 / total_delay * 100, 'Security'))
        delays.append((delay4, delay4 / total_delay * 100, 'Late Aircraft'))
        delays.sort(key=lambda x: x[0], reverse=True)
        
        output = html.Div([
            html.Div([
                html.Div([
                    html.H5(f'Number of Flights: {df_routes.shape[0]}'),
                    html.H5(f'Average Arrival Delay: {arr_delay:.0f} min'),
                    html.H5(f'Average Departure Delay: {dep_delay:.0f} min'),
                    html.H5(f'Percentage of Flights Arriving Early/On-Time: {ontime_arr:.2f}%'),
                    html.H5(f'Percentage of Flights Departing Early/On-Time: {ontime_dep:.2f}%'),
                    html.H5(f'Percentage of Flights Departing Late but Arriving Early/On-Time: {ontime_conv:.2f}% of delays'),
                ],className='six columns'),
                html.Div([
                    html.H5(f'Total Delay Caused by {delays[0][2]}: {delays[0][0]} min ({delays[0][1]:.2f}% of delays)'),
                    html.H5(f'Total Delay Caused by {delays[1][2]}: {delays[1][0]} min ({delays[1][1]:.2f}% of delays)'),
                    html.H5(f'Total Delay Caused by {delays[2][2]}: {delays[2][0]} min ({delays[2][1]:.2f}% of delays)'),
                    html.H5(f'Total Delay Caused by {delays[3][2]}: {delays[3][0]} min ({delays[3][1]:.2f}% of delays)'),
                    html.H5(f'Total Delay Caused by {delays[4][2]}: {delays[4][0]} min ({delays[4][1]:.2f}% of delays)'),
                ], className='six columns'),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(figure=px.line(daily_avg_delay, x='FL_DATE', y=['ARR_DELAY', 'DEP_DELAY'], markers=True))
                ]),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(figure=px.histogram(df_routes, x='FL_DATE'))
                ], className='six columns'),
                html.Div([
                    dcc.Graph(figure=px.line(days_of_week, x='Day of Week', y='Number of Flights', hover_name='Day Name')),
                ], className='six columns')
            ], className='twelve columns'),
        ])

    return output

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

#Run with debug mode active on port 8080
if __name__ == '__main__':
    application.run_server(debug=True, port=8080)