# DASH IMPORTS
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction

# DATA ANALYSIS IMPORTS
import pandas as pd, numpy as np, geopandas as gpd, json  # delete json

# GRAPHING IMPORTS
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# import options from options.py
from options import hfs, counties

######################################################################################################################
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                  meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server

# DATA
his = pd.read_csv('Data/all_routes.csv')
hr = pd.read_csv('Data/speed_file.csv')
lottr = pd.read_csv('Lottr.csv', usecols=['LinkDir', 'LOTTR', 'Year'])
streets = gpd.read_file('Data/shp/prj.shp')
streets = streets.dropna(subset=['geometry'])
streets.reset_index(drop=True, inplace=True)

# GLOBAL VARIABLES
mapbox_access_token = "pk.eyJ1IjoiZWFuMjQ2IiwiYSI6ImNrbGkwNG9kMTFmdTUybm82ZTBtZXYzbjMifQ.md30QLwkApQpQ_2U3mEO7Q"
embed_link = "https://uky-edu.maps.arcgis.com/apps/webappviewer/index.html?id=9b04f7cf872b492897f77f04a73fbe75"
s = "dark"
rt = list(his.route.unique())
rtpref = list(his.rtprefix.unique())
bar_color = 'rgb(123, 199, 255)'

# APP LAYOUT
app.layout = html.Div(id="mainContainer", style={"display": "flex", "flex-direction": "column"}, children=[
    html.Div(id="output-clientside"),  # empty Div to trigger javascript file for resizing
    html.Div(className="row flex-display", id="header", style={"margin-bottom": "25px"}, children=[
        html.Div(className="one-third column", children=[
            html.Img(
                src=app.get_asset_url("KTC_png.png"),
                id="ktc-image",
            )
        ]),
        html.Div(className="one-half column", id="title", children=[
            html.Div([
                html.H3('SHRP2 DASHBOARD PROTOTYPE', id='toptext', ),
                html.H5("PROTOTYPE OVERVIEW", id='bottomtext', ),
            ])
        ]),
        html.Div(className="one-third column", id="button", children=[
            html.A(
                html.Button("See Statewide Data", id="learn-more-button"),
                href="https://uky-edu.maps.arcgis.com/apps/opsdashboard/index.html#/762ae87adc5f4d5ca21f02ed5f97c2da",
                target="_blank"
            )
        ]),
    ]),
    html.Div(className="row flex-display", children=[
        html.Div(className="pretty_container four columns", id="cross-filter-options", children=[
            html.P("Select filters below:", className="control_label", style={'font-weight': 'bold'}),
            html.Label("KYTC District:"),
            dcc.Dropdown(id="district",
                         options=[{"label": str(x), "value": x} for x in range(1, 13)],
                         value=None,
                         placeholder="select a district",
                         clearable=True,
                         persistence=False,
                         # persistence_type="memory",
                         className="dcc_control"),

            html.Label("County: "),
            dcc.Dropdown(id="county",
                         options=[{"label": str(x), "value": x} for x in sorted(counties.keys())],
                         value=None,
                         placeholder="select a county",
                         clearable=True,
                         persistence=False,
                         # persistence_type="memory",
                         className="dcc_control"),

            html.Label("Route: "),
            dcc.Dropdown(id="route",
                         options=[{"label": str(x), "value": x} for x in sorted(rt)],
                         value=None,
                         placeholder="select a route",
                         clearable=True,
                         persistence=False,
                         # persistence_type="memory",
                         className="dcc_control"),

            html.P(id="mprange", className="control_label"),

            html.Br(),
            html.Label("MilePoint Range:"),
            dcc.Input(id="bmp",
                      debounce=True,
                      inputMode="numeric",
                      placeholder="begin milepoint",
                      persistence=False,
                      # persistence_type="memory",
                      type="number",
                      className="dcc_control"),

            dcc.Input(id="emp",
                      debounce=True,
                      inputMode="numeric",
                      placeholder="ending milepoint",
                      persistence=False,
                      # persistence_type="memory",
                      type="number",
                      className="dcc_control"),

            html.P(id="selectedRange", className="control_label"),

        ]),

        html.Div(className="eight columns", id="right-column", children=[

            html.Div(className="pretty_container", id="mapboxGraphContainer", children=[
                # dcc.Graph(id="mapbox_plot")
                html.Iframe(id="embed",
                            src=embed_link,
                            width='100%',
                            height='590px')
            ]),
        ]),

    ]),

#    html.Div(id="hiddenDiv", ),

    html.Div(className="row flex-display", children=[
        html.Div([html.H3("CARDINAL", id="ctext", )], className="pretty_container six columns"),
        html.Div([html.H3("NON-CARDINAL", id="nctext", )], className="pretty_container six columns")
    ]),

    html.Div(className="row flex-display", children=[
        html.Div([dcc.Graph(id="HrSpdc")], className="pretty_container six columns"),
        html.Div([dcc.Graph(id="HrSpdnc")], className="pretty_container six columns")
    ]),
    html.Div(className="row flex-display", children=[
        html.Div([dcc.Graph(id="DTc")], className="pretty_container six columns"),
        html.Div([dcc.Graph(id="DTnc")], className="pretty_container six columns")
    ]),
    html.Div(className="row flex-display", children=[
        html.Div([dcc.Graph(id="Delc")], className="pretty_container six columns"),
        html.Div([dcc.Graph(id="Delnc")], className="pretty_container six columns")
    ]),

    html.Div(className="row flex-display", children=[
        html.Div([dcc.Graph(id="Hmapc")], className="pretty_container six columns"),
        html.Div([dcc.Graph(id="Hmapnc")], className="pretty_container six columns")
    ]),

    html.Div(className="row flex-display", children=[
        html.Div([dcc.Graph(id="Lottrc")], className="pretty_container six columns"),
        html.Div([dcc.Graph(id="Lottrnc")], className="pretty_container six columns")
    ]),
])


# HELPER FUNCTIONS
def routes(x):
    f = x.split('-')[1:3]
    return "-".join(f)


def countyfunc(x):
    f = x.split('-')[0]
    return f


def diction(lnk, dx, lg, v):
    dr = dict(zip(list(lnk), list(dx)))
    lt = dict(zip(list(lnk), list(lg)))
    vo = dict(zip(list(lnk), list(v)))
    return dr, lt, vo


def hvol(lnk, v):
    vo = dict(zip(list(lnk), list(v)))
    return vo


streets['route'] = streets['ROUTE_ID'].map(routes)


# define color for mapbox plot speeds
def color(x, df):
    if x < (np.average(df['SPEED_LIMI'], weights=df['SECTION_LE']) / 1.5):
        cl = 'red'
    elif x < (np.average(df['SPEED_LIMI'], weights=df['SECTION_LE']) / 1.2):
        cl = 'gold'
    else:
        cl = 'green'
    return cl


# def zoomlevel()

def generate_street_colormap(streets, style, x, y, z):
    street_colors = ['#006600', '#99ff00', '#ffff33', '#ff9900', '#ff6600']
    layout = go.Layout(
        title_text='Road Heatmap',
        autosize=True,
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat=x, lon=y),  # center coordinate dynamic
            pitch=0,
            zoom=z,
            style=style,
        ),
    )

    data = []

    for i in range(len(streets)):
        new_trace = go.Scattermapbox(
            lat=list(streets.loc[i]['geometry'].coords.xy[1]),
            lon=list(streets.loc[i]['geometry'].coords.xy[0]),
            hoverinfo='text',
            mode="markers+lines",
            hovertext='Peak hour speed: {0}   Milepoints: [{1}, {2}]'.format(str(streets.loc[i, 'PeakHrSpee']),
                                                                             str(streets.loc[i, 'BEGIN_POIN']),
                                                                             str(streets.loc[i, 'END_POINT'])),
            marker=go.scattermapbox.Marker(color=color(streets.loc[i, 'PeakHrSpee'], streets),
                                           size=4,
                                           opacity=0.7),
        )
        data.append(new_trace)
    return {"data": data, "layout": layout}


def getkeys(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:  # get keys based on items
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys


# CALLBACKS

# SCREEN_SIZING CALLBACK
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("mapbox_plot", "figure")],
)


# CALLBACKS FOR USER OPTIONS
@app.callback(
    [Output("county", "options"),
     Output("county", "value"), ],
    Input("district", "value"),
    prevent_initial_call=True
)
def updateCountyOptions(d):
    if d is not None:
        return [{"label": str(x), "value": x} for x in
                sorted(list(his.loc[(his['District'] == d), 'County'].unique()))], None
    else:
        return [{"label": str(x), "value": x} for x in sorted(counties.keys())], None


@app.callback(
    [Output("route", "options"), Output("route", "value"), ],
    [Input("district", "value"), Input("county", "value")],
    prevent_initial_call=True
)
def updateRouteOptions(d, c):
    if d is None:
        if (c is None):
            r = rt
        elif (c is not None):
            r = list(his.loc[(his['County'] == c), 'route'].unique())

    elif (d is not None):
        if (c is None):
            r = list(his.loc[(his['District'] == d), 'route'].unique())
        elif (c is not None):
            r = list(his.loc[(his['County'] == c), 'route'].unique())

    return [{"label": str(x), "value": x} for x in sorted(r)], None

    # only interstates are continuous over county and district lines. produce an error to show that.


@app.callback(
    Output("mprange", "children"),
    [Input("route", "value"), Input("district", "value"), Input("county", "value")],
    prevent_initial_call=True
)
def showMPRange(r, d, c):  # optimize this later
    if r is None:
        sentence = ''
    elif r is not None:
        if c is not None:
            x = his.loc[(his['County'] == c) & (his['route'] == r), 'BEGIN_POINT'].min()
            y = his.loc[(his['County'] == c) & (his['route'] == r), 'END_POINT'].max()
        elif d is not None:
            x = his.loc[(his['District'] == d) & (his['route'] == r), 'BEGIN_POINT'].min()
            y = his.loc[(his['District'] == d) & (his['route'] == r), 'END_POINT'].max()
        else:
            x = his.loc[(his['route'] == r), 'BEGIN_POINT'].min()
            y = his.loc[(his['route'] == r), 'END_POINT'].max()
        sentence = "Route begins from milepoint {} to {}".format(x, y)

    return sentence


@app.callback(
    [Output("bmp", "min"), Output("bmp", "max"), Output("emp", "min"), Output("emp", "max"), Output("bmp", "value"),
     Output("emp", "value")],
    [Input("route", "value"), Input("district", "value"), Input("county", "value")],
    prevent_initial_call=True
)
def define_mpinput_min_max(r, d, c):  # optimize this later. create a hidden div to share data between callbacks
    if c is not None:
        bmn = his.loc[(his['County'] == c) & (his['route'] == r), 'BEGIN_POINT'].min()
        bmx = his.loc[(his['County'] == c) & (his['route'] == r), 'BEGIN_POINT'].max()
        emn = his.loc[(his['County'] == c) & (his['route'] == r), 'END_POINT'].min()
        emx = his.loc[(his['County'] == c) & (his['route'] == r), 'END_POINT'].max()
    elif d is not None:
        bmn = his.loc[(his['District'] == d) & (his['route'] == r), 'BEGIN_POINT'].min()
        bmx = his.loc[(his['District'] == d) & (his['route'] == r), 'BEGIN_POINT'].max()
        emn = his.loc[(his['District'] == d) & (his['route'] == r), 'END_POINT'].min()
        emx = his.loc[(his['District'] == d) & (his['route'] == r), 'END_POINT'].max()
    else:
        bmn = his.loc[(his['route'] == r), 'BEGIN_POINT'].min()
        bmx = his.loc[(his['route'] == r), 'BEGIN_POINT'].max()
        emn = his.loc[(his['route'] == r), 'END_POINT'].min()
        emx = his.loc[(his['route'] == r), 'END_POINT'].max()
    return bmn, bmx, emn, emx, bmn, emx


# CALLBACKS FOR DATA ANALYSIS AND GRAPH GENERATION.
@app.callback(
    [Output("HrSpdc", "figure"),
     Output("HrSpdnc", "figure"),
     Output("DTc", "figure"),
     Output("DTnc", "figure"),
     Output("Delc", "figure"),
     Output("Delnc", "figure"),
     Output("Hmapc", "figure"),
     Output("Hmapnc", "figure"),
     Output("Lottrc", "figure"),
     Output("Lottrnc", "figure"),
     ],
    [Input("route", "value"),
     Input("bmp", "value"),
     Input("emp", "value")
     ],
    prevent_initial_call=True
)
def showHS(r, b, e):
    df = his.loc[(his['ROUTE_ID'].str.contains(r)) & (his['BEGIN_POINT'].astype(float).between(b, e, inclusive=True))]
    c = set(df.loc[df['AllRds_Dir'] == 'Cardinal', 'LinkDir'].unique())
    nc = set(df.loc[df['AllRds_Dir'] != 'Cardinal', 'LinkDir'].unique())
    limit = np.mean(df.loc[df['AllRds_Dir'] == 'Cardinal', 'SPEED_LIMIT_LWA'].unique())
    limitn = np.mean(df.loc[df['AllRds_Dir'] != 'Cardinal', 'SPEED_LIMIT_LWA'].unique())

    # AGGREGATION FUNCS
    wmdc = lambda x: round(np.average(x, weights=dc.loc[x.index, 'Length']), 2)
    lwlt = lambda x: round(np.average(x, weights=lot.loc[x.index, 'Length']), 2)
    nsum = lambda x: round(np.sum(x), 1)

    # CARDINAL
    dc = hr[hr['LinkDir'].isin(c)]  #
    lot = lottr[lottr['LinkDir'].isin(c)]

    drt, leng, adt = diction(df['LinkDir'], df['AllRds_Dir'], df['SECTION_LENGTH'], df['AADT'])
    dc['Length'] = dc['LinkDir'].map(lambda x: leng[x])
    dc['AADT'] = dc['LinkDir'].map(lambda x: adt[x])
    lot['Length'] = lot['LinkDir'].map(lambda x: leng[x])

    dtc = dc.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdc})
    seg_len = np.sum(dc.Length.unique())

    d = dc.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdc, 'AADT': wmdc})
    lotr = lot.groupby('Year', as_index=False).agg({'LOTTR': lwlt})

    if 1 in df['F_SYSTEM'].unique():
        rf = round(np.average(dc['refspeed'], weights=dc['Length']),
                   2)  # if working check if more than speed limit
        rfspd = limit if rf > limit else rf
        d['refspeed'] = rfspd
    else:
        rf = round(np.average(dc.loc[(dc.Hour >= 6) & (dc.Hour < 20), 'refspeed'],
                              weights=dc.loc[(dc.Hour >= 6) & (dc.Hour < 20), 'Length']),
                   2)
        rfspd = limit if rf > limit else rf
        d['refspeed'] = rfspd

    d['TT'] = seg_len / d['AvgSpeed']
    d['TTref'] = seg_len / d['refspeed']

    d['delay'] = d['TT'] - d['TTref']
    d.loc[(d.delay < 0), 'delay'] = 0

    d['perc'] = d['Hour'].map(lambda x: hfs[x])
    d['Vol'] = d['AADT'] * d['perc']
    d['delT'] = d['delay'] * d['Vol'] * 52 * 5
    dl = d.loc[(d.Hour >= 6) & (d.Hour < 20)].reset_index(drop=True)
    dl = d.groupby(["Hour", "Year"], as_index=False).agg({'delT': nsum})

    dtc['refspeed'] = rfspd
    tdl = dl.groupby('Year', as_index=False).agg({'delT': nsum})

    hrv = pd.DataFrame(data={'Hour': range(0, 24), 'AADT': [np.mean(d.AADT.unique())] * 24, })
    hrv['perc'] = hrv['Hour'].map(lambda x: hfs[x])
    hrv['hvol'] = np.ceil((hrv['AADT'] * hrv['perc']))

    #####################################################################
    ############################## HEATMAP ###############################

    filters = ['BEGIN_POINT', 'END_POINT', 'LinkDir', 'AllRds_Dir']

    ###
    dfc = df.loc[df['AllRds_Dir'] == 'Cardinal', filters].sort_values('BEGIN_POINT')
    dfc['Length'] = dfc['END_POINT'] - dfc['BEGIN_POINT']
    islc = np.sum(dfc['Length'])  # sum of individual segment lengths (isl)
    tslc = max(dfc['END_POINT']) - min(dfc['BEGIN_POINT'])  # total segment length (tdl)

    dfca = dfc.append(dfc.iloc[-1])
    dfca.iloc[-1, 0] = max(dfc['END_POINT'])
    dfca = dfca.append(dfc.iloc[0])
    dfca.iloc[-1, 1] = min(dfc['BEGIN_POINT'])
    dfca['BEGIN_POINT'] = dfca['BEGIN_POINT'].round(3)
    dfca['END_POINT'] = dfca['END_POINT'].round(3)
    dfca.reset_index(drop=True, inplace=True)

    null_mpc = []
    if tslc != islc:
        for ind, row in dfca.iterrows():  # iter over rows
            if row['END_POINT'] not in np.array(dfca['BEGIN_POINT']):  # if end mp not in bmp
                null_mpc.append(row['END_POINT'] + 0.001)  # get its value and increment it slightly
                null_mpc.append(round(dfca.loc[ind + 1, "BEGIN_POINT"] - 0.001, 3))  # get next bmp
                dfca = dfca.append(dfc.iloc[ind + 1])
                dfca.iloc[-1, 1] = dfca.iloc[-1, 0]
                dfca.reset_index(drop=True, inplace=True)

    speedsc = dc[['LinkDir', 'Hour', 'AvgSpeed', 'Year']]
    speedsc = pd.merge(dfca, speedsc, on='LinkDir', how='inner')
    speedsc = speedsc.groupby(['Hour', 'BEGIN_POINT', 'END_POINT'], as_index=False).agg(
        {'AvgSpeed': lambda x: np.mean(x)})

    pvc = pd.pivot_table(speedsc, index=['END_POINT'], columns=[
        'Hour'], values='AvgSpeed')

    for i in null_mpc:
        pvc = pvc.append(pd.Series(name=i))
    pvc = pvc.sort_index()

    tc = []
    for x in range(0, len(pvc.index)):
        tc.append(list(pvc.iloc[x]))

    ##########################################################################################
    # NON-CARDINAL

    # noncardinal hourly speed
    dn = hr[hr['LinkDir'].isin(nc)]  #
    lotn = lottr[lottr['LinkDir'].isin(nc)]
    wmdn = lambda x: round(np.average(x, weights=dn.loc[x.index, 'Length']), 2)
    lwltn = lambda x: round(np.average(x, weights=lotn.loc[x.index, 'Length']), 2)

    dn['Length'] = dn['LinkDir'].map(lambda x: leng[x])
    lotn['Length'] = lotn['LinkDir'].map(lambda x: leng[x])
    dn['AADT'] = dn['LinkDir'].map(lambda x: adt[x])

    dtn = dn.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdn})
    seg_lenn = np.sum(dn.Length.unique())

    dnc = dn.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdn, 'AADT': wmdn})

    lotrn = lotn.groupby('Year', as_index=False).agg({'LOTTR': lwltn})

    if 1 in df['F_SYSTEM'].unique():
        rfn = round(np.average(dn['refspeed'], weights=dn['Length']),
                    2)  # if working check if more than speed limit
        rfspdn = limitn if rfn > limitn else rfn
        dnc['refspeed'] = rfspdn
    else:
        rfn = round(np.average(dn.loc[(dn.Hour >= 6) & (dn.Hour < 20), 'refspeed'],
                               weights=dn.loc[(dn.Hour >= 6) & (dn.Hour < 20), 'Length']),
                    2)
        rfspdn = limitn if rfn > limitn else rfn
        dnc['refspeed'] = rfspdn

    dnc['TT'] = seg_lenn / dnc['AvgSpeed']
    dnc['TTref'] = seg_lenn / dnc['refspeed']

    dnc['delay'] = dnc['TT'] - dnc['TTref']
    dnc.loc[(dnc.delay < 0), 'delay'] = 0

    dnc['perc'] = dnc['Hour'].map(lambda x: hfs[x])
    dnc['Vol'] = dnc['AADT'] * dnc['perc']
    dnc['delT'] = dnc['delay'] * dnc['Vol'] * 52 * 5

    dln = dnc.loc[(dnc.Hour >= 6) & (dnc.Hour < 20)].reset_index(drop=True)
    dln = dnc.groupby(["Hour", "Year"], as_index=False).agg({'delT': nsum})

    dtn['refspeed'] = rfspdn

    tdn = dln.groupby('Year', as_index=False).agg({'delT': nsum})

    ############# non-cardinal heatmap ##############################################

    dfnc = df.loc[df['AllRds_Dir'] != 'Cardinal', filters].sort_values('BEGIN_POINT')
    dfnc['Length'] = dfnc['END_POINT'] - dfnc['BEGIN_POINT']
    islnc = np.sum(dfnc['Length'])  # sum of individual segment lengths (isl)
    tslnc = max(dfnc['END_POINT']) - min(dfnc['BEGIN_POINT'])  # total segment length (tdl)

    dfnca = dfnc.append(dfnc.iloc[-1])
    dfnca.iloc[-1, 0] = max(dfnc['END_POINT'])
    dfnca = dfnca.append(dfnc.iloc[0])
    dfnca.iloc[-1, 1] = min(dfnc['BEGIN_POINT'])
    dfnca['BEGIN_POINT'] = dfnca['BEGIN_POINT'].round(3)
    dfnca['END_POINT'] = dfnca['END_POINT'].round(3)
    dfnca.reset_index(drop=True, inplace=True)

    null_mpnc = []
    if tslnc != islnc:
        for ind, row in dfnca.iterrows():  # iter over rows
            if row['END_POINT'] not in np.array(dfnca['BEGIN_POINT']):  # if end mp not in bmp
                null_mpnc.append(row['END_POINT'] + 0.001)  # get its value and increment it slightly
                null_mpnc.append(round(dfnca.loc[ind + 1, "BEGIN_POINT"] - 0.001, 3))  # get next bmp
                dfnca = dfnca.append(dfnc.iloc[ind + 1])
                dfnca.iloc[-1, 1] = dfnca.iloc[-1, 0]
                dfnca.reset_index(drop=True, inplace=True)

    speedsnc = dn[['LinkDir', 'Hour', 'AvgSpeed', 'Year']]
    speedsnc = pd.merge(dfnca, speedsnc, on='LinkDir', how='inner')
    speedsnc = speedsnc.groupby(['Hour', 'BEGIN_POINT', 'END_POINT'], as_index=False).agg(
        {'AvgSpeed': lambda x: np.mean(x)})

    pvnc = pd.pivot_table(speedsnc, index=['END_POINT'], columns=[
        'Hour'], values='AvgSpeed')

    for i in null_mpnc:
        pvnc = pvnc.append(pd.Series(name=i))
    pvnc = pvnc.sort_index()

    tnc = []
    for x in range(0, len(pvnc.index)):
        tnc.append(list(pvnc.iloc[x]))

    # chart y range hr graph
    min_y = dtc['AvgSpeed'].min() - 1 if (dtc['AvgSpeed'].min() < dtn['AvgSpeed'].min()) else dtn['AvgSpeed'].min() - 1
    max_y = dtc['AvgSpeed'].max() + 1 if (dtc['AvgSpeed'].max() > dtn['AvgSpeed'].max()) else dtn['AvgSpeed'].max() + 1

    # chart y range hr delay graph
    min_d = dl['delT'].min() - 1 if (dl['delT'].min() < dln['delT'].min()) else dln['delT'].min() - 1
    max_d = dl['delT'].max() + 1 if (dl['delT'].max() > dln['delT'].max()) else dln['delT'].max() + 1

    # cardinal hourly speed figure
    figc = make_subplots(specs=[[{"secondary_y": True}]])

    figc.add_trace(go.Bar(name='Volume',
                          x=hrv["Hour"],
                          y=hrv['hvol'],
                          marker_color=bar_color,
                          opacity=0.5, ),
                   secondary_y=False, )

    figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2018, "Hour"],
                              y=dtc.loc[dtc.Year == 2018, "AvgSpeed"],
                              mode='lines+markers',
                              name='2018 Speed',
                              marker=dict(color='red', ),
                              line=dict(color='red')),
                   secondary_y=True, )

    figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2019, "Hour"],
                              y=dtc.loc[dtc.Year == 2019, "AvgSpeed"],
                              mode='lines+markers',
                              name='2019 Speed',
                              marker=dict(color='blue', ),
                              line=dict(color='blue')),
                   secondary_y=True, )
    figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2019, "Hour"],
                              y=dtc.loc[dtc.Year == 2019, "refspeed"],
                              line=dict(color='green', width=2, dash='dash'),
                              name='Reference Speed'),
                   secondary_y=True, )

    # figc.data = figc.data[::-1]

    figc.update_layout({
        'plot_bgcolor': "#F9F9F9",
        'paper_bgcolor': "#F9F9F9",
        'title': 'Average hourly speeds',
        'hovermode': 'x'
    })

    figc.update_xaxes(title_text="Hour")

    figc.update_yaxes(title_text="Speed(mph)", range=[min_y, max_y], secondary_y=True)
    figc.update_yaxes(title_text="Volume", secondary_y=False)

    # cardinal hourly delay figure
    figdc = go.Figure()
    figdc.add_trace(go.Scatter(x=dl.loc[dl.Year == 2018, "Hour"],
                               y=dl.loc[dl.Year == 2018, "delT"],
                               mode='lines+markers',
                               marker=dict(color='red', ),
                               line=dict(color='red', ),
                               name='2018 delay'))
    figdc.add_trace(go.Scatter(x=dl.loc[dl.Year == 2019, "Hour"],
                               y=dl.loc[dl.Year == 2019, "delT"],
                               mode='lines+markers',
                               marker=dict(color='blue', ),
                               line=dict(color='blue', ),
                               name='2019 delay'))
    figdc.update_layout({
        'plot_bgcolor': "#F9F9F9",
        'paper_bgcolor': "#F9F9F9",
        'title': 'Average hourly Delay',
        'hovermode': 'x'
    })

    figdc.update_yaxes(title_text="Delay - hr", range=[min_d, max_d])
    figdc.update_xaxes(title_text="Hour")

    # cardinal total year delay figure
    figtdc = go.Figure(data=[
        go.Bar(name='cardinal', x=["2018", "2019"], y=list(tdl['delT']))
    ])

    figtdc.update_layout({
        'plot_bgcolor': "#F9F9F9",
        'paper_bgcolor': "#F9F9F9",
        'title': 'Total Delay by Year',
        'hovermode': 'x'
    })

    figtdc.update_xaxes(title_text="Year")
    figtdc.update_yaxes(title_text="veh-hr")

    # cardinal heatmap figure
    fighmc = go.Figure(data=go.Heatmap(
        x=np.array(pvc.columns),
        y=np.array(pvc.index),
        z=tc,
        zsmooth=False,
        zmin=rfspd // 1.5,
        zmax=rfspd,
        connectgaps=False,
        colorscale='RdYlGn'))

    fighmc.update_layout({'title': "Speed Distribution"})

    # cardinal LOTTR figure
    figltc = go.Figure(data=[
        go.Bar(name='cardinal', x=["2018", "2019"], y=list(lotr['LOTTR']))
    ])

    figltc.update_layout({
        'plot_bgcolor': "#F9F9F9",
        'paper_bgcolor': "#F9F9F9",
        'title': 'LOTTR by Year',
        'hovermode': 'x'
    })

    # non-cardinal hourly speed figure
    fignc = make_subplots(specs=[[{"secondary_y": True}]])

    fignc.add_trace(go.Bar(name='Volume',
                           x=hrv["Hour"],
                           y=hrv['hvol'],
                           marker_color=bar_color,
                           opacity=0.5, ),
                    secondary_y=False, )

    fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2018, "Hour"],
                               y=dtn.loc[dtn.Year == 2018, "AvgSpeed"],
                               mode='lines+markers',
                               name='2018 Speed',
                               marker=dict(color='red', ),
                               line=dict(color='red')),
                    secondary_y=True, ),
    fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2019, "Hour"],
                               y=dtn.loc[dtn.Year == 2019, "AvgSpeed"],
                               mode='lines+markers',
                               name='2019 Speed',
                               marker=dict(color='blue'),
                               line=dict(color='blue')),
                    secondary_y=True, ),
    fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2019, "Hour"],
                               y=dtn.loc[dtn.Year == 2019, "refspeed"],
                               line=dict(color='green', width=2, dash='dash'),
                               name='Reference Speed'),
                    secondary_y=True, ),
    fignc.update_layout({
        'plot_bgcolor': "#F9F9F9",
        'paper_bgcolor': "#F9F9F9",
        'title': 'Average hourly speeds',
        'hovermode': 'x'
    })

    fignc.update_xaxes(title_text="Hour of Day")

    fignc.update_yaxes(title_text="Speed (mph)", range=[min_y, max_y], secondary_y=True)
    fignc.update_yaxes(title_text="Volume", secondary_y=False)

    # non-cardinal hourly delay figure
    figdnc = go.Figure()
    figdnc.add_trace(go.Scatter(x=dln.loc[dln.Year == 2018, "Hour"],
                                y=dln.loc[dln.Year == 2018, "delT"],
                                mode='lines+markers',
                                name='2018 delay'))
    figdnc.add_trace(go.Scatter(x=dln.loc[dln.Year == 2019, "Hour"],
                                y=dln.loc[dln.Year == 2019, "delT"],
                                mode='lines+markers',
                                name='2019 delay'))
    figdnc.update_layout({
        'plot_bgcolor': "#F9F9F9",
        'paper_bgcolor': "#F9F9F9",
        'title': 'Average hourly Delay',
        'hovermode': 'x'
    })

    figdnc.update_yaxes(title_text="Delay - hr", range=[min_d, max_d])
    figdnc.update_xaxes(title_text="Hour")

    # non-cardinal total year delay figure
    figtdnc = go.Figure(data=[
        go.Bar(name='noncardinal', x=['2018', '2019'], y=list(tdn['delT']))
    ])
    figtdnc.update_layout({
        'plot_bgcolor': "#F9F9F9",
        'paper_bgcolor': "#F9F9F9",
        'title': 'Total Delay by Year',
        'hovermode': 'x'
    })
    figtdnc.update_xaxes(title_text="Year")
    figtdnc.update_yaxes(title_text="veh-hr")

    # non-cardinal heatmap figure
    fighmnc = go.Figure(data=go.Heatmap(
        x=np.array(pvnc.columns),
        y=np.array(pvnc.index),
        z=tnc,
        zsmooth=False,
        zmin=rfspdn // 1.5,
        zmax=rfspdn,
        connectgaps=False,
        colorscale='RdYlGn'))

    fighmnc.update_layout({'title': "Speed Distribution"})

    # non-cardinal LOTTR figure
    figltnc = go.Figure(data=[
        go.Bar(name='non-cardinal', x=["2018", "2019"], y=list(lotrn['LOTTR']))
    ])

    figltnc.update_layout({
        'plot_bgcolor': "#F9F9F9",
        'paper_bgcolor': "#F9F9F9",
        'title': 'LOTTR by Year',
        'hovermode': 'x'
    })
    figltnc.update_xaxes(title_text="Year")

    return figc, fignc, figdc, figdnc, figtdc, figtdnc, fighmc, fighmnc, figltc, figltnc


@app.callback(
    Output("mapboxGraphContainer", "children"),
    [Input("route", "value"),
     Input("bmp", "value"),
     Input("emp", "value"),
     Input("district", "value"),
     Input("county", "value"), ],
    prevent_initial_call=True
)
def showMapboxplot(r, b, e, d, c):
    if r is not None:
        dst = streets[(streets.route == r) & (streets['BEGIN_POIN'].astype(float).between(b, e, inclusive=True))]
        dst.reset_index(drop=True, inplace=True)
        length = dst.SECTION_LE.sum()
        lat = list(dst.loc[0]['geometry'].coords.xy[1])
        lon = list(dst.loc[0]['geometry'].coords.xy[0])
        x, y = np.mean([max(lat), min(lat)]), np.mean([max(lon), min(lon)])  # add zoom here

        if length <= 1:
            z = 16
        elif length <= 4:
            z = 14
        elif length <= 8:
            z = 12
        elif length <= 12:
            z = 11
        else:
            z = 10

        d = generate_street_colormap(dst, s, x, y, z)
        fig = go.Figure(
            data=d['data'],
            layout=d['layout'])
        fig.update_layout({'height': 650})
        ret = dcc.Graph(id="mapbox_plot", figure=fig)  # ret=return

    else:
        if c is not None:
            ret = html.Iframe(id="embed", src=f"{embed_link}&find={c}", width='100%', height='590px')
        elif d is not None:
            ret = html.Iframe(id="embed", src=f"{embed_link}&find={d}", width='100%', height='590px')
        else:
            ret = html.Iframe(id="embed", src=embed_link, width='100%', height='590px')
    return ret


if __name__ == '__main__':
    app.run_server(debug=True)
