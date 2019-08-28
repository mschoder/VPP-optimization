# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table as dt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_daq as daq
import json
import copy
import pandas as pd
import models
import sim_data
import settings
import urllib

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
# app.config['suppress_callback_exceptions']=True

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="NA",
)

model_opts = {
    'stoc_vpp': 'Stochastic VPP Model',
    'stat_vpp': 'Static VPP Model',
    'stoc_nb': 'Stochastic Model (No Batt)',
    'stat_nb': 'Static Model (No Batt)'
}

app.layout = html.Div(children=[
    # Hidden div inside the app that stores the df from the model query
    html.Div(id='model-result-hidden', style={'display': 'none'}),

    html.Div( # Header block
        [
            html.Div( # Logo Block Left
                [   html.Img(
                        src=app.get_asset_url("llama-logo.png"),
                        id="logo-image-l",
                        style={
                            "height": "200px",
                            "width": "auto",
                            "margin-bottom": "25px",
                        },
                    ),
                ],
                className="three columns",
                id="logo-box-left"
            ),
            html.Div( # Title box, center
                [
                    html.H1(
                        "Llama Power VPP Energy Management System",
                        style={"margin-bottom": "0px", "text-align": "center"},
                    ),
                    html.H4(
                        "Llama Power Co", style={"margin-top": "0px", "text-align": "center"}
                    ),
                ],
                className="seven columns",
                id="title-box",
            ),
            html.Div( # Logo Block Right
                [   html.Img(
                        src=app.get_asset_url("lgo.png"),
                        id="logo-image-r",
                        style={
                            "height": "auto",
                            "width": "300px",
                            "margin-bottom": "25px",
                        },
                    ),
                ],
                className="three columns",
                id="logo-box-right",
                style = {"text-align": "right"}
            ),
        ],
        id="header",
        className="row flex-display",
        style={"margin-bottom": "25px"

        },
    ),
    html.Div( #Top Main Box (controls + graph)
        [
            html.Div( #Main Control Box
            [
                html.H5(
                    "Select Model Parameters", style={"margin-top": "0px", "text-align": "center"},

                # className="pretty_container four columns",
                # id="cross-filter-options",
                ),

                html.Div( #Two col control box
                    [
                        html.Div( #LHS control box
                            [
                                html.Button('Update Model', id='update_btn',
                                style={'margin-bottom': '10px', 'font-weight': 'bold', 'font-size': '12px'}),
                                dcc.Dropdown(
                                    id='model_dd_state',
                                    options=[{'label': v, 'value': k} for k,v in model_opts.items()],
                                    value='stoc_vpp',
                                    style={'height': '40px', 'width': '240px'}
                                ),
                                html.Label('# of Sims', style = {'margin-top': '10px'}),
                                dcc.Input(id='nsims_state', value=5, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Batt Cap (MWh)'),
                                dcc.Input(id='batt_cap_state', value=240, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Batt Charge Rate (MW)'),
                                dcc.Input(id='batt_rate_state', value=240/6, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Min Output (MW)'),
                                dcc.Input(id='req_output_state', value=12, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Underage Fine ($)'),
                                dcc.Input(id='fine_state', value=1500, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Solar Cap (MW)'),
                                dcc.Input(id='solar_cap_state', value=24, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Wind Cap (MW)'),
                                dcc.Input(id='wind_cap_state', value=61.5, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Gas Turb Cap (MW)'),
                                dcc.Input(id='turb_cap_state', value=52, type='number', style={'height': '25px', 'width': '80px'}),

                            ],
                            className="six columns",
                            id="cross-filter-options-l",
                        ),
                        html.Div( # RHS control box
                            [

                                html.P("Season:", className="control_label"),
                                dcc.RadioItems(
                                    id="season_selector_state",
                                    options=[
                                        {"label": "Winter", "value": 0},
                                        {"label": "Summer", "value": 1},
                                    ],
                                    value=0,
                                    labelStyle={"display": "inline-block"},
                                    className="dcc_control",
                                ),
                                html.Label('Gas Turb Start Cost'),
                                dcc.Input(id='gt_startup_cost_state', value=435, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Gas Turb Run Cost/hr'),
                                dcc.Input(id='gt_run_cost_state', value=2600, type='number', style={'height': '25px', 'width': '80px'}),
                                html.P("Forecast Error SD", style = {'margin-top': '15px'}),
                                html.Label('Solar (MW)'),
                                dcc.Input(id='sigma_p_state', value=2, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Wind (MW)'),
                                dcc.Input(id='sigma_w_state', value=4, type='number', style={'height': '25px', 'width': '80px'}),
                                html.Label('Elect Price ($)'),
                                dcc.Input(id='sigma_l_state', value=3, type='number', style={'height': '25px', 'width': '80px'}),
                            ],
                            className="six columns",
                            id="cross-filter-options-r",
                            style = {'margin-top': '90px'}
                        ),
                    ],
                    className="pretty_container ow flex-display",
                    id='two-col-control-box'
                ),
            ],
            id="control-box",
            className="four columns",
            style={"width": "28%"}
            ),

            html.Div(
                [
                    html.H5(id="model_res_header", style={"margin-top": "0px", "text-align": "center"}
                    ),
                    html.Div(
                        [
                            html.Div(
                                [html.H3(id="blk_1", style={"text-align": "center"}),
                                html.P("Weekly Profit", style={"text-align": "center"})],
                                id="mc-1",
                                className="mini_container",
                                style={"width": "24%"}
                            ),
                            html.Div(
                                [html.H3(id="blk_2", style={"text-align": "center"}),
                                html.P("Turbine Starts", style={"text-align": "center"})],
                                id="mc-2",
                                className="mini_container",
                                style={"width": "24%"}
                            ),
                            html.Div(
                                [html.H3(id="blk_3", style={"text-align": "center"}),
                                html.P("Net Generation", style={"text-align": "center"})],
                                id="mc-3",
                                className="mini_container",
                                style={"width": "24%"}
                            ),
                            html.Div(
                                [html.H3(id="blk_4", style={"text-align": "center"}),
                                html.P("Fines Incurred", style={"text-align": "center"})],
                                id="mc-4",
                                className="mini_container",
                                style={"width": "24%"}
                            ),
                        ],
                        id="info-container",
                        className="row container-display",
                    ),
                    html.Div(
                        [dcc.Tabs(id="tabs", value='tab-1', children=[
                            dcc.Tab(label='Power Generation', value='tab-1'),
                            dcc.Tab(label='Scheduling', value='tab-2'),
                        ]),
                        dcc.Graph(id='graph_gen'),],
                        id="tabGraphContainer",
                        className="pretty_container",
                    ),
                ],
                id="right-column",
                className="nine columns",
            ),
        ],
        className="row flex-display",
    ),

    html.Div(
        [
            html.Div(
                [dcc.Graph(id="cum_profit_graph")],
                className="pretty_container five columns",
            ),
            html.Div(
                [dcc.Tabs(id="tabs_small", value='tab-1', children=[
                    dcc.Tab(label='Resource Mix', value='tab-1'),
                    dcc.Tab(label='Profit Distribution', value='tab-2'),
                    dcc.Tab(label='Solar Inputs', value='tab-3'),
                    dcc.Tab(label='Wind Inputs', value='tab-4'),
                    dcc.Tab(label='Price Inputs', value='tab-5'),
                    ],
                ),
                dcc.Graph(id="profit_dist_graph"),],
                id="tabSmGraphContainer",
                className="pretty_container seven columns",
            ),
        ],
        className="row flex-display",
    ),
    html.A(
        'Download Output Data',
        id='download-link',
        download="model_output.csv",
        href="",
        target="_blank"
    ),
    html.Div(id='model_table'),
],
id="mainContainer",
style={"display": "flex", "flex-direction": "column"},
)

@app.callback(
    [Output('model-result-hidden', 'children')],
    [Input('update_btn', 'n_clicks')],
    [State('model_dd_state', 'value'),
    State('season_selector_state', 'value'),
    State('nsims_state', 'value'),
    State('batt_cap_state', 'value'),
    State('batt_rate_state', 'value'),
    State('gt_startup_cost_state', 'value'),
    State('gt_run_cost_state', 'value'),
    State('req_output_state', 'value'),
    State('fine_state', 'value'),
    State('solar_cap_state', 'value'),
    State('wind_cap_state', 'value'),
    State('turb_cap_state', 'value'),
    State('sigma_p_state', 'value'),
    State('sigma_w_state', 'value'),
    State('sigma_l_state', 'value'),
    ]
)
def get_model_data(n_ck, mod_dd, ss, nsims, bat_cap, bat_rate, sc, rc,\
                ro, fine, sol_cap, wind_cap, turb_cap, sig_p, sig_w, sig_l):
    param_dict = {
        'mod': mod_dd,
        'season': ss,
        'sims': nsims,
        'V_MAX': bat_cap,
        'B_MAX': bat_rate,
        'B_MIN': -1*bat_rate,
        'S': sc,
        'CG': rc,
        'min_gen_list': ([-1*bat_rate]*6 + [ro]*15 + [-1*bat_rate]*3)*7,
        'fine': fine,
        'solar_cap': sol_cap,
        'wind_cap': wind_cap,
        'turb_cap': turb_cap,
        'sig_p': sig_p,
        'sig_w': sig_w,
        'sig_l': sig_l,
    }

    if mod_dd == 'stoc_vpp':
        df,rev,fine = models.get_stochastic_vpp_output(param_dict)
    elif mod_dd == 'stat_vpp':
        df,rev,fine = models.get_static_vpp_output(param_dict), None, None
    elif mod_dd == 'stoc_nb':
        df,rev,fine = models.get_stochastic_nb_output(param_dict)
    elif mod_dd == 'stat_nb':
        df,rev,fine = models.get_static_nb_output(param_dict), None, None
    output = [[df.to_json(orient='split'), rev, fine, json.dumps(param_dict)]]
    return output

## Model Results Header
@app.callback(
    Output('model_res_header', 'children'),
    [Input('model-result-hidden', 'children')],)
def update_header(model_data):
    if model_data == None:
        return ''
    else:
        param_dict = json.loads(model_data[3])
        mod = param_dict['mod']
        mod_txt = model_opts[mod]
        return "{} Results".format(mod_txt)


## MINI_CONTAINERS
@app.callback(
    [
        Output("blk_1", "children"),
        Output("blk_2", "children"),
        Output("blk_3", "children"),
        Output("blk_4", "children"),
    ],
    [Input("model-result-hidden", 'children')],
)
def update_text(model_data):
    if model_data == None:
        return ['', '', '', '']
    else:
        df = pd.read_json(model_data[0], orient='split')
        param_dict = json.loads(model_data[3])

        if model_data[1] == None:
            cum_prof = str(int(sum(df.hourly_profit)))
        else:
            cum_prof = str(int(sum(model_data[1])/len(model_data[1])))

        turb_starts = str(sum(df.turb_start))
        net_gen = str(int(sum(df.net_gen)))
        fines =  str(int(sum(df.fine)/param_dict['fine']))
        return " $" + cum_prof, turb_starts, net_gen + " MWh", fines

# TABS --> GRAPHS
@app.callback(Output('graph_gen', 'figure'),
              [Input('tabs', 'value'),
              Input('model-result-hidden', 'children')])
def render_content(tab, model_data):
    if model_data == None:
        return {}
    else:
        df = pd.read_json(model_data[0], orient='split')
        if tab == 'tab-1':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.solar_gen,
                name = 'Solar',
                line=dict(width=1, color='rgb(242, 113, 7)'),
                stackgroup='one',
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.wind_gen,
                name = 'Wind',
                line=dict(width=1, color='rgb(7, 166, 28)'),
                stackgroup='one',
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.turb_gen,
                name = 'Turbine',
                line=dict(width=1, color='rgb(179, 33, 7)'),
                stackgroup='one',
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.batt_flow,
                name = 'Battery',
                line=dict(width=1, color='rgb(8, 72, 156)'),
                stackgroup='one',
                visible = 'legendonly'
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.net_gen,
                line=dict(width=2, color='rgb(167, 11, 214)'),
                name = 'Net Generation',
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.net_out,
                line=dict(width=2, color='rgb(8, 72, 156)'),
                name = 'Net Power Output',
                visible = 'legendonly'
            )),
            fig.update_layout(legend_orientation="h",
                legend=dict(x=-.1, y=1.2),
                margin={'l': 30, 'b': 50, 't': 50, 'r': 20},)
            fig.update_xaxes(title_text="Time (h)")
            fig.update_yaxes(title_text="Power (MW)")
            return fig

        elif tab == 'tab-2':
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.batt_flow,
                name = 'Battery',
                line=dict(width=1, color='rgb(8, 72, 156)'),
                stackgroup='one',
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.turb_gen,
                name = 'Turbine',
                line=dict(width=1, color='rgb(179, 33, 7)'),
                stackgroup='one',
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.net_out,
                line=dict(width=2, color='rgb(8, 72, 156)'),
                name = 'Net Power Output',
                visible = 'legendonly'
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.net_gen,
                line=dict(width=2, color='rgb(167, 11, 214)'),
                name = 'Net Generation',
                visible = 'legendonly'
            )),
            fig.add_trace(go.Scatter(
                x = df.hour,
                y = df.elect_price,
                line=dict(width=2, color='rgb(201, 61, 18)'),
                name = 'Electricity Price',
                visible = 'legendonly'
            ),
            secondary_y=True,
            ),

            fig.update_layout(legend_orientation="h",
                legend=dict(x=-.1, y=1.2),
                margin={'l': 30, 'b': 50, 't': 50, 'r': 20})
            fig.update_xaxes(title_text="Time (h)")
            fig.update_yaxes(title_text="Power (MW)", secondary_y=False)
            fig.update_yaxes(title_text="Electricity Price ($)", secondary_y=True)
            return fig
            # margin={'l': 60, 'b': 40, 't': 10, 'r': 0},


## CUMULATIVE PROFIT
@app.callback(Output('cum_profit_graph', 'figure'),
              [Input('model-result-hidden', 'children')])
def render_content(model_data):
    if model_data == None:
        return {}
    else:
        df = pd.read_json(model_data[0], orient='split')
        prof = df.hourly_profit
        cum_prof = []
        for t in range(settings.T):
            cum_prof.append(sum(prof[:t]))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x = df.hour,
                y = cum_prof,
                name = 'Cumuative Profit',
                line=dict(width=1.5, color='rgb(35, 178, 222)'),
                stackgroup='one',
                ),
        )
        fig.update_layout(
            title_text="Cumulative Profit Over Time",
            margin={'l': 60, 'b': 30, 't': 50, 'r': 10},
            height=450,
            hovermode='closest',
        )
        fig.update_xaxes(title_text="Time (h)")
        fig.update_yaxes(title_text="Cumulative Profit ($)")
        return fig


## TAB SMALL GRAPHS
@app.callback(Output('profit_dist_graph', 'figure'),
              [Input('tabs_small', 'value'),
              Input('model-result-hidden', 'children')])
def render_content(tab, model_data):
    if model_data == None:
        return {}
    else:
        layout_pie = copy.deepcopy(layout)
        df = pd.read_json(model_data[0], orient='split')
        rev = model_data[1]
        param_dict = json.loads(model_data[3])
        p_sim, w_sim, l_sim, sim_perm = sim_data.get_sim_data(param_dict)
        if tab == 'tab-1':
            data = [
                dict(
                    type="pie",
                    labels=["Solar", "Wind", "Gas Turbine"],
                    values=[sum(df.solar_gen), sum(df.wind_gen), sum(df.turb_gen)],
                    name="Production Breakdown",
                    text=[
                        "Solar Energy (MWh)",
                        "Wind Energy (MWh)",
                        "Gas Turbine Energy (MWh)",
                    ],
                    hoverinfo="text+value+percent",
                    textinfo="label+percent+name",
                    hole=0.3,
                    marker=dict(colors=['rgb(230,154,71)', "rgb(119,173,113)", "rgb(217,108,89)"]),
                    # domain={"x": [0, 0.9], "y": [0.2, 0.8]},
                ),
            ]
            layout_pie["title"] = "Energy Production Summary"
            layout_pie["font"] = dict(color="#777777")
            layout_pie["legend"] = dict(
                font=dict(color="#CCCCCC", size="10")
            )
            layout_pie['paper_bgcolor']='rgba(0,0,0,0)'
            layout_pie['plot_bgcolor']='rgba(0,0,0,0)'
            # layout_pie["margin"]=dict(l=50, r=50, b=20, t=40)
            figure = dict(data=data, layout=layout_pie)
            return figure

        elif tab == 'tab-2':
            fig = go.Figure(data=[go.Histogram(x=rev)])
            fig.update_layout(
            title_text='Distribution of Simulation Profit',
            xaxis_title_text='Weekly Profit ($)',
            yaxis_title_text='Count',
            bargap=0.1,
            )
            return fig

        elif tab == 'tab-3':
            fig = go.Figure()
            cols = p_sim.columns[2:]
            for col in cols:
                fig.add_trace(go.Scatter(x=df.hour, y=p_sim[col],
                            mode='lines',
                            name='sim_%s'%col))
            fig.add_trace(go.Scatter(x=df.hour, y=p_sim['solar_fcst'],
                        mode='lines',
                        name='actual_fcst',
                        line=dict(width=2.5, color='rgb(0, 0, 0)')))
            fig.update_layout(
            xaxis_title_text='Time (h)',
            yaxis_title_text='Solar Prod (MW)',
            legend_orientation="h", legend=dict(x=-.1, y=1.2))
            return fig

        elif tab == 'tab-4':
            fig = go.Figure()
            cols = w_sim.columns[2:]
            for col in cols:
                fig.add_trace(go.Scatter(x=df.hour, y=w_sim[col],
                            mode='lines',
                            name='sim_%s'%col))
            fig.add_trace(go.Scatter(x=df.hour, y=w_sim['wind_fcst'],
                        mode='lines',
                        name='actual_fcst',
                        line=dict(width=2.5, color='rgb(0, 0, 0)')))
            fig.update_layout(
            xaxis_title_text='Time (h)',
            yaxis_title_text='Wind Prod (MW)',
            legend_orientation="h", legend=dict(x=-.1, y=1.2))
            return fig

        elif tab == 'tab-5':
            fig = go.Figure()
            cols = l_sim.columns[2:]
            for col in cols:
                fig.add_trace(go.Scatter(x=df.hour, y=l_sim[col],
                            mode='lines',
                            name='sim_%s'%col))
            fig.add_trace(go.Scatter(x=df.hour, y=l_sim['elect_fcst'],
                        mode='lines',
                        name='actual_fcst',
                        line=dict(width=2.5, color='rgb(0, 0, 0)')))
            fig.update_layout(
            xaxis_title_text='Time (h)',
            yaxis_title_text='Electricity Price ($/kW)',
            legend_orientation="h", legend=dict(x=-.1, y=1.2))
            return fig

## Download Link
@app.callback(
    Output('download-link', 'href'),
    [Input('model-result-hidden', 'children')])
def update_download_link(model_data):
    if model_data == None:
        return ''
    else:
        df = pd.read_json(model_data[0], orient='split')
        csv_string = df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string


## table
@app.callback(
    Output('model_table', 'children'),
    [Input('model-result-hidden', 'children'),
])
def update_table(model_data):
    if model_data == None:
        return ''
    else:
        df = pd.read_json(model_data[0], orient='split')
        df_dict = df.to_dict('records')
        for item in df_dict:
            for k in item:
                if item[k] == int(item[k]):
                    item[k] = int(item[k])
                else:
                    item[k] = round(item[k],2)

        return html.Div([
                dt.DataTable(
    				id='table',
    				columns=[{"name": i, "id": i} for i in df.columns],
    				data=df_dict,
                    # filter_action='native',
                    sort_action='native',
                    # page_action='native',
                    # page_size= 24,
    				style_cell={'width': '100px',
    				'height': '40px',
    				'textAlign': 'left'},
                    style_table={
                        'maxHeight': '500px',
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid'
                    },
                )
        ])


if __name__ == '__main__':
    app.run_server(debug=True)
