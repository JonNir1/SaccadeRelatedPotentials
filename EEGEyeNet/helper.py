import os

import numpy as np
import pandas as pd
from pymatreader import read_mat

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_gaze_pixels(mat_path: str) -> go.Figure:
    mat = read_mat(mat_path)['sEEG']
    ts = mat['times']
    data = mat['data']
    labels = pd.Series(mat['chanlocs']['labels'])
    gaze = pd.DataFrame(
        np.vstack([ts, data[np.isin(labels, ['L-GAZE-X', 'L-GAZE-Y', 'R-GAZE-X', 'R-GAZE-Y', 'L-AREA', 'R-AREA'])]]),
        index=['t', 'x', 'y', 'p']
    ).T
    is_missing_gaze = np.all(gaze[['x', 'y']] == 0, axis=1)
    is_missing_pupil = gaze['p'] == 0
    gaze.loc[is_missing_gaze, ['x', 'y', 'p']] = np.nan      # replace all-zero samples with NaN
    gaze.loc[is_missing_pupil, ['x', 'y', 'p']] = np.nan

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col in ['x', 'y', 'p']:
        fig.add_trace(
            go.Scatter(x=gaze['t'], y=gaze[col], mode='lines', name=col.capitalize()),
            secondary_y=(col == 'p')
        )
    fig.add_hline(y=800, line_dash='dash', line_color='darkblue', name='X Edge', showlegend=True)
    fig.add_hline(y=600, line_dash='dash', line_color='darkred', name='Y Edge', showlegend=True)
    fig.add_annotation(
        text=f"Max X Coord: {gaze['x'].max():.2f}<br>Max Y Coord: {gaze['y'].max():.2f}",
        x=0.005, y=0.975, xref='paper', yref='paper', xanchor='left', yanchor='top', showarrow=False,
    )
    subj = os.path.basename(mat_path).split('_')[0]
    fig.update_layout(
        width=900, height=500,
        title=dict(text=f'<b>{subj}</b>', y=0.975, yanchor='top', x=0.5, xanchor='center'),
        yaxis=dict(title='Pixels'),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig
