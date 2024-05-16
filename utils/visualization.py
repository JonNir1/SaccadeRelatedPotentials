from typing import Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_line_with_error(
        fig: go.Figure,
        x: np.array,
        y: np.ndarray,
        y_err: np.ndarray,
        color: Union[str, Tuple[int, int, int]],
        name: str,
        show_error: bool = True,
) -> go.Figure:
    rgb = extract_color(color)
    if show_error:
        fig.add_trace(
            go.Scatter(
                x=np.hstack([x, x[::-1]]),
                y=np.hstack([y + y_err, y - y_err[::-1]]),
                fill='toself',
                fillcolor=f'rgba{rgb + (0.2,)}',
                line=dict(color=f'rgba{rgb + (0.2,)}'),
                hoverinfo='skip',
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color=f'rgba{rgb + (1,)}'),
            name=name,
        )
    )
    return fig


def extract_color(color: Union[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(color, str):
        assert color.startswith('#') and len(color) == 7
        color = color.lstrip('#')
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    assert len(color) == 3
    return color
