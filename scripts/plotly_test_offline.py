#! /usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api
#py.sign_in('BroverCleveland', 'nbnj7sygqr')
init_notebook_mode()

x = np.linspace(-5,5,1000)
sinx = np.sin(x)
cosx = np.cos(x)
logx = np.log(x)

df = pd.DataFrame({'x':x, 'sinx':sinx, 'cosx':cosx, 'logx':logx})

ax = df.plot(kind='line', style=['-', '--', '-.'], legend=False)
ax.set_title('Setting Line Styles Per column in Matplotlib and pandas')
fig = ax.get_figure()

plotly_fig = tls.mpl_to_plotly( fig )
plotly_fig['layout']['showlegend'] = True
#plotly_url = py.plot(plotly_fig, filename='mpl-linestyles-per-column')
iplot(plotly_fig)
