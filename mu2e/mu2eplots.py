#! /usr/bin/env python

import os
import re
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from offline import init_notebook_mode, iplot, plot
import plotly.tools as tls
import plotly.graph_objs as go
from mpldatacursor import datacursor


# Definitions of mu2e-specific plotting functions.
# Plots can be displayed in matplotlib, plotly (offline mode), or plotly (html only)
# Plot functions all take a dataframe input

def mu2e_plot(df, x, y, conditions = None, mode = 'mpl', info = None, savename = None):
    '''Currently plotly can convert simple mpl plots directly into plotly.
    Therefor, generate the mpl first, then convert to plotly if necessary.'''

    _modes = ['mpl', 'plotly', 'plotly_html', 'plotly_nb']

    if conditions:
        df = df.query(conditions)

    if mode not in _modes:
        raise ValueError(mode+' not in '+_modes)

    ax = df.plot(x, y, kind='line')
    ax.grid(True)
    plt.ylabel(y)
    plt.title(' '.join(filter(lambda x:x, [info, x, 'v', y, conditions])))
    if mode == 'mpl':
        plt.legend()

    elif 'plotly' in mode:
        fig = ax.get_figure()
        py_fig = tls.mpl_to_plotly(fig)
        py_fig['layout']['showlegend'] = True

        if mode == 'plotly_nb':
            init_notebook_mode()
            iplot(py_fig)
        elif mode == 'plotly_html':
            plot(py_fig)

    if savename:
        plt.savefig(savename)

def mu2e_plot3d(df, x, y, z, conditions = None, mode = 'mpl', info = None, save_dir = None, df_fit = None):
    from mpl_toolkits.mplot3d import Axes3D
    del Axes3D
    '''Currently, plotly cannot convert 3D mpl plots directly into plotly (without a lot of work).
    For now, the mpl and plotly generation are seperate (may converge in future if necessary).'''

    _modes = ['mpl', 'plotly', 'plotly_html', 'plotly_nb']
    save_name = None

    if conditions:

        # Special treatment to detect cylindrical coordinates:
        # Some regex matching to find any 'Phi' condition and treat it as (Phi & Phi-Pi).
        # This is done because when specifying R-Z coordinates, we almost always want the
        # 2-D plane that corresponds to that (Phi & Phi-Pi) pair.  In order to plot this,
        # we assign a value of -R to all R points that correspond to the 'Phi-Pi' half-plane

        p = re.compile(r'(?:and)*\s*Phi\s*==\s*([-+]?(?:[0-9]*\.[0-9]+|[0-9]+))')
        phi_str = p.search(conditions)
        conditions_nophi = p.sub('', conditions)
        conditions_nophi = re.sub(r'^\s*and\s*','',conditions_nophi)
        conditions_nophi = conditions_nophi.strip()
        try:
            phi = float(phi_str.group(1))
        except:
            phi = None

        df = df.query(conditions_nophi)

        # Make radii negative for negative phi values (for plotting purposes)
        if phi != None:
            isc = np.isclose
            if isc(phi,0):
                nphi = np.pi
            else:
                nphi = phi-np.pi
            df = df[(isc(phi,df.Phi)) | (isc(nphi,df.Phi))]
            df.ix[isc(nphi,df.Phi), 'R']*=-1

        conditions_title = conditions_nophi.replace(' and ', ', ')
        conditions_title = conditions_title.replace('R!=0','')
        conditions_title = conditions_title.strip()
        conditions_title = conditions_title.strip(',')
        if phi != None:
            conditions_title +=', Phi=={0:.2f}'.format(phi)

    if mode not in _modes:
        raise ValueError(mode+' not in '+_modes)

    piv = df.pivot(x, y, z)
    X=piv.index.values
    Y=piv.columns.values
    Z=np.transpose(piv.values)
    Xi,Yi = np.meshgrid(X, Y)
    if df_fit:
        piv_fit = df.pivot(x, y, z+'_fit')
        Z_fit=np.transpose(piv_fit.values)
        data_fit_diff = (Z - Z_fit)*10000
        Xa = np.concatenate(([X[0]],0.5*(X[1:]+X[:-1]),[X[-1]]))
        Ya = np.concatenate(([Y[0]],0.5*(Y[1:]+Y[:-1]),[Y[-1]]))

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #save_name = '{0}_{1}{2}_{3}'.format(z,x,y,'_'.join([i for i in conditions.split() if i!='and']))
        save_name = '{0}_{1}{2}_{3}'.format(z,x,y,'_'.join([i for i in conditions_title.split(', ') if i!='and']))
        save_name = re.sub(r'[<>=!]', '', save_name)

        if df_fit:
            save_name += '_fit'

    if mode == 'mpl':
        fig = plt.figure().gca(projection='3d')

        if df_fit:
            fig.plot(Xi.ravel(), Yi.ravel(), Z.ravel(), 'ko',markersize=2 )
            fig.plot_wireframe(Xi, Yi, Z_fit,color='green')
        else:
            fig.plot_surface(Xi, Yi, Z, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'), linewidth=0, antialiased=False)

        #fig.zaxis.set_major_locator(LinearLocator(10))
        #fig.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        plt.xlabel(x+' (mm)', fontsize=18)
        plt.ylabel(y+' (mm)', fontsize=18)
        fig.set_zlabel(z+' (T)', fontsize=18)
        #fig.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        #fig.zaxis.get_major_formatter().set_useOffset(True)
        fig.ticklabel_format(style='sci', axis='z')
        fig.zaxis.labelpad=20
        fig.xaxis.labelpad=20
        fig.yaxis.labelpad=20
        fig.zaxis.set_tick_params(direction='out',pad=10)
        plt.title('{0} vs {1} and {2} for DS\n{3}'.format(z,x,y,conditions_title), fontsize=20)
        fig.view_init(elev=35., azim=30)
        if save_dir:
            plt.savefig(save_dir+'/'+save_name+'.png')

        if df_fit:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            max_val = np.max(data_fit_diff)
            min_val = np.min(data_fit_diff)
            if (max(abs(max_val), abs(min_val))) >5:
                heat = ax2.pcolormesh(Xa,Ya,data_fit_diff,vmin=-5,vmax=5,cmap=plt.get_cmap('viridis'))
                cb = plt.colorbar(heat, aspect=7)
                cb_ticks = cb.ax.get_yticklabels()
                cb_ticks[0] = '< -5'
                cb_ticks[-1] = '> 5'
                cb_ticks = cb.ax.set_yticklabels(cb_ticks)
            else:
                heat = ax2.pcolormesh(Xa,Ya,data_fit_diff,cmap=plt.get_cmap('viridis'))
                cb = plt.colorbar(heat, aspect=7)
            plt.title('{0} vs {1} and {2} for DS\n{3}'.format(z,x,y,conditions_title), fontsize=20)
            cb.set_label('Data-Fit (G)', fontsize=18)
            ax2.set_xlabel(x+' (mm)', fontsize=18)
            ax2.set_ylabel(y+' (mm)', fontsize=18)
            datacursor(heat, hover=True, bbox=dict(alpha=1, fc='w'))
            if save_dir:
                plt.savefig(save_dir+'/'+save_name+'_heat.pdf')


    elif 'plotly' in mode:
        axis_title_size = 18
        axis_tick_size = 14
        layout = go.Layout(
                        title='{0} vs {1} and {2} for DS<br>{3}'.format(z,x,y,conditions_title),
                        titlefont=dict(size=30),
                        autosize=False,
                        width=800,
                        height=650,
                        scene=dict(
                                xaxis=dict(
                                        title='{} (mm)'.format(x),
                                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                                        tickfont=dict(size=axis_tick_size),
                                        gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=True,
                                        backgroundcolor='rgb(230, 230,230)',
                                        ),
                                yaxis=dict(
                                        title='{} (mm)'.format(y),
                                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                                        tickfont=dict(size=axis_tick_size),
                                        gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=True,
                                        backgroundcolor='rgb(230, 230,230)',
                                        ),
                                zaxis=dict(
                                        title='{} (T)'.format(z),
                                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                                        tickfont=dict(size=axis_tick_size),
                                        gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=True,
                                        backgroundcolor='rgb(230, 230,230)',
                                        ),
                                ),
                        showlegend=True,
                        legend=dict(x=0.8,y=0.9, font=dict(size=18, family='Overpass')),
                        )
        if df_fit:
            scat = go.Scatter3d(x=Xi.ravel(), y=Yi.ravel(), z=Z.ravel(),
                mode='markers',
                marker=dict(size=3, color='rgb(0, 0, 0)',
                            line=dict(color='rgb(0, 0, 0)'), opacity=1),
                name = 'Data')
            lines = [scat]
            line_marker = dict(color='green', width=2)
            do_leg = True
            for i, j, k in zip(Xi, Yi, Z_fit):
                if do_leg:
                    lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines',
                        line=line_marker,name='Fit',legendgroup='fitgroup'))
                else:
                    lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines',
                        line=line_marker, name='Fit',legendgroup='fitgroup',showlegend=False))
                do_leg = False

            z_offset=(np.min(Z)-abs(np.min(Z)*0.3))*np.ones(Z.shape)
            textz=[['x: '+'{:0.5f}'.format(Xi[i][j])+'<br>y: '+'{:0.5f}'.format(Yi[i][j])+
                '<br>z: '+'{:0.5f}'.format(data_fit_diff[i][j]) for j in
                range(data_fit_diff.shape[1])] for i in range(data_fit_diff.shape[0])]
            proj_z=lambda x, y, z: z #projection in the z-direction
            colorsurfz=proj_z(Xi,Yi,data_fit_diff)
            tracez = go.Surface(z=z_offset,
                x=Xi,
                y=Yi,
                colorscale='Viridis',
                colorbar=dict(title='Data-Fit (G)',
                    titlefont=dict(size=18),
                    tickfont=dict(size=20),
                    xanchor='center'),
                zmin=-2,
                zmax=2,
                name = 'residual',
                showlegend=True,
                showscale=True,
                surfacecolor=colorsurfz,
                text=textz,
                hoverinfo='text',
               )
            lines.append(tracez)


        else:
            surface = go.Surface(x=Xi, y=Yi, z=Z,
                colorbar = go.ColorBar(title='Tesla',titleside='right'), colorscale = 'Viridis')
            lines = [surface]

        fig = go.Figure(data=lines, layout=layout)

        if mode == 'plotly_nb':
            init_notebook_mode()
            iplot(fig)
        elif mode == 'plotly_html_img':
            if save_dir:
                plot(fig, filename=save_dir+'/'+save_name+'.html', image='jpeg', image_filename = save_name)
            else:
                plot(fig)
        elif mode == 'plotly_html':
            if save_dir:
                plot(fig, filename=save_dir+'/'+save_name+'.html', auto_open=False)
            else:
                plot(fig)

    return save_name


