#! /usr/bin/env python

import numpy as np
from scipy.interpolate import griddata

from mu2e import mu2e_ext_path
from mu2e.dataframeprod import DataFrameMaker


def get_ds_interp_func(filename="datafiles/Mau13/Mu2e_DSMap_V13", input_type='pkl'):
    '''
    This factory function will return an interpolating function for any field map. An input x,y,z will output the corresponding Bx,By,Bz or Br,Bphi,Bz. Will decide later if linear interpolation is good enough.
    '''
    # load dataframe
    DS_DF = DataFrameMaker(mu2e_ext_path+"datafiles/Mau13/Mu2e_DSMap_V13", input_type='pkl').data_frame
    # set the spacing between points in each direction
    x_step = DS_DF.X.unique()[1] - DS_DF.X.unique()[0]
    y_step = DS_DF.Y.unique()[1] - DS_DF.Y.unique()[0]
    z_step = DS_DF.Z.unique()[1] - DS_DF.Z.unique()[0]

    # slight adjustment for if you pick right on an existing point
    x_step = x_step + 0.01*x_step
    y_step = y_step + 0.01*y_step
    z_step = z_step + 0.01*z_step

    def interp(x,y,z,method='linear',cart=True):
        '''
        Returns an interpolation of a magnetic field map at an input x,y,z.
        '''
        # cut out a 'cube'
        df_small = DS_DF[(DS_DF.X < x + x_step) & (DS_DF.X > x - x_step) & (DS_DF.Y < y + y_step) & (DS_DF.Y > y - y_step) & (DS_DF.Z < z + z_step) & (DS_DF.Z > z - z_step)]

        # calculate bx,by,bz
        bx = griddata((df_small.X.values,df_small.Y.values,df_small.Z.values),df_small.Bx.values, (x,y,z), method=method)
        by = griddata((df_small.X.values,df_small.Y.values,df_small.Z.values),df_small.By.values, (x,y,z), method=method)
        bz = griddata((df_small.X.values,df_small.Y.values,df_small.Z.values),df_small.Bz.values, (x,y,z), method=method)

        if type(bx) is np.ndarray:
            bx = bx.item()
            by = by.item()
            bz = bz.item()

        return (bx,by,bz)

    return interp
