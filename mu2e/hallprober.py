#! /usr/bin/env python
"""Module for generating mock FMS measurements.

The Field Mapping System (FMS) hall probe equipment will travel through the PS and DS, measuring a
sparse set of magnetic field components at regular positions.  These field measurements will then be
fed to the fitting software, which should be able to reproduce the entire magnetic field map within
a given region of interest.

The :class:`mu2e.hallprober.HallProbeGenerator` takes an external field simulation, made available
by the Mu2E collaboration, and generates a set of mock measurements, subject to many potential
variations.  The default geometry is cylindrical, but can be changed to cartesian.  The default
measurements are "perfect," such that no additional sources of error are introduced into each
component value.  There are optional functions that modify the measurements to reflect various
errors that could arise during data-taking.

This module also contains functions that assist in the full scope of operation, from field
simulation input, to hall probe data generation, to field fitting, to final plotting and analysis.

Example:
    Incomplete excerpt, see :func:`mu2e.fieldfitter.field_map_analysis` and `scripts/hallprobesim`
    for more typical use cases:

    .. code-block:: python

        # assuming config files already defined...

        In [10]: input_data = DataFileMaker(cfg_data.path, use_pickle=True).data_frame
        ...      input_data.query(' and '.join(cfg_data.conditions))

        In [11]: hpg = HallProbeGenerator(
        ...         input_data, z_steps = cfg_geom.z_steps,
        ...         r_steps = cfg_geom.r_steps, phi_steps = cfg_geom.phi_steps,
        ...         x_steps = cfg_geom.x_steps, y_steps = cfg_geom.y_steps)


        # Introduce miscalibrations
        In [12]: if cfg_geom.bad_calibration[0]:
        ...         hpg.bad_calibration(measure = True, position = False, rotation = False)
        ...      if cfg_geom.bad_calibration[1]:
        ...         hpg.bad_calibration(measure = False, position = True, rotation=False)
        ...      if cfg_geom.bad_calibration[2]:
        ...         hpg.bad_calibration(measure = False, position = False, rotation = True)

        In [13]: print hpg.get_toy().head()
        Out[13]:
        ...                  X    Y       Z        Bx   By        Bz      R       Phi  Bphi
        ...      833646 -800.0  0.0  4221.0  0.039380  0.0  1.976202  800.0  3.141593  -0.0
        ...      833650 -800.0  0.0  4321.0 -0.015489  0.0  1.985269  800.0  3.141593   0.0
        ...      833654 -800.0  0.0  4421.0 -0.068838  0.0  1.975510  800.0  3.141593   0.0
        ...      833658 -800.0  0.0  4521.0 -0.122017  0.0  1.944508  800.0  3.141593   0.0
        ...      833662 -800.0  0.0  4621.0 -0.170256  0.0  1.885879  800.0  3.141593   0.0

        ...                  Br     Bzerr     Brerr       Bphierr     Bxerr         Byerr
        ...      833646 -0.039099  0.000198  0.000004  1.000000e-15  0.000004  1.000000e-15
        ...      833650  0.015771  0.000199  0.000002  1.000000e-15  0.000002  1.000000e-15
        ...      833654  0.069119  0.000198  0.000007  1.000000e-15  0.000007  1.000000e-15
        ...      833658  0.122293  0.000194  0.000012  1.000000e-15  0.000012  1.000000e-15
        ...      833662  0.170523  0.000189  0.000017  1.000000e-15  0.000017  1.000000e-15

Notes:
    * Remove 'getter' usage (non-pythonic)
    * Static method could probably be placed elsewhere
    * Interpolation scheme is not accurate enough for this analysis, eventually replace with
        something more powerful
    * Should analysis functions be located in this module?


*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import time
import shutil
import math
import collections
import warnings
import six.moves.cPickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import Rbf
import mu2e
from mu2e.dataframeprod import DataFrameMaker
from mu2e.fieldfitter_redux2 import FieldFitter
from mu2e.mu2eplots import mu2e_plot3d
from mu2e import mu2e_ext_path
import imp
from six.moves import range
interp_studies = imp.load_source(
    'interp_studies', '/Users/brianpollack/Coding/Mu2E/scripts/FieldFitting/interp_studies.py')

warnings.simplefilter('always', DeprecationWarning)


class HallProbeGenerator(object):
    """Class for generating toy outputs for mimicing the Mu2E FMS hall probe measurements.

    This class takes an input DF, and a set of config namedtuples that contain necessary information
    regarding the desired geometry and spacing of the mock data that will be produced.  Unless
    interpolation is specified, the mock data must always be a subset (usually sparse) of the input
    data. Additional methods may be used to manipulate the output data in order to represent
    different measurement scenarios (like introducing measurement error).

    Args:
        input_data (:class:`pandas.DataFrame`): The input magnetic field datal properly formatted
            via :mod:`mu2e.dataframeprod`.
        z_steps (int or List[numbers], optional): If an int, will select `z_steps` number of evenly
            spaced z values for sampling. If a list of ints, will select those specific z values.
        x_steps (int or List[numbers], optional): If an int, will select `2*x_steps` number of
            evenly.  spaced x values for sampling, symmetrically about 0. If a list of ints, will
            select those specific x values. This arg is overridden if 'r_steps' is not None.
        y_steps (int or List[numbers], optional): If an int, will select `2*y_steps` number of
            evenly spaced y values for sampling, symmetrically about 0. If a list of ints, will
            select those specific y values. This arg is overridden if 'r_steps' is not None.
        r_steps (int or List[numbers], optional): If an int, will select `r_steps` number of evenly
            spaced r values for sampling. If a list of ints, will select those specific r values.
            This arg is overrides `x_steps` and `y_steps`.
        phi_steps (List[numbers], optional): Will select specified `phi` values, along with their
            `pi-val` counterparts. Pairs with `r_steps`.
        interpolate (bool, optional): If true, perform the interpolation procedure instead of
            selecting values directly from the input field map.

    Attributes:
        cylindrical_norm (func): Calculate norm (distance) in cylindrical coordinates.
        full_field (:class:`pandas.DataFrame`): Input data, before selection.
        sparse_field (:class:`pandas.DataFrame`): Output data, after selection is applied. This is
            the primary object that is produced by the `hallprober` class.
        r_steps (int or List[numbers]): The input arg `r_steps`.
        phi_steps (List[numbers]): The input arg `phi_steps`.
        z_steps (int or List[numbers]): The input arg `z_steps`.

    Notes:
        Some clean-up and rewrites should be done in the future, may lead to slight changes in call
        signature. Interpolation methods will be expanded shortly.

    """

    @staticmethod
    def cylindrical_norm(x1, x2):
        """Define a distance metric in cylindrical coordinates."""
        return np.sqrt(
            (x1[0, :]*np.cos(x1[1, :])-x2[0, :]*np.cos(x2[1, :]))**2 +
            (x1[0, :]*np.sin(x1[1, :])-x2[0, :]*np.sin(x2[1, :]))**2 +
            (x1[2, :]-x2[2, :])**2)

    def __init__(self, input_data, z_steps=None, x_steps=None,
                 y_steps=None, r_steps=None, phi_steps=None, interpolate=False, do2pi=False):
        self.full_field = input_data
        self.sparse_field = self.full_field
        self.r_steps = r_steps
        self.z_steps = z_steps
        self.x_steps = x_steps
        self.y_steps = y_steps
        self.phi_steps = phi_steps
        if self.phi_steps:
            self.phi_nphi_steps = list(self.phi_steps[:])
            for phi in self.phi_steps:
                if do2pi:
                    # determine phi and phi+pi
                    self.phi_nphi_steps.append(phi+np.pi)
                else:
                    # determine phi and negative phi
                    if phi == 0:
                        self.phi_nphi_steps.append(np.pi)
                    else:
                        self.phi_nphi_steps.append(phi-np.pi)

        if self.x_steps or self.y_steps:
            raise NotImplementedError('Oh no! you got lazy during refactoring')

        if interpolate is not False:
            self.interpolate_points(interpolate)
        else:
            # complicated indexing
            # (because phi values must be "close", but R and Z can be exact matches)
            # print(self.sparse_field.Phi)
            # print(self.phi_nphi_steps)
            # print(np.ravel(self.r_steps))
            self.sparse_field = self.sparse_field[
                (np.isclose(self.sparse_field.Phi.values[:, None],
                            self.phi_nphi_steps).any(axis=1)) &
                (np.isclose(self.sparse_field.R.values[:, None],
                            np.ravel(self.r_steps)).any(axis=1)) &
                (self.sparse_field.Z.isin(self.z_steps))]
            self.sparse_field = self.sparse_field.sort_values(['Z', 'R', 'Phi'])
            # print(self.z_steps)
            # print(self.phi_nphi_steps)
            # print(self.full_field.Phi.unique())
            # print(self.sparse_field)
            # input()

            # self.apply_selection('Z', z_steps)
            # if r_steps:
            #     self.apply_selection('R', r_steps)
            #     self.apply_selection('Phi', phi_steps)
            #     self.phi_steps = phi_steps
            # else:
            #     self.apply_selection('X', x_steps)
            #     self.apply_selection('Y', y_steps)

        # for mag in ['Bz', 'Br', 'Bphi', 'Bx', 'By', 'Bz']:
        #     try:
        #         self.sparse_field.eval('{0}err = abs(0.0001*{0}+1e-15)'.format(mag), inplace=True)
        #     except:
        #         pass

    def takespread(self, sequence, num):
        """Return an evenly-spaced sequence of length `num` from the input sequence.

        Args:
            sequence (collections.Sequence): A list-like object of numeric values.
            num (int): Number of desired values to be selected.

        Returns:
            spread (List[numbers]):
        """
        length = float(len(sequence))
        spread = []
        for i in range(num):
            spread.append(sequence[int(math.ceil(i * length / num))])
        return spread

    def apply_selection(self, coord, steps):
        """
        Apply selections to different coordinate values in order to create a sparse dataframe from
        the input data.

        Args:
            coord (str): Label of the dataframe column that will be queried.  Typically a positional
                coordinate ('X', 'Y', 'Z', 'R', 'Phi').
            steps (List[numbers]): If a list, those values will be taken
                (along with their respective negative or inverses, if applicable)

        Returns:
            Nothing. Generates `spare_field` class member.
        """

        if isinstance(steps, int):
            if coord in ['Z', 'R']:
                coord_vals = np.sort(self.full_field[coord].unique())
                coord_vals = self.takespread(coord_vals, steps)

            else:
                coord_vals = np.sort(self.full_field[coord].abs().unique())[:steps]
                coord_vals = np.concatenate((coord_vals, -coord_vals[np.where(coord_vals > 0)]))

        elif isinstance(steps, collections.Sequence) and type(steps) != str:
            if coord == 'Phi':
                coord_vals = []
                for step in steps:
                    coord_vals.append(step)
                    if step != 0:
                        coord_vals.append(step-np.pi)
                    else:
                        coord_vals.append(step+np.pi)
            elif coord == 'R':
                if isinstance(steps[0], collections.Sequence):
                    coord_vals = np.sort(np.unique([val for sublist in steps for val in sublist]))
                else:
                    coord_vals = steps
            elif coord in ['Z', 'X', 'Y']:
                coord_vals = steps
        elif steps == 'all':
                coord_vals = np.sort(self.full_field[coord].unique())
        else:
            raise TypeError(coord+" steps must be scalar or list of values!")

        if coord == 'R' or coord == 'Phi':
            self.sparse_field = self.sparse_field.query(
                '|'.join(['(-1e-6<'+coord+'-'+str(i)+'<1e-6)' for i in coord_vals])
            )
        else:
            self.sparse_field = self.sparse_field[self.sparse_field[coord].isin(coord_vals)]
        if len(self.sparse_field[coord].unique()) != len(coord_vals):
            print('Warning!: specified vals:')
            print(np.sort(coord_vals))
            print('remaining vals:')
            print(np.sort(self.sparse_field[coord].unique()))

    def get_toy(self):
        """Return `sparse_field`. Deprecated."""
        warnings.warn(("`get_toy()` is deprecated, please use the `sparse_field` class member"),
                      DeprecationWarning)
        return self.sparse_field

    def bad_calibration(self, measure=False, position=False, rotation=False, seed=None):
        print(seed)
        """
        Manipulate the `sparse_field` member values, in order to mimic imperfect measurement
        scenarios. By default, no manipulations are performed.

        Args:
            measure (bool, optional): Apply a hard-coded measurement error.
            position (bool, optional): Apply a hard-coded positional error.
            rotation (bool, optional): Apply a hard-coded rotational error.
            seed (int, optional): Input a seed value to generate a replicable set of error
                values.

        Returns:
            Nothing, modifies `sparse_field` class member.
        """
        if seed is None:
            # measure_sf = [1-2.03e-4, 1+1.48e-4, 1-0.81e-4, 1-1.46e-4, 1-0.47e-4]
            measure_sf = [1-2.58342250e-05, 1-5.00578244e-05, 1+4.87132812e-05, 1+7.79452585e-05,
                          1+1.85119047e-05]  # uniform(-0.0001, 0.0001)
            # pos_offset = [-1.5, 0.23, -0.62, 0.12, -0.18]
            pos_offset = [0.9557545, 0.7018995, -0.8877238, 0.3336723, -0.4361852]  # uniform(-1, 1)
            # rotation_angle = [ 0.00047985,  0.00011275,  0.00055975, -0.00112114,  0.00051197]
            # rotation_angle = [ 0.0005,  0.0004,  0.0005, 0.0003,  0.0004]
            rotation_angle = [6.58857659e-05, -9.64816467e-05, 8.92011209e-05, 4.42270175e-05,
                              -7.09926476e-05]
        else:
            np.random.seed(seed)
            measure_sf = np.random.uniform(-1e-4, 1e-4, 5)+1
            pos_offset = np.random.normal(0, 1, 5)
            rotation_angle = np.random.normal(0, 1e-4, 5)

        for phi in self.phi_steps:
            probes = self.sparse_field[np.isclose(self.sparse_field.Phi, phi)].R.unique()
            if measure:
                print('using measure sfs:', end=' ')
                print(measure_sf)
                if len(probes) > len(measure_sf):
                    raise IndexError('need more measure_sf, too many probes')
                for i, probe in enumerate(probes):
                    self.sparse_field.ix[
                        (abs(self.sparse_field.R) == probe), 'Bz'] *= measure_sf[i]
                    self.sparse_field.ix[
                        (abs(self.sparse_field.R) == probe), 'Br'] *= measure_sf[i]
                    self.sparse_field.ix[
                        (abs(self.sparse_field.R) == probe), 'Bphi'] *= measure_sf[i]

            if position:
                print('using pos offsets:', end=' ')
                print(pos_offset)
                if len(probes) > len(pos_offset):
                    raise IndexError('need more pos_offset, too many probes')
                for i, probe in enumerate(probes):
                    if probe == 0:
                        self.sparse_field.ix[
                            abs(self.sparse_field.R) == probe, 'R'] += pos_offset[i]
                    else:
                        self.sparse_field.ix[
                            abs(self.sparse_field.R) == probe, 'R'] += pos_offset[i]
                        self.sparse_field.ix[
                            abs(self.sparse_field.R) == -probe, 'R'] -= pos_offset[i]

            if rotation:
                print('using rot angles:', end=' ')
                print(rotation_angle)
                if len(probes) > len(rotation_angle):
                    raise IndexError('need more rotation_angle, too many probes')
                for i, probe in enumerate(probes):
                    tmp_Bz = self.sparse_field[self.sparse_field.R == probe].Bz
                    tmp_Br = self.sparse_field[self.sparse_field.R == probe].Br
                    self.sparse_field.ix[(abs(self.sparse_field.R) == probe), 'Bz'] = (
                        tmp_Br*np.sin(rotation_angle[i])+tmp_Bz*np.cos(rotation_angle[i]))
                    self.sparse_field.ix[(abs(self.sparse_field.R) == probe), 'Br'] = (
                        tmp_Br*np.cos(rotation_angle[i])-tmp_Bz*np.sin(rotation_angle[i]))

    def interpolate_points(self, version=1):
        """Method for obtaining required selection through interpolation.  Work in progress."""
        if version == 'load1':
            if os.path.isfile(mu2e_ext_path+'tmp_rbf.p'):
                self.sparse_field = pkl.load(open(mu2e_ext_path+'tmp_rbf.p', "rb"))
                return
        elif version == 'load2':
            if os.path.isfile(mu2e_ext_path+'tmp_phi.p'):
                self.sparse_field = pkl.load(open(mu2e_ext_path+'tmp_phi.p', "rb"))
                return
        elif version == 'load3':
            if os.path.isfile(mu2e_ext_path+'tmp_quad.p'):
                self.sparse_field = pkl.load(open(mu2e_ext_path+'tmp_quad.p', "rb"))
                return

        elif version == 1:
            row_list = []
            all_phis = []
            for phi in self.phi_steps:
                all_phis.append(phi)
                if phi == 0:
                    all_phis.append(np.pi)
                else:
                    all_phis.append(phi-np.pi)
            print('interpolating data points')
            for r in tqdm(self.r_steps[0], desc='R (mm)', leave=False):
                for p in tqdm(all_phis, desc='Phi (rads)', leave=False):
                    for z in tqdm(self.z_steps, desc='Z (mm)', leave=False):
                        x = r*math.cos(p)
                        y = r*math.sin(p)
                        field_subset = self.full_field.query(
                            '{0}<=X<={1} and {2}<=Y<={3} and {4}<=Z<={5}'.format(
                                x-100, x+100, y-100, y+100, z-100, z+100))

                        rbf = Rbf(field_subset.X, field_subset.Y, field_subset.Z,
                                  field_subset.Bz, function='linear')
                        bz = rbf(x, y, z)
                        rbf = Rbf(field_subset.X, field_subset.Y, field_subset.Z,
                                  field_subset.Bx, function='linear')
                        bx = rbf(x, y, z)
                        rbf = Rbf(field_subset.X, field_subset.Y, field_subset.Z,
                                  field_subset.By, function='linear')
                        by = rbf(x, y, z)

                        br = bx*math.cos(p)+by*math.sin(p)
                        bphi = -bx*math.sin(p)+by*math.cos(p)

                        row_list.append([r, p, z, br, bphi, bz])

            row_list = np.asarray(row_list)
            self.sparse_field = pd.DataFrame({
                'R': row_list[:, 0], 'Phi': row_list[:, 1], 'Z': row_list[:, 2],
                'Br': row_list[:, 3], 'Bphi': row_list[:, 4], 'Bz': row_list[:, 5]})

        elif version == 2:
            row_list = []
            all_phis = []
            for phi in self.phi_steps:
                all_phis.append(phi)
                if phi == 0:
                    all_phis.append(np.pi)
                else:
                    all_phis.append(phi-np.pi)
            print('interpolating data points')
            for r in tqdm(self.r_steps[0], desc='R (mm)', leave=False):
                for p in tqdm(all_phis, desc='Phi (rads)', leave=False):
                    for z in tqdm(self.z_steps, desc='Z (mm)', leave=False):
                        x = r*math.cos(p)
                        y = r*math.sin(p)
                        field_subset = self.full_field.query(
                            '{0}<=X<={1} and {2}<=Y<={3} and {4}<=Z<={5}'.format(
                                x-100, x+100, y-100, y+100, z-100, z+100))

                        _, b_lacey = interp_studies.interp_phi(field_subset, x, y, z, plot=False)
                        bx = b_lacey[0]
                        by = b_lacey[1]
                        bz = b_lacey[2]

                        br = bx*math.cos(p)+by*math.sin(p)
                        bphi = -bx*math.sin(p)+by*math.cos(p)

                        row_list.append([r, p, z, br, bphi, bz])

            row_list = np.asarray(row_list)
            self.sparse_field = pd.DataFrame({
                'R': row_list[:, 0], 'Phi': row_list[:, 1], 'Z': row_list[:, 2],
                'Br': row_list[:, 3], 'Bphi': row_list[:, 4], 'Bz': row_list[:, 5]})

        elif version == 3:
            row_list = []
            all_phis = []
            for phi in self.phi_steps:
                all_phis.append(phi)
                if phi == 0:
                    all_phis.append(np.pi)
                else:
                    all_phis.append(phi-np.pi)
            print('interpolating data points')
            for r in tqdm(self.r_steps[0], desc='R (mm)', leave=False):
                for p in tqdm(all_phis, desc='Phi (rads)', leave=False):
                    for z in tqdm(self.z_steps, desc='Z (mm)', leave=False):
                        x = r*math.cos(p)
                        y = r*math.sin(p)
                        field_subset = self.full_field.query(
                            '{0}<=X<={1} and {2}<=Y<={3} and {4}<=Z<={5}'.format(
                                x-100, x+100, y-100, y+100, z-100, z+100))

                        _, b_lacey = interp_studies.interp_phi_quad(field_subset, x, y, z,
                                                                    plot=False)
                        bx = b_lacey[0]
                        by = b_lacey[1]
                        bz = b_lacey[2]

                        br = bx*math.cos(p)+by*math.sin(p)
                        bphi = -bx*math.sin(p)+by*math.cos(p)

                        row_list.append([r, p, z, br, bphi, bz])

            row_list = np.asarray(row_list)
            self.sparse_field = pd.DataFrame({
                'R': row_list[:, 0], 'Phi': row_list[:, 1], 'Z': row_list[:, 2],
                'Br': row_list[:, 3], 'Bphi': row_list[:, 4], 'Bz': row_list[:, 5]})

        self.sparse_field = self.sparse_field[['R', 'Phi', 'Z', 'Br', 'Bphi', 'Bz']]
        if version == 1:
            pkl.dump(self.sparse_field, open(mu2e_ext_path+'tmp_rbf.p', "wb"), pkl.HIGHEST_PROTOCOL)
        elif version == 2:
            pkl.dump(self.sparse_field, open(mu2e_ext_path+'tmp_phi.p', "wb"), pkl.HIGHEST_PROTOCOL)
        elif version == 3:
            pkl.dump(self.sparse_field, open(mu2e_ext_path+'tmp_quad.p', "wb"),
                     pkl.HIGHEST_PROTOCOL)

        print('interpolation complete')


def make_fit_plots(df, cfg_data, cfg_geom, cfg_plot, name):
    """Make a series of comparison plots with the fit output and hallprobe input.

    This function takes input DFs and `namedtuple` config files, and generates a comprehensive
    set of comparison plots in 3D.  The plots are typically of the form 'B-component vs two
    positional variables', where the input hall probe measurements are displayed as a scatter plot,
    and the resulting fit is displayed as wireframe plot.  Additionally, heatmaps are produced to
    display the absolute residuals between the data and fit.  The heatmaps are produced separately
    in 'mpl' mode, or are integrated into the main plot in 'plotly' mode.

    Example:
        Incomplete excerpt, see `scripts/hallprobesim` for more typical use cases:

        .. code-block:: python

            # assuming config files already defined...

            In [12]: ff = FieldFitter(sparse_field, cfg_geom)

            In [13]: ff.fit(cfg_geom.geom, cfg_params, cfg_pickle)
            ...      # This will take some time, especially for many data points and free params

            In [14]: ff.merge_data_fit_res() # merge the results in for easy plotting

            In [15]: cfg_plot = namedtuple('cfg_plot', 'plot_type zlims save_loc sub_dir')

            In [16]: cfg_plot_plotly = cfg_plot('plotly',[-10,10],'html', None)
            ...      # make plotly plots, set limits, save loc, etc.

            In [17]: make_fit_plots(ff.input_data, cfg_data, cfg_geom, cfg_plot, name)


    Args:
       df (:class:`pandas.DataFrame`): DF that contains both the input data and the fit data.
       cfg_data (namedtuple): Data config file.
       cfg_geom (namedtuple): Geometry config file.
       cfg_plot (namedtuple): Plotting config file.
       name (str): Name of output save directory.

    Returns:
       Nothing.

    Todo:
        * Move this function to more logical module.
    """

    geom = cfg_geom.geom
    plot_type = cfg_plot.plot_type
    if geom == 'cyl':
        steps = cfg_geom.phi_steps
    if geom == 'cart':
        steps = cfg_geom.y_steps
    conditions = cfg_data.conditions

    ABC_geom = {'cyl': [['R', 'Z', 'Bz'], ['R', 'Z', 'Br'], ['R', 'Z', 'Bphi']],
                'cart': [['X', 'Z', 'Bx'], ['X', 'Z', 'By'], ['X', 'Z', 'Bz']]}

    if cfg_plot.save_loc == 'local':
        save_dir = mu2e.mu2e_ext_path+'plots/'+name
    elif cfg_plot.save_loc == 'html':
        save_dir = '/Users/brianpollack/Documents/PersonalWebPage/mu2e_plots/'+name

    for step in steps:
        for ABC in ABC_geom[geom]:
            if geom == 'cyl':
                conditions_str = ' and '.join(conditions+('Phi=={}'.format(step),))
            else:
                conditions_str = ' and '.join(conditions+('Y=={}'.format(step),))
            save_name = mu2e_plot3d(df, ABC[0], ABC[1], ABC[2], conditions=conditions_str,
                                    df_fit=True, mode=plot_type, save_dir=save_dir,
                                    do2pi=cfg_geom.do2pi)

            # If we are saving the plotly_html, we also want to download stills
            # and transfer them to the appropriate save location.
            if plot_type == 'plotly_html_img':

                init_loc = '/Users/brianpollack/Downloads/'+save_name+'.jpeg'
                final_loc = save_dir+'/'+save_name+'.jpeg'
                while not os.path.exists(init_loc):
                        print('waiting for', init_loc, 'to download')
                        time.sleep(2)
                shutil.move(init_loc, final_loc)

    if plot_type == 'mpl':
        plt.show()


def field_map_analysis(name, cfg_data, cfg_geom, cfg_params, cfg_pickle, cfg_plot, profile=False):
    """Universal function to perform all types of hall probe measurements, plots, and further
    analysis.

    Args:
        name (str): Name of output directory.
        cfg_data (namedtuple): Data config file.
        cfg_geom (namedtuple): Geometry config file.
        cfg_params (namedtuple): Fit parameter config file.
        cfg_pickle (namedtuple): Pickling config file.
        cfg_plot (namedtuple): Plotting config file.
        profile (bool, optional): If True, return data before fitting for the purposes of continuing
            on to profiling methods.

    Returns:
        If `profile==False`, returns a DF of the hall probe data, and the FieldFitter object. If
        `profile==True`, returns field components and position values.
    """

    plt.close('all')
    input_data = DataFrameMaker(cfg_data.path, input_type='pkl').data_frame
    input_data.query(' and '.join(cfg_data.conditions))
    hpg = HallProbeGenerator(input_data, z_steps=cfg_geom.z_steps,
                             r_steps=cfg_geom.r_steps, phi_steps=cfg_geom.phi_steps,
                             x_steps=cfg_geom.x_steps, y_steps=cfg_geom.y_steps,
                             interpolate=cfg_geom.interpolate, do2pi=cfg_geom.do2pi)

    seed = None
    if len(cfg_geom.bad_calibration) == 4:
        seed = cfg_geom.bad_calibration[3]

    if cfg_geom.bad_calibration[0]:
        hpg.bad_calibration(measure=True, position=False, rotation=False, seed=seed)
    if cfg_geom.bad_calibration[1]:
        hpg.bad_calibration(measure=False, position=True, rotation=False, seed=seed)
    if cfg_geom.bad_calibration[2]:
        hpg.bad_calibration(measure=False, position=False, rotation=True, seed=seed)

    hall_measure_data = hpg.sparse_field
    # print hall_measure_data.head()
    # raw_input()

    ff = FieldFitter(hall_measure_data, cfg_geom)
    if profile:
        ZZ, RR, PP, Bz, Br, Bphi = ff.fit(cfg_geom.geom, cfg_params, cfg_pickle, profile=profile)
        return ZZ, RR, PP, Bz, Br, Bphi
    else:
        ff.fit(cfg_geom.geom, cfg_params, cfg_pickle)

    ff.merge_data_fit_res()

    if cfg_plot.plot_type != 'none':
        make_fit_plots(ff.input_data, cfg_data, cfg_geom, cfg_plot, name)

    return hall_measure_data, ff


if __name__ == "__main__":
    pi = np.pi
    data_maker1 = DataFrameMaker('../datafiles/FieldMapData_1760_v5/Mu2e_DSmap', use_pickle=True)
    r_steps = [25, 225, 425, 625, 800]
    phi_steps = [(i/8.0)*np.pi for i in range(-7, 9)]
    z_steps = list(range(5021, 13021, 50))
    hpg = HallProbeGenerator(data_maker1.data_frame,
                             z_steps=z_steps, r_steps=r_steps, phi_steps=phi_steps)
