#! /usr/bin/env python
"""Module for fitting magnetic field data with a parametric expression.

This is the main workhorse module for fitting the magnetic field data.  It is assumed that the data
has passed through the :class:`mu2e.dataframeprod.DataFrameMaker`, and thus in the expected format.
In most cases the data has also passed through the :mod:`mu2e.hallprober` module, to imitate the act
of surveying one of the Mu2E solenoids with a series of hall probes.  The
:class:`mu2e.fieldfitter.FieldFitter` preps the data, flattens it into 1D, and uses the :mod:`lmfit`
package extensively for parameter-handling, fitting, optimization, etc.

Example:
    Incomplete excerpt, see :func:`mu2e.hallprober.field_map_analysis` and `scripts/hallprobesim`
    for more typical use cases:

    .. code-block:: python

        # assuming config files already defined...

        In [10]: input_data = DataFileMaker(cfg_data.path, use_pickle=True).data_frame
        ...      input_data.query(' and '.join(cfg_data.conditions))

        In [11]: hpg = HallProbeGenerator(
        ...         input_data, z_steps = cfg_geom.z_steps,
        ...         r_steps = cfg_geom.r_steps, phi_steps = cfg_geom.phi_steps,
        ...         x_steps = cfg_geom.x_steps, y_steps = cfg_geom.y_steps)

        In [12]: ff = FieldFitter(hpg.get_toy(), cfg_geom)

        In [13]: ff.fit(cfg_geom.geom, cfg_params, cfg_pickle)
        ...      # This will take some time, especially for many data points and free params

        In [14]: ff.merge_data_fit_res() # merge the results in for easy plotting

        In [15]: make_fit_plots(ff.input_data, cfg_data, cfg_geom, cfg_plot, name)
        ...      # defined in :class:`mu2e.hallprober`

*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
from time import time
import numpy as np
import six.moves.cPickle as pkl
import pandas as pd
from lmfit import Model, Parameters, report_fit
from mu2e import mu2e_ext_path
from mu2e.tools import fit_funcs_redux as ff


class FieldFitter:
    """Input field measurements, perform parametric fit, return relevant quantities.

    The :class:`mu2e.fieldfitter.FieldFitter` takes a 3D set of field measurements and their
    associated position values, and performs a parametric fit.  The parameters and fit model are
    handled by the :mod:`lmfit` package, which in turn wraps the :mod:`scipy.optimize` module, which
    actually performs the parameter optimization.  The default optimizer is the Levenberg-Marquardt
    algorithm.

    The :func:`mu2e.fieldfitter.FieldFitter.fit` requires multiple cfg `namedtuples`, and performs
    the actual fitting (or recreates a fit for a given set of saved parameters).  After fitting, the
    generated class members can be used for further analysis.

    Args:
        input_data (pandas.DataFrame): DF that contains the field component values to be fit.
        cfg_geom (namedtuple): namedtuple with the following members:
            'geom z_steps r_steps phi_steps x_steps y_steps bad_calibration'

    Attributes:
        input_data (pandas.DataFrame): The input DF, with possible modifications.
        phi_steps (List[float]): The axial values of the field data (cylindrial coords)
        r_steps (List[float]): The radial values of the field data (cylindrial coords)
        x_steps (List[float]): The x values of the field data (cartesian coords)
        y_steps (List[float]): The y values of the field data (cartesian coords)
        pickle_path (str): Location to read/write the pickled fit parameter values
        params (lmfit.Parameters): Set of Parameters, inherited from `lmfit`
        result (lmfit.ModelResult): Container for resulting fit information, inherited from `lmfit`

    """
    def __init__(self, input_data, cfg_geom):
        self.input_data = input_data
        if cfg_geom.geom == 'cyl':
            self.phi_steps = cfg_geom.phi_steps
            self.r_steps = cfg_geom.r_steps
        elif cfg_geom.geom == 'cart':
            self.x_steps = cfg_geom.x_steps
            self.y_steps = cfg_geom.y_steps
        self.pickle_path = mu2e_ext_path+'fit_params/'
        self.geom = cfg_geom.geom

    def fit(self, geom, cfg_params, cfg_pickle):
        """Helper function that chooses one of the subsequent fitting functions."""

        self.fit_solenoid(cfg_params, cfg_pickle)

    def fit_solenoid(self, cfg_params, cfg_pickle):
        """Main fitting function for FieldFitter class.

        The typical magnetic field geometry for the Mu2E experiment is determined by one or more
        solenoids, with some contaminating external fields.  The purpose of this function is to fit
        a set of sparse magnetic field data that would, in practice, be generated by a field
        measurement device.

        The following assumptions must hold for the input data:
           * The data is represented in a cylindrical coordiante system.
           * The data forms a series of planes, where all planes intersect at R=0.
           * All planes has the same R and Z values.
           * All positive Phi values have an associated negative phi value, which uniquely defines a
             single plane in R-Z space.

        Args:
           cfg_params (namedtuple): 'ns ms cns cms Reff func_version'
           cfg_pickle (namedtuple): 'use_pickle save_pickle load_name save_name recreate'

        Returns:
            Nothing.  Generates class attributes after fitting, and saves parameter values, if
            saving is specified.
        """
        func_version = cfg_params.version
        Bz           = []
        Br           = []
        Bphi         = []
        RR           = []
        ZZ           = []
        PP           = []
        XX           = []
        YY           = []

        # Load pre-defined starting values for parameters, or start a new set
        if cfg_pickle.recreate:
            try:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"))
            except UnicodeDecodeError:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"), encoding='latin1')
        elif cfg_pickle.use_pickle:
            try:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"))
            except UnicodeDecodeError:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"), encoding='latin1')
            self.add_params_default(cfg_params)
        else:
            self.params = Parameters()
            self.add_params_default(cfg_params)

        ZZ = self.input_data.Z.values
        RR = self.input_data.R.values
        PP = self.input_data.Phi.values
        Bz = self.input_data.Bz.values
        Br = self.input_data.Br.values
        Bphi = self.input_data.Bphi.values
        XX = self.input_data.X.values
        YY = self.input_data.Y.values

        # Choose the type of fitting function we'll be using.
        pvd = self.params.valuesdict()  # Quicker way to grab params and init the fit functions

        if func_version == 1000:
            fit_func = ff.brzphi_3d_producer_giant_function(
                ZZ, RR, PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        else:
            raise NotImplementedError(f'Function version={func_version} not implemented.')

        # Generate an lmfit Model
        self.mod = Model(fit_func, independent_vars=['r', 'z', 'phi', 'x', 'y'])

        # Start loading in additional parameters based on the function version.
        # This is coded POORLY.  THIS SHOULD BE HANDLED IN `hallprobesim`, not here!

        if func_version == 1000:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(on_list=['k3'])
            # self.add_params_cart_simple(all_on=True)
            self.add_params_biot_savart(xyz_tuples=(
                (0.25, 0, -46),
                (0.25, 0, 46)),
                xy_bounds=0.1, z_bounds=1, v_bounds=5)

        if not cfg_pickle.recreate:
            print(f'fitting with func_version={func_version},')
            print(cfg_params)

        else:
            print(f'recreating fit with func_version={func_version},')
            print(cfg_params)
        start_time = time()

        if cfg_pickle.recreate:
            for param in self.params:
                self.params[param].vary = False
            self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                       r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 1})
        elif cfg_pickle.use_pickle:
            # mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                       # weights=np.concatenate([mag, mag, mag]).ravel(),
                                       r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 10000})
                                       # method='least_squares')
        else:
            # mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                       # weights=np.concatenate([mag, mag, mag]).ravel(),
                                       r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 10000})
                                       #method='least_squares', fit_kws={'verbose': 1,
                                       #                                 'gtol': 1e-16,
                                       #                                 'ftol': 1e-16,
                                       #                                 'xtol': 1e-16,
                                       ##                                   # 'tr_solver': 'lsmr',
                                       ##                                   # 'tr_options':
                                       ##                                   # {'regularize': True}
                                       #                                })

        self.params = self.result.params
        end_time = time()
        print(("Elapsed time was %g seconds" % (end_time - start_time)))
        report_fit(self.result, show_correl=False)
        if cfg_pickle.save_pickle:  # and not cfg_pickle.recreate:
            self.pickle_results(self.pickle_path+cfg_pickle.save_name)

    def fit_external(self, cfg_params, cfg_pickle, profile=False):
        raise NotImplementedError('Oh no! you got lazy during refactoring')

    def pickle_results(self, pickle_name='default'):
        """Pickle the resulting Parameters after a fit is performed."""

        pkl.dump(self.result.params, open(pickle_name+'_results.p', "wb"), pkl.HIGHEST_PROTOCOL)

    def merge_data_fit_res(self):
        """Combine the fit results and the input data into one dataframe for easier
        comparison of results.

        Adds three columns to input_data: `Br_fit, Bphi_fit, Bz_fit` or `Bx_fit, By_fit, Bz_fit`,
        depending on the geometry.
        """
        bf = self.result.best_fit

        self.input_data.loc[:, 'Br_fit'] = bf[0:len(bf)//3]
        self.input_data.loc[:, 'Bz_fit'] = bf[len(bf)//3:2*len(bf)//3]
        self.input_data.loc[:, 'Bphi_fit'] = bf[2*len(bf)//3:]

    def add_params_default(self, cfg_params):
        if 'pitch1' not in self.params:
            self.params.add('pitch1', value=cfg_params.pitch1, vary=False)
        else:
            self.params['pitch1'].value = cfg_params.pitch1
        if 'ms_h1' not in self.params:
            self.params.add('ms_h1', value=cfg_params.ms_h1, vary=False)
        else:
            self.params['ms_h1'].value = cfg_params.ms_h1
        if 'ns_h1' not in self.params:
            self.params.add('ns_h1', value=cfg_params.ns_h1, vary=False)
        else:
            self.params['ns_h1'].value = cfg_params.ns_h1
        if 'pitch2' not in self.params:
            self.params.add('pitch2', value=cfg_params.pitch2, vary=False)
        else:
            self.params['pitch2'].value = cfg_params.pitch2
        if 'ms_h2' not in self.params:
            self.params.add('ms_h2', value=cfg_params.ms_h2, vary=False)
        else:
            self.params['ms_h2'].value = cfg_params.ms_h2
        if 'ns_h2' not in self.params:
            self.params.add('ns_h2', value=cfg_params.ns_h2, vary=False)
        else:
            self.params['ns_h2'].value = cfg_params.ns_h2
        if 'length1' not in self.params:
            self.params.add('length1', value=cfg_params.length1, vary=False)
        else:
            self.params['length1'].value = cfg_params.length1
        if 'ms_c1' not in self.params:
            self.params.add('ms_c1', value=cfg_params.ms_c1, vary=False)
        else:
            self.params['ms_c1'].value = cfg_params.ms_c1
        if 'ns_c1' not in self.params:
            self.params.add('ns_c1', value=cfg_params.ns_c1, vary=False)
        else:
            self.params['ns_c1'].value = cfg_params.ns_c1
        if 'length2' not in self.params:
            self.params.add('length2', value=cfg_params.length2, vary=False)
        else:
            self.params['length2'].value = cfg_params.length2
        if 'ms_c2' not in self.params:
            self.params.add('ms_c2', value=cfg_params.ms_c2, vary=False)
        else:
            self.params['ms_c2'].value = cfg_params.ms_c2
        if 'ns_c2' not in self.params:
            self.params.add('ns_c2', value=cfg_params.ns_c2, vary=False)
        else:
            self.params['ns_c2'].value = cfg_params.ns_c2

    def add_params_hel(self, num):
        ms_range = range(self.params[f'ms_h{num}'].value)
        ns_range = range(self.params[f'ns_h{num}'].value)

        for m in ms_range:
            for n in ns_range:
                if f'Ah{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Ah{num}_{m}_{n}', value=0, vary=False)
                if f'Bh{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Bh{num}_{m}_{n}', value=0, vary=False)
                if f'Ch{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Ch{num}_{m}_{n}', value=0, vary=True)
                if f'Dh{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Dh{num}_{m}_{n}', value=0, vary=True)

    def add_params_cyl(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        np.random.seed(101)
        d_vals = np.linspace(-np.pi, np.pi, len(ns_range))

        for m in ms_range:
            for n in ns_range:
                # if (n-1) % 4!= 0:
                if n == 0:
                    if f'Ac{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                    if f'Bc{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                    if f'Dc{num}_{n}' not in self.params:
                        self.params.add(f'Dc{num}_{n}', value=0, min=-1, max=1, vary=False)
                else:
                    if f'Ac{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=True)
                    if f'Bc{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=True)
                    if f'Dc{num}_{n}' not in self.params:
                        self.params.add(f'Dc{num}_{n}', value=d_vals[n], min=-np.pi, max=np.pi, vary=True)

    def add_params_cart_simple(self, all_on=False, on_list=None):
        cart_names = [f'k{i}' for i in range(1, 11)]
        if on_list is None:
            on_list = []

        for k in cart_names:
            if all_on:
                if k not in self.params:
                    self.params.add(k, value=0, vary=True)
            else:
                if k not in self.params:
                    if k == 'k3':
                        self.params.add(k, value=768, vary=(k in on_list))
                    else:
                        self.params.add(k, value=0, vary=(k in on_list))

    def add_params_finite_wire(self):
        if 'k1' not in self.params:
            self.params.add('k1', value=0, vary=True)
        if 'k2' not in self.params:
            self.params.add('k2', value=0, vary=True)
        if 'xp1' not in self.params:
            self.params.add('xp1', value=1050, vary=False, min=900, max=1200)
        if 'xp2' not in self.params:
            self.params.add('xp2', value=1050, vary=False, min=900, max=1200)
        if 'yp1' not in self.params:
            self.params.add('yp1', value=0, vary=False, min=-100, max=100)
        if 'yp2' not in self.params:
            self.params.add('yp2', value=0, vary=False, min=-100, max=100)
        if 'zp1' not in self.params:
            self.params.add('zp1', value=4575, vary=False, min=4300, max=4700)
        if 'zp2' not in self.params:
            self.params.add('zp2', value=-4575, vary=False, min=-4700, max=-4300)

    def add_params_biot_savart(self, xyz_tuples=None, v_tuples=None, xy_bounds=100, z_bounds=100,
                               v_bounds=2):
        if v_tuples and len(v_tuples) != len(xyz_tuples):
            raise AttributeError('If v_tuples is specified it must be same size as xyz_tuples')

        for i in range(1, len(xyz_tuples)+1):
            x, y, z = xyz_tuples[i-1]
            if f'x{i}' not in self.params:
                self.params.add(f'x{i}', value=x, vary=True,
                                min=x-xy_bounds, max=x+xy_bounds)
            else:
                self.params[f'x{i}'].vary = False
            if f'y{i}' not in self.params:
                self.params.add(f'y{i}', value=y, vary=True,
                                min=y-xy_bounds, max=y+xy_bounds)
            else:
                self.params[f'y{i}'].vary = False
            if f'z{i}' not in self.params:
                self.params.add(f'z{i}', value=z, vary=True,
                                min=z-z_bounds, max=z+z_bounds)
            else:
                self.params[f'z{i}'].vary = False

            if v_tuples:
                vx, vy, vz = v_tuples[i-1]
            else:
                vx = vy = vz = 0
            if f'vx{i}' not in self.params:
                self.params.add(f'vx{i}', value=vx, vary=True,
                                min=vx-v_bounds, max=vx+v_bounds)
            else:
                self.params[f'vx{i}'].vary = False
            if f'vy{i}' not in self.params:
                self.params.add(f'vy{i}', value=vy, vary=True,
                                min=vy-v_bounds, max=vy+v_bounds)
            else:
                self.params[f'vy{i}'].vary = False
            if f'vz{i}' not in self.params:
                self.params.add(f'vz{i}', value=vz, vary=True,
                                min=vz-v_bounds, max=vz+v_bounds)
            else:
                self.params[f'vz{i}'].vary = False
