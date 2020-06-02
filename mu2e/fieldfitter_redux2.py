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
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)

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
                                       # method='leastsq', fit_kws={'maxfev': 10000})
                                       method='least_squares', fit_kws={'verbose': 1,
                                                                        'gtol': 1e-10,
                                                                        'ftol': 1e-10,
                                                                        'xtol': 1e-10,
                                                                        })
        else:
            # mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                       # weights=np.concatenate([mag, mag, mag]).ravel(),
                                       r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                       #method='leastsq', fit_kws={'maxfev': 10000})
                                       method='least_squares', fit_kws={'verbose': 1,
                                                                        'gtol': 1e-10,
                                                                        'ftol': 1e-10,
                                                                        'xtol': 1e-10,
                                                                        })
                                       ##                                   # 'tr_solver': 'lsmr',
                                       ##                                   # 'tr_options':
                                       ##                                   # {'regularize': True}

        self.params = self.result.params
        end_time = time()
        print(("Elapsed time was %g seconds" % (end_time - start_time)))
        report_fit(self.result, show_correl=False)
        if cfg_pickle.save_pickle and not cfg_pickle.recreate:
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
                if num == 1:
                    if f'Ah{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ah{num}_{m}_{n}', value=0, vary=False)
                    else:
                        self.params[f'Ah{num}_{m}_{n}'].vary = False

                    if f'Bh{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Bh{num}_{m}_{n}', value=0, vary=False)
                    else:
                        self.params[f'Bh{num}_{m}_{n}'].vary = False

                    if f'Ch{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ch{num}_{m}_{n}', value=-1e-6, vary=True)
                    else:
                        self.params[f'Ch{num}_{m}_{n}'].vary = False

                    if f'Dh{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Dh{num}_{m}_{n}', value=-1e-6, vary=True)
                    else:
                        self.params[f'Dh{num}_{m}_{n}'].vary = False
                else:
                    if f'Ah{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ah{num}_{m}_{n}', value=-1e-6, vary=True)
                    else:
                        self.params[f'Ah{num}_{m}_{n}'].vary = False

                    if f'Bh{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Bh{num}_{m}_{n}', value=1e-6, vary=True)
                    else:
                        self.params[f'Bh{num}_{m}_{n}'].vary = False

                    if f'Ch{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ch{num}_{m}_{n}', value=0, vary=False)
                    else:
                        self.params[f'Ch{num}_{m}_{n}'].vary = False

                    if f'Dh{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Dh{num}_{m}_{n}', value=0, vary=False)
                    else:
                        self.params[f'Dh{num}_{m}_{n}'].vary = False

    def add_params_cyl(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        # np.random.seed(101)
        d_vals = np.linspace(0, 1, len(ns_range))[::-1]

        for m in ms_range:
            for n in ns_range:
                # if (n-1) % 4!= 0:
                if n == -1:
                    if f'Ac{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                    if f'Bc{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                    if f'Dc{num}_{n}' not in self.params:
                        self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                else:
                    if f'Ac{num}_{m}_{n}' not in self.params:
                        # self.params.add(f'Ac{num}_{m}_{n}', value=1e-6, vary=True)
                        self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                    if f'Bc{num}_{m}_{n}' not in self.params:
                        # self.params.add(f'Bc{num}_{m}_{n}', value=-1e-6, vary=True)
                        self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                    if f'Dc{num}_{n}' not in self.params:
                        self.params.add(f'Dc{num}_{n}', value=d_vals[n],
                                        min=0, max=1, vary=True)

    def add_params_cart_simple(self, cfg_params):
        ks_dict = cfg_params.ks_dict
        if ks_dict is None:
            ks_dict = {}
        cart_names = [f'k{i}' for i in range(1, 11)]

        for k in cart_names:
            if k not in self.params and k in ks_dict:
                self.params.add(k, value=ks_dict[k], vary=True)
            elif k not in self.params:
                self.params.add(k, value=0, vary=False)

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

    def add_params_biot_savart(self, cfg_params, recreate=False):
        xyz_tuples = cfg_params.bs_tuples
        bounds = cfg_params.bs_bounds
        if xyz_tuples is None:
            return
        if bounds is None:
            bounds = (0.1, 0.1, 5)

        for i in range(1, len(xyz_tuples)+1):
            if len(xyz_tuples[i-1]) == 3:
                x, y, z = xyz_tuples[i-1]
                vx = vy = vz = 0
                do_vary = True
            else:
                x, y, z, vx, vy, vz = xyz_tuples[i-1]
                do_vary = False

            if f'x{i}' not in self.params:
                self.params.add(f'x{i}', value=x, vary=do_vary,
                                min=x-bounds[0], max=x+bounds[0])
            elif not recreate:
                self.params[f'x{i}'].value = x
                self.params[f'x{i}'].min = x-bounds[0]
                self.params[f'x{i}'].max = x+bounds[0]

            if f'y{i}' not in self.params:
                self.params.add(f'y{i}', value=y, vary=do_vary,
                                min=y-bounds[0], max=y+bounds[0])
            elif not recreate:
                self.params[f'y{i}'].value = y
                self.params[f'y{i}'].min = y-bounds[0]
                self.params[f'y{i}'].max = y+bounds[0]

            if f'z{i}' not in self.params:
                self.params.add(f'z{i}', value=z, vary=do_vary,
                                min=z-bounds[1], max=z+bounds[1])
            elif not recreate:
                self.params[f'z{i}'].value = z
                self.params[f'z{i}'].min = z-bounds[1]
                self.params[f'z{i}'].max = z+bounds[1]

            if f'vx{i}' not in self.params:
                self.params.add(f'vx{i}', value=vx, vary=do_vary,
                                min=-bounds[2], max=bounds[2])
            elif not recreate:
                self.params[f'vx{i}'].value = vx
                self.params[f'vx{i}'].min = -bounds[2]
                self.params[f'vx{i}'].max = bounds[2]

            if f'vy{i}' not in self.params:
                self.params.add(f'vy{i}', value=vy, vary=do_vary,
                                min=-bounds[2], max=bounds[2])
            elif not recreate:
                self.params[f'vy{i}'].value = vy
                self.params[f'vy{i}'].min = -bounds[2]
                self.params[f'vy{i}'].max = bounds[2]

            if f'vz{i}' not in self.params:
                self.params.add(f'vz{i}', value=vz, vary=do_vary,
                                min=-bounds[2], max=bounds[2])
            elif not recreate:
                self.params[f'vz{i}'].value = vz
                self.params[f'vz{i}'].min = -bounds[2]
                self.params[f'vz{i}'].max = bounds[2]
