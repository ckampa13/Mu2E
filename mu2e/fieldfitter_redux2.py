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

        In [12]: ff = FieldFitter(hpg.get_toy())

        In [13]: ff.fit(cfg_params, cfg_pickle)
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
from copy import deepcopy
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
    def __init__(self, input_data):
        self.input_data = input_data
        self.pickle_path = mu2e_ext_path+'fit_params/'

    def fit(self, cfg_params, cfg_pickle, iterative=False):
        if iterative:
            self.fit_iter(cfg_params, cfg_pickle)
        else:
            self.fit_simple(cfg_params, cfg_pickle)

    def fit_simple(self, cfg_params, cfg_pickle):
        """Helper function that chooses one of the subsequent fitting functions."""
        self.prep_fit_func(cfg_params, cfg_pickle)
        self.fit_solenoid(cfg_params, cfg_pickle)

    def fit_iter(self, cfg_params, cfg_pickle):
        """Helper function that chooses one of the subsequent fitting functions."""
        self.prep_fit_func(cfg_params, cfg_pickle)

        # self.fit_solenoid(cfg_params, cfg_pickle)

        # # check if any param uncertainties could not be estimated. if yes, fit again with them set to not vary
        # self.N_nan_stderr = self.set_nan_stderr_non_vary()
        # if self.N_nan_stderr > 0:
        #     self.fit_solenoid(cfg_params, cfg_pickle)
        # self.redchi_hist = []
        # self.i_fit = 0
        self.redchi_best = np.inf
        self.redchi_dec = True
        self.params_best = deepcopy(self.params)
        self.N_nan_stderr = 1
        # repeat fit up to 2 extra times
        #while (self.N_nan_stderr > 0) and (self.i_fit < 3):
        while (self.N_nan_stderr > 0) and (self.redchi_dec):
            self.fit_solenoid(cfg_params, cfg_pickle)
            redchi = self.result.redchi
            if redchi < self.redchi_best:
                self.redchi_dec = True
                self.redchi_best = redchi
                self.params_best = deepcopy(self.params)
                try:
                    self.correl_dict_best = deepcopy(self.correl_dict)
                except:
                    pass
            else:
                self.redchi_dec = False
            # self.N_nan_stderr, self.non_vary_list = self.set_nan_stderr_non_vary()
            self.N_nan_stderr = self.set_nan_stderr_non_vary()
            # self.i_fit += 1
        # if we start doing worse, go to best params for final fit
        if not self.redchi_dec:
            # self.revert_nan_stderr_vary(self.non_vary_list, self.params_best = deepcopy(self.params))
            # final fit
            self.params = deepcopy(self.params_best)
            # reset ?
            # for k in self.params.keys():
            #     if self.params[k].vary:
            #         self.params[k].value = 0.
            self.fit_solenoid(cfg_params, cfg_pickle)


    def prep_fit_func(self, cfg_params, cfg_pickle):
        func_version = cfg_params.version
        # self.Bz           = []
        # self.Br           = []
        # self.Bphi         = []
        # self.RR           = []
        # self.ZZ           = []
        # self.PP           = []
        # self.XX           = []
        # self.YY           = []

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

        self.ZZ = self.input_data.Z.values
        self.RR = self.input_data.R.values
        self.PP = self.input_data.Phi.values
        self.Bz = self.input_data.Bz.values
        self.Br = self.input_data.Br.values
        self.Bphi = self.input_data.Bphi.values
        self.XX = self.input_data.X.values
        self.YY = self.input_data.Y.values

        # Choose the type of fitting function we'll be using.
        pvd = self.params.valuesdict()  # Quicker way to grab params and init the fit functions

        if func_version == 1000:
            self.fit_func = ff.brzphi_3d_producer_giant_function(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        elif func_version == 1001:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1001(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        elif func_version == 1002:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1002(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        else:
            raise NotImplementedError(f'Function version={func_version} not implemented.')

        # Start loading in additional parameters based on the function version.
        # This is coded POORLY.  THIS SHOULD BE HANDLED IN `hallprobesim`, not here!
        if func_version == 1000:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        elif func_version == 1001:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1001(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        elif func_version == 1002:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1002(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)


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
        # Bz           = []
        # Br           = []
        # Bphi         = []
        # RR           = []
        # ZZ           = []
        # PP           = []
        # XX           = []
        # YY           = []

        # # Load pre-defined starting values for parameters, or start a new set
        # if cfg_pickle.recreate:
        #     try:
        #         self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
        #                                     "rb"))
        #     except UnicodeDecodeError:
        #         self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
        #                                     "rb"), encoding='latin1')
        # elif cfg_pickle.use_pickle:
        #     try:
        #         self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
        #                                     "rb"))
        #     except UnicodeDecodeError:
        #         self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
        #                                     "rb"), encoding='latin1')
        #     self.add_params_default(cfg_params)
        # else:
        #     self.params = Parameters()
        #     self.add_params_default(cfg_params)

        # ZZ = self.input_data.Z.values
        # RR = self.input_data.R.values
        # PP = self.input_data.Phi.values
        # Bz = self.input_data.Bz.values
        # Br = self.input_data.Br.values
        # Bphi = self.input_data.Bphi.values
        # XX = self.input_data.X.values
        # YY = self.input_data.Y.values

        # # Choose the type of fitting function we'll be using.
        # pvd = self.params.valuesdict()  # Quicker way to grab params and init the fit functions

        # if func_version == 1000:
        #     self.fit_func = ff.brzphi_3d_producer_giant_function(
        #         ZZ, RR, PP,
        #         pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
        #         pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
        #         pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
        #         pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        # elif func_version == 1001:
        #     self.fit_func = ff.brzphi_3d_producer_giant_function_v1001(
        #         ZZ, RR, PP,
        #         pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
        #         pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
        #         pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
        #         pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        # elif func_version == 1002:
        #     self.fit_func = ff.brzphi_3d_producer_giant_function_v1002(
        #         ZZ, RR, PP,
        #         pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
        #         pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
        #         pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
        #         pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        # else:
        #     raise NotImplementedError(f'Function version={func_version} not implemented.')

        # Generate an lmfit Model
        self.mod = Model(self.fit_func, independent_vars=['r', 'z', 'phi', 'x', 'y'])

        # Start loading in additional parameters based on the function version.
        # This is coded POORLY.  THIS SHOULD BE HANDLED IN `hallprobesim`, not here!

        # if func_version == 1000:
        #     self.add_params_hel(1)
        #     self.add_params_hel(2)
        #     self.add_params_cyl(1)
        #     self.add_params_cyl(2)
        #     self.add_params_cart_simple(cfg_params)
        #     self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        # elif func_version == 1001:
        #     self.add_params_hel(1)
        #     self.add_params_hel(2)
        #     self.add_params_cyl_v1001(1)
        #     self.add_params_cyl(2)
        #     self.add_params_cart_simple(cfg_params)
        #     self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        # elif func_version == 1002:
        #     self.add_params_hel(1)
        #     self.add_params_hel(2)
        #     self.add_params_cyl_v1002(1)
        #     self.add_params_cyl(2)
        #     self.add_params_cart_simple(cfg_params)
        #     self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)

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
            self.result = self.mod.fit(np.concatenate([self.Br, self.Bz, self.Bphi]).ravel(),
                                       r=self.RR, z=self.ZZ, phi=self.PP, x=self.XX, y=self.YY, params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 1})
        # FIXME! I think we can remove this elif block. The else is exactly the same and should cover it.
        elif cfg_pickle.use_pickle:
            # mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            self.result = self.mod.fit(np.concatenate([self.Br, self.Bz, self.Bphi]).ravel(),
                                       # weights=np.concatenate([mag, mag, mag]).ravel(),
                                       r=self.RR, z=self.ZZ, phi=self.PP, x=self.XX, y=self.YY, params=self.params,
                                       # method='leastsq', fit_kws={'maxfev': 10000})
                                       method='least_squares', fit_kws={'verbose': 1,
                                                                        'gtol': 1e-10,
                                                                        'ftol': 1e-10,
                                                                        'xtol': 1e-10,
                                                                        'loss': cfg_params.loss,
                                                                        })
        else:
            # mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            self.result = self.mod.fit(np.concatenate([self.Br, self.Bz, self.Bphi]).ravel(),
                                       # weights=np.concatenate([mag, mag, mag]).ravel(),
                                       r=self.RR, z=self.ZZ, phi=self.PP, x=self.XX, y=self.YY, params=self.params,
                                       scale_covar=False,
                                       # method='leastsq', fit_kws={'maxfev': 10000})
                                       method='least_squares', fit_kws={'verbose': 1,
                                                                        # 'gtol': 1e-15,
                                                                        # 'ftol': 1e-15,
                                                                        # #'xtol': 1e-20,
                                                                        # 'xtol': 1e-15,
                                                                        'gtol': 1e-10,
                                                                        'ftol': 1e-10,
                                                                        'xtol': 1e-10,
                                                                        'loss': cfg_params.loss,
                                       #                                  # NOT ALLOWED!
                                       #                                  #'method': 'lm', # default: trf
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
            self.pickle_correl(self.pickle_path+cfg_pickle.save_name)

    def fit_external(self, cfg_params, cfg_pickle, profile=False):
        raise NotImplementedError('Oh no! you got lazy during refactoring')

    def pickle_results(self, pickle_name='default'):
        """Pickle the resulting Parameters after a fit is performed."""

        pkl.dump(self.result.params, open(pickle_name+'_results.p', "wb"), pkl.HIGHEST_PROTOCOL)

    def pickle_correl(self, pickle_name='default'):
        """Pickle the resulting Parameters after a fit is performed."""
        # compute correlation matrix, if possible
        try:
            # v = np.sqrt(np.diag(self.result.covar))
            # #v[np.isnan(v)] = 1e6
            # outer_v = np.outer(v, v)
            # self.correlation = self.result.covar / outer_v
            # self.correlation[self.result.covar == 0] = 0
            vn = self.result.var_names
            N = len(vn)
            self.correl = np.identity(N)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        self.correl[i, j] = self.result.params[vn[i]].correl[vn[j]]

            self.correl_dict = {'variables': vn,
                               'covar': self.result.covar,
                               'correl': self.correl}

            pkl.dump(self.correl_dict, open(pickle_name+'_results_correl.p', "wb"), pkl.HIGHEST_PROTOCOL)
        except:
            print('Correlation could not be retrieved from the fit.')


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

    def set_nan_stderr_non_vary(self):
        # Note this function should only be run after a fit. Before the fit,
        # params will not be seeded and will not contain 'stderr'.
        # non_vary_list = []
        vn = self.result.var_names
        N = 0
        for v in vn:
            if (np.isnan(self.params[v].stderr)):
            # nan stderr OR value too close to zero.
            # if (np.isnan(self.params[v].stderr)) or (abs(self.params[v].value) < 1e-6):
                self.params[v].value = 0.
                self.params[v].vary = False
                N += 1
                #non_vary_list.append(v)
        return N
        #return N, non_vary_list

    # def revert_nan_stderr_vary(self, non_vary_list):
    #     for v in non_vary_list:
    #         self.params[v].vary = True

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

    def add_params_cyl_v1002(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        # np.random.seed(101)
        # why do we initialize d like this?
        # d_vals = np.linspace(0, 1, len(ns_range))[::-1]
        d_vals = 0.5*np.ones(len(ns_range))

        # kludge for now...FIXME!
        if self.params[f'ns_c{num}'].value < 2:
            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                        if f'Cc{num}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{n}', value=0, min=0, max=1, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        # seems like a weird initialization...
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Ac{num}_{m}_{n}', value=1e-6, vary=True)
                            self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1e-6, vary=True)
                            self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Cc{num}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{n}', value=1.,
                                            min=0, max=1, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0.,
                                            min=0, max=1, vary=False)
        else:
            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                        if f'Cc{num}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{n}', value=0, min=0, max=1, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0., vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0., vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Cc{num}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{n}', value=d_vals[n], vary=True)
                                            #min=0, max=1, vary=True)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=d_vals[n], vary=True)
                                            #min=0, max=1, vary=True)
                                            # min=-1, max=1, vary=True)

    def add_params_cyl_v1001(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        # np.random.seed(101)
        # why do we initialize d like this?
        # d_vals = np.linspace(0, 1, len(ns_range))[::-1]
        d_vals = 0.5*np.ones(len(ns_range))

        # kludge for now...FIXME!
        if self.params[f'ns_c{num}'].value < 2:
            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, min=0, max=1, vary=False)
                        if f'Cc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{m}_{n}', value=0, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        # seems like a weird initialization...
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Ac{num}_{m}_{n}', value=1e-6, vary=True)
                            self.params.add(f'Ac{num}_{m}_{n}', value=0.5, min=0, max=1, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Cc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{m}_{n}', value=0., vary=True)
                        #     self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                        #     # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                        #     # self.params.add(f'Bc{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=d_vals[n],
                                            min=0, max=1, vary=False)
        else:
            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Cc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{m}_{n}', value=0, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Ac{num}_{m}_{n}', value=0., vary=True)
                            self.params.add(f'Ac{num}_{m}_{n}', value=0.5, min=0, max=1, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Cc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{m}_{n}', value=0., vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=d_vals[n],
                                            min=0, max=1, vary=True)
                                            # min=-1, max=1, vary=True)


    def add_params_cyl(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        # np.random.seed(101)
        # why do we initialize d like this?
        # d_vals = np.linspace(0, 1, len(ns_range))[::-1]
        d_vals = 0.5*np.ones(len(ns_range))

        # kludge for now...FIXME!
        if self.params[f'ns_c{num}'].value < 2:
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
                        # seems like a weird initialization...
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Ac{num}_{m}_{n}', value=1e-6, vary=True)
                            self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1e-6, vary=True)
                            self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0.0,
                                            min=0, max=1, vary=False)
        else:
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
                            self.params.add(f'Ac{num}_{m}_{n}', value=0., vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0., vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=d_vals[n],
                                            min=0, max=1, vary=True)
                                            # min=-1, max=1, vary=True)

    def add_params_cart_simple(self, cfg_params):
        ks_dict = cfg_params.ks_dict
        if ks_dict is None:
            ks_dict = {}
        cart_names = [f'k{i}' for i in range(1, 11)]

        for k in cart_names:
            if k not in self.params and k in ks_dict:
                if k == 'k3':
                #     # self.params.add(k, value=ks_dict[k], min=ks_dict[k]*0.5, max=ks_dict[k]*1.5, vary=True)
                #     #self.params.add(k, value=ks_dict[k], min=0, vary=True)
                #     self.params.add(k, value=ks_dict[k], min=0, vary=False)
                # else:
                #     self.params.add(k, value=ks_dict[k], vary=True)
                    self.params.add(k, value=ks_dict[k], vary=True)
                    # self.params.add(k, value=ks_dict[k], vary=True, min=0)
                else:
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
