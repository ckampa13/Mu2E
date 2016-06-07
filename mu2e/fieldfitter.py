#! /usr/bin/env python

from __future__ import division
import os
from time import time
import collections
import numpy as np
import cPickle as pkl
from lmfit import Model, Parameters, report_fit
import mu2e
import tools.fit_funcs as ff


class FieldFitter:
    """Input hall probe measurements, perform semi-analytical fit, return fit function and other stuff."""
    # def __init__(self, input_data, phi_steps = None, r_steps = None, xy_steps = None, no_save = False):
    def __init__(self, input_data, cfg_geom):
        self.input_data = input_data
        if cfg_geom.geom == 'cyl':
            self.phi_steps = cfg_geom.phi_steps
            self.r_steps= cfg_geom.r_steps
        elif cfg_geom.geom == 'cart':
            self.xy_steps = cfg_geom.xy_steps
        self.pickle_path = os.path.abspath(os.path.dirname(mu2e.__file__))+'/../fit_params/'


    def fit(self, geom, cfg_params, cfg_pickle, profile=False):
        if profile:
            return self.fit_solenoid(cfg_params, cfg_pickle, profile)
        if geom=='cyl':
            self.fit_solenoid(cfg_params, cfg_pickle, profile)
        elif geom=='cart':
            self.fit_external(cfg_params, cfg_pickle)


    def fit_solenoid(self,cfg_params, cfg_pickle, profile):
        Reff = cfg_params.Reff
        ns = cfg_params.ns
        ms = cfg_params.ms
        func_version = cfg_params.func_version
        Bz = []
        Br =[]
        Bphi = []
        Bzerr = []
        Brerr =[]
        Bphierr = []
        RR =[]
        ZZ = []
        PP = []
        for phi in self.phi_steps:
            if phi==0: nphi = np.pi
            else: nphi=phi-np.pi

            input_data_phi = self.input_data[(np.abs(self.input_data.Phi-phi)<1e-6)|(np.abs(self.input_data.Phi-nphi)<1e-6)]
            input_data_phi.ix[np.abs(input_data_phi.Phi-nphi)<1e-6, 'R']*=-1

            piv_bz = input_data_phi.pivot('Z','R','Bz')
            piv_br = input_data_phi.pivot('Z','R','Br')
            piv_bphi = input_data_phi.pivot('Z','R','Bphi')
            #piv_bz_err = input_data_phi.pivot('Z','R','Bzerr')
            #piv_br_err = input_data_phi.pivot('Z','R','Brerr')
            #piv_bphi_err = input_data_phi.pivot('Z','R','Bphierr')

            R = piv_br.columns.values
            Z = piv_br.index.values
            Bz.append(piv_bz.values)
            Br.append(piv_br.values)
            Bphi.append(piv_bphi.values)
            #Bzerr.append(piv_bz_err.values)
            #Brerr.append(piv_br_err.values)
            #Bphierr.append(piv_bphi_err.values)
            RR_slice,ZZ_slice = np.meshgrid(R, Z)
            RR.append(RR_slice)
            ZZ.append(ZZ_slice)
            if phi>np.pi/2:
                PP_slice = np.full_like(RR_slice,input_data_phi.Phi.unique()[1])
                PP_slice[:,int(PP_slice.shape[1]/2):]=input_data_phi.Phi.unique()[0]
            else:
                PP_slice = np.full_like(RR_slice,input_data_phi.Phi.unique()[0])
                PP_slice[:,int(PP_slice.shape[1]/2):]=input_data_phi.Phi.unique()[1]
            #PP_slice = np.full_like(RR_slice,input_data_phi.Phi.unique()[0])
            #PP_slice[:,PP_slice.shape[1]/2:]=input_data_phi.Phi.unique()[1]
            PP.append(PP_slice)

        ZZ = np.concatenate(ZZ)
        RR = np.concatenate(RR)
        PP = np.concatenate(PP)
        Bz = np.concatenate(Bz)
        Br = np.concatenate(Br)
        Bphi = np.concatenate(Bphi)
        #Bzerr = np.concatenate(Bzerr)
        #Brerr = np.concatenate(Brerr)
        #Bphierr = np.concatenate(Bphierr)
        if profile:
            return ZZ,RR,PP,Bz,Br,Bphi

        #brzphi_3d_fast = brzphi_3d_producer_v2(ZZ,RR,PP,Reff,ns,ms)
        if func_version==1:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel(ZZ,RR,PP,Reff,ns,ms)
        elif func_version==2:
            brzphi_3d_fast = ff.brzphi_3d_producer_bessel(ZZ,RR,PP,Reff,ns,ms)
        elif func_version==3:
            brzphi_3d_fast = ff.brzphi_3d_producer_bessel_hybrid(ZZ,RR,PP,Reff,ns,ms)
        elif func_version==4:
            brzphi_3d_fast = ff.brzphi_3d_producer_numba_v2(ZZ,RR,PP,Reff,ns,ms)
        elif func_version==5:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase(ZZ,RR,PP,Reff,ns,ms)
        else: raise KeyError('func version '+func_version+' does not exist')

        self.mod = Model(brzphi_3d_fast, independent_vars=['r','z','phi'])

        if cfg_pickle.use_pickle or cfg_pickle.recreate:
            self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',"rb"))
        else:
            self.params = Parameters()
        #delta_seeds = [0, 0.00059746, 0.00452236, 1.82217664, 1.54383364, 0.92910890, 2.3320e-6, 1.57188824, 3.02599942, 3.04222595]


        if 'R' not in self.params: self.params.add('R',value=Reff,vary=False)
        if 'ns' not in self.params: self.params.add('ns',value=ns,vary=False)
        else: self.params['ns'].value=ns
        if 'ms' not in self.params: self.params.add('ms',value=ms,vary=False)
        else: self.params['ms'].value=ms
        #if 'delta1' not in self.params: self.params.add('delta1',value=0.0,min=-np.pi,max=np.pi,vary=False)
        #else: self.params['delta1'].vary=False

        for n in range(ns):
            if func_version==5:
                if 'D_{0}'.format(n) not in self.params: self.params.add('D_{0}'.format(n),value=0,min=-np.pi*0.5,max=np.pi*0.5)
                else: self.params['D_{0}'.format(n)].vary=True
            else:
                if 'C_{0}'.format(n) not in self.params: self.params.add('C_{0}'.format(n),value=1)
                else: self.params['C_{0}'.format(n)].vary=True
                if 'D_{0}'.format(n) not in self.params: self.params.add('D_{0}'.format(n),value=0.001)
                else: self.params['D_{0}'.format(n)].vary=True
            for m in range(max(5,ms-n*10)):
            #for m in range(ms):
                if 'A_{0}_{1}'.format(n,m) not in self.params: self.params.add('A_{0}_{1}'.format(n,m),value=0,vary=True)
                else: self.params['A_{0}_{1}'.format(n,m)].vary=True
                if 'B_{0}_{1}'.format(n,m) not in self.params: self.params.add('B_{0}_{1}'.format(n,m),value=0,vary=True)
                else: self.params['B_{0}_{1}'.format(n,m)].vary=True
                if func_version==3:
                    if 'E_{0}_{1}'.format(n,m) not in self.params: self.params.add('E_{0}_{1}'.format(n,m),value=0,vary=True)
                    else: self.params['E_{0}_{1}'.format(n,m)].vary=True
                    if 'F_{0}_{1}'.format(n,m) not in self.params: self.params.add('F_{0}_{1}'.format(n,m),value=0,vary=True)
                    else: self.params['F_{0}_{1}'.format(n,m)].vary=True
                    if m>3:
                        self.params['E_{0}_{1}'.format(n,m)].vary=False
                        self.params['F_{0}_{1}'.format(n,m)].vary=False


        if not cfg_pickle.recreate: print 'fitting with n={0}, m={1}'.format(ns,ms)
        else: print 'recreating fit with n={0}, m={1}, pickle_file={2}'.format(ns,ms,cfg_pickle.load_name)
        start_time=time()
        if cfg_pickle.recreate:
            for param in self.params:
                self.params[param].vary=False
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([Brerr,Bzerr,Bphierr]).ravel(),
                r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq',fit_kws={'maxfev':1})
        elif cfg_pickle.use_pickle:
            mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            #for param in self.params:
            #    self.params[param].vary=False
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([np.ones(Br.shape,dtype=float)*1e-1,
                #    np.ones(Bz.shape,dtype=float),
                #    np.ones(Bphi.shape,dtype=float)*0.05]).ravel(),
                weights = np.concatenate([mag,mag,mag]).ravel(),
                r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq',fit_kws={'maxfev':10000})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='powell')
        else:
            mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([np.ones(Br.shape,dtype=float)*1e-1,
                #    np.ones(Bz.shape,dtype=float),
                #    np.ones(Bphi.shape,dtype=float)*0.05]).ravel(),
                weights = np.concatenate([mag,mag,mag]).ravel(),
                r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq',fit_kws={'maxfev':5000})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='lbfgsb',fit_kws={'options':{'maxiter':1000}})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='differential_evolution',fit_kws={'maxfun':1})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='nelder')

        self.params = self.result.params
        end_time=time()
        #if not cfg_pickle.recreate:
        print("Elapsed time was %g seconds" % (end_time - start_time))
        report_fit(self.result, show_correl=False)
        if cfg_pickle.save_pickle and not cfg_pickle.recreate: self.pickle_results(self.pickle_path+cfg_pickle.save_name)

    def fit_external(self,cns=1,cms=1, use_pickle = False, pickle_name = 'default', line_profile=False, recreate=False):
        a = 3e4
        b = 3e4
        c = 3e4

        Bz = []
        Bx =[]
        By = []
        Bzerr = []
        Bxerr =[]
        Byerr = []
        ZZ = []
        XX =[]
        YY = []
        for y in self.xy_steps:

            input_data_y = self.input_data[self.input_data.Y==y]
            #print input_data_phi.Phi.unique()

            piv_bz = input_data_y.pivot('Z','X','Bz')
            piv_bx = input_data_y.pivot('Z','X','Bx')
            piv_by = input_data_y.pivot('Z','X','By')
            piv_bz_err = input_data_y.pivot('Z','X','Bzerr')
            piv_bx_err = input_data_y.pivot('Z','X','Bxerr')
            piv_by_err = input_data_y.pivot('Z','X','Byerr')

            X = piv_bx.columns.values
            Z = piv_bx.index.values
            Bz.append(piv_bz.values)
            Bx.append(piv_bx.values)
            By.append(piv_by.values)
            Bzerr.append(piv_bz_err.values)
            Bxerr.append(piv_bx_err.values)
            Byerr.append(piv_by_err.values)
            XX_slice,ZZ_slice = np.meshgrid(X, Z)
            XX.append(XX_slice)
            ZZ.append(ZZ_slice)
            YY_slice = np.full_like(XX_slice,y)
            YY.append(YY_slice)

        ZZ = np.concatenate(ZZ)
        XX = np.concatenate(XX)
        YY = np.concatenate(YY)
        Bz = np.concatenate(Bz)
        Bx = np.concatenate(Bx)
        By = np.concatenate(By)
        Bzerr = np.concatenate(Bzerr)
        Bxerr = np.concatenate(Bxerr)
        Byerr = np.concatenate(Byerr)
        if line_profile:
            return ZZ,XX,YY,Bz,Bx,By

        b_external_3d_fast = ff.b_external_3d_producer(a,b,c,ZZ,XX,YY,cns,cms)
        self.mod = Model(b_external_3d_fast, independent_vars=['x','y','z'])

        if use_pickle or recreate:
            self.params = pkl.load(open(pickle_name+'_results.p',"rb"))
        else:
            self.params = Parameters()


        if 'cns' not in self.params: self.params.add('cns',value=cns,vary=False)
        else: self.params['cns'].value=cns
        if 'cms' not in self.params: self.params.add('cms',value=cms,vary=False)
        else: self.params['cms'].value=cms

        #if 'a' not in self.params: self.params.add('a',value= 3.0368e5,min=0,vary=False)
        #else: self.params['a'].vary=False
        #if 'b' not in self.params: self.params.add('b',value=83795.4340,min=0,vary=False)
        #else: self.params['b'].vary=False
        #if 'c' not in self.params: self.params.add('c',value=12354.7856,min=0,vary=False)
        #else: self.params['c'].vary=False
        if 'epsilon1' not in self.params: self.params.add('epsilon1',value=0.1,min=0,max=2*np.pi,vary=True)
        else: self.params['epsilon1'].vary=True
        if 'epsilon2' not in self.params: self.params.add('epsilon2',value=0.1,min=0,max=2*np.pi,vary=True)
        else: self.params['epsilon2'].vary=True

        for cn in range(1,cns+1):
            for cm in range(1,cms+1):
                if 'C_{0}_{1}'.format(cn,cm) not in self.params: self.params.add('C_{0}_{1}'.format(cn,cm),value=1,vary=True)
                else: self.params['C_{0}_{1}'.format(cn,cm)].vary=True
                #if 'D_{0}_{1}'.format(cn,cm) not in self.params: self.params.add('D_{0}_{1}'.format(cn,cm),value=1,vary=True)
                #else: self.params['D_{0}_{1}'.format(cn,cm)].vary=True

        if not recreate: print 'fitting external with cn={0}, cm={1}'.format(cns,cms)
        start_time=time()
        if recreate:
            for param in self.params:
                self.params[param].vary=False
            self.result = self.mod.fit(np.concatenate([Bx,By,Bz]).ravel(),
                #weights = np.concatenate([Bxerr,Byerr,Bzerr]).ravel(),
                x=XX, y=YY, z=ZZ, params = self.params, method='leastsq',fit_kws={'maxfev':1})
        elif use_pickle:
            self.result = self.mod.fit(np.concatenate([Bx,By,Bz]).ravel(),
                #weights = np.concatenate([Bxerr,Byerr,Bzerr]).ravel(),
                x=XX, y=YY, z=ZZ, params = self.params, method='leastsq',fit_kws={'maxfev':1000})
        else:
            self.result = self.mod.fit(np.concatenate([Bx,By,Bz]).ravel(),
                #weights = np.concatenate([Bxerr,Byerr,Bzerr]).ravel(),
                #x=XX, y=YY, z=ZZ, params = self.params, method='leastsq',fit_kws={'maxfev':1000})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='differential_evolution',fit_kws={'maxfun':1})
                x=XX, y=YY, z=ZZ, params = self.params, method='leastsq')

        self.params = self.result.params
        end_time=time()
        if not recreate:
            print("Elapsed time was %g seconds" % (end_time - start_time))
            report_fit(self.result, show_correl=False)
        if not self.no_save and not recreate: self.pickle_results(pickle_name)

    def fit_full(self,ns=7, ms=40, cns=7, cms=7, use_pickle = False, pickle_name = 'default', line_profile=False, recreate=False):
        Reff=11000
        a = 3e4
        b = 3e4
        c = 3e4
        Bz = []
        Br =[]
        Bphi = []
        Bzerr = []
        Brerr =[]
        Bphierr = []
        RR =[]
        ZZ = []
        PPs = []
        PPe = []
        for phi in self.phi_steps:
            if phi==0: nphi = np.pi
            else: nphi=phi-np.pi

            input_data_phi = self.input_data[(np.abs(self.input_data.Phi-phi)<1e-6)|(np.abs(self.input_data.Phi-nphi)<1e-6)]
            input_data_phi.ix[np.abs(input_data_phi.Phi-nphi)<1e-6, 'R']*=-1

            piv_bz = input_data_phi.pivot('Z','R','Bz')
            piv_br = input_data_phi.pivot('Z','R','Br')
            piv_bphi = input_data_phi.pivot('Z','R','Bphi')
            piv_bz_err = input_data_phi.pivot('Z','R','Bzerr')
            piv_br_err = input_data_phi.pivot('Z','R','Brerr')
            piv_bphi_err = input_data_phi.pivot('Z','R','Bphierr')

            R = piv_br.columns.values
            Z = piv_br.index.values
            Bz.append(piv_bz.values)
            Br.append(piv_br.values)
            Bphi.append(piv_bphi.values)
            Bzerr.append(piv_bz_err.values)
            Brerr.append(piv_br_err.values)
            Bphierr.append(piv_bphi_err.values)
            RR_slice,ZZ_slice = np.meshgrid(R, Z)
            RR.append(RR_slice)
            ZZ.append(ZZ_slice)
            if phi>np.pi/2:
                PP_slice_s = np.full_like(RR_slice,input_data_phi.Phi.unique()[1])
                PP_slice_s[:,PP_slice_s.shape[1]/2:]=input_data_phi.Phi.unique()[0]
            else:
                PP_slice_s = np.full_like(RR_slice,input_data_phi.Phi.unique()[0])
                PP_slice_s[:,PP_slice_s.shape[1]/2:]=input_data_phi.Phi.unique()[1]
            PP_slice_e = np.full_like(RR_slice,input_data_phi.Phi.unique()[0])
            PP_slice_e[:,PP_slice_e.shape[1]/2:]=input_data_phi.Phi.unique()[1]
            PPs.append(PP_slice_s)
            PPe.append(PP_slice_e)

        ZZ = np.concatenate(ZZ)
        RR = np.concatenate(RR)
        PPs = np.concatenate(PPs)
        PPe = np.concatenate(PPe)
        Bz = np.concatenate(Bz)
        Br = np.concatenate(Br)
        Bphi = np.concatenate(Bphi)
        Bzerr = np.concatenate(Bzerr)
        Brerr = np.concatenate(Brerr)
        Bphierr = np.concatenate(Bphierr)
        if line_profile:
            return ZZ,RR,PPe,Bz,Br,Bphi

        print
        b_full_3d_fast = ff.b_full_3d_producer(a,b,c,Reff,ZZ,RR,PPs,ns,ms,cns,cms)
        self.mod = Model(b_full_3d_fast, independent_vars=['r','z','phi'])

        if use_pickle or recreate:
            if isinstance(pickle_name, collections.Sequence) and type(pickle_name)!=str:
                self.params = Parameters()
                for name in pickle_name:
                    self.params += pkl.load(open(name+'_results.p',"rb"))
            else:
                self.params = pkl.load(open(pickle_name+'_results.p',"rb"))

        else:
            self.params = Parameters()

        #if 'R' not in    self.params: self.params.add('R',value=Reff,vary=False)
        #if 'ns' not in self.params: self.params.add('ns',value=ns,vary=False)
        #else: self.params['ns'].value=ns
        #if 'ms' not in self.params: self.params.add('ms',value=ms,vary=False)
        #else: self.params['ms'].value=ms
        #if 'delta1' not in self.params: self.params.add('delta1',value=0.0,min=0,max=np.pi,vary=False)
        #else: self.params['delta1'].vary=False

        #for n in range(ns):
        #    for m in range(ms):
        #        if 'A_{0}_{1}'.format(n,m) not in self.params: self.params.add('A_{0}_{1}'.format(n,m),value=0)
        #        else: self.params['A_{0}_{1}'.format(n,m)].vary=True
        #        if 'B_{0}_{1}'.format(n,m) not in self.params: self.params.add('B_{0}_{1}'.format(n,m),value=0)
        #        else: self.params['B_{0}_{1}'.format(n,m)].vary=True

        if not recreate: print 'fitting with n={0}, m={1}, cns={2}, cms={3}'.format(ns,ms,cns,cms)
        start_time=time()
        if recreate:
            for param in self.params:
                self.params[param].vary=False
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([Brerr,Bzerr,Bphierr]).ravel(),
                r=RR, z=ZZ, phi=PPs, params = self.params, method='leastsq',fit_kws={'maxfev':1})
        elif use_pickle:
            for param in self.params:
                self.params[param].vary=False
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([Brerr,Bzerr,Bphierr]).ravel(),
                r=RR, z=ZZ, phi=PPs, params = self.params, method='leastsq',fit_kws={'maxfev':1000})
        else:
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([Brerr,Bzerr,Bphierr]).ravel(),
                r=RR, z=ZZ, phi=PPs, params = self.params, method='leastsq',fit_kws={'maxfev':1000})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='differential_evolution',fit_kws={'maxfun':1})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq')

        self.params = self.result.params
        end_time=time()
        if not recreate:
            print("Elapsed time was %g seconds" % (end_time - start_time))
            report_fit(self.result, show_correl=False)
        if not self.no_save and not recreate: self.pickle_results('full')


    def pickle_results(self,pickle_name='default'):
        pkl.dump( self.result.params, open( pickle_name+'_results.p', "wb" ),pkl.HIGHEST_PROTOCOL )
