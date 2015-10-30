#! /usr/bin/env python

import pandas as pd
import numpy as np
from tools.fit_funcs import *
from lmfit import  Model
import cPickle as pkl
from time import time

class FieldFitter:
  """Input hall probe measurements, perform semi-analytical fit, return fit function and other stuff."""
  def __init__(self, input_data, phi_steps = None):
    self.input_data = input_data
    if phi_steps: self.phi_steps = phi_steps
    else: self.phi_steps = (np.pi/2,)
    #self.add_zero_data()

  def add_zero_data(self):
    df_highz = self.input_data[self.input_data.Z==self.input_data.Z.max()]
    df_highz.Z = 1000000
    df_highz.Bz = 0
    df_highz.Bx = 0
    df_highz.By = 0
    df_highz.Br = 0
    df_lowz = df_highz.copy()
    df_lowz.Z = -1000000
    self.zero_data = pd.concat([df_highz, self.input_data, df_lowz], ignore_index=True)
    self.zero_data.sort(['Z','X'],inplace=True)

  def fit_3d_v3(self,ns=5,ms=10,use_pickle = False):
    Reff=9000
    Bz = []
    Br =[]
    Bphi = []
    RR =[]
    ZZ = []
    PP = []
    for phi in self.phi_steps:
      if phi==0: nphi = np.pi
      else: nphi=phi-np.pi

      input_data_phi = self.input_data[(np.abs(self.input_data.Phi-phi)<1e-6)|(np.abs(self.input_data.Phi-nphi)<1e-6)]
      input_data_phi.ix[np.abs(input_data_phi.Phi-nphi)<1e-6, 'R']*=-1
      print input_data_phi.Phi.unique()

      piv_bz = input_data_phi.pivot('Z','R','Bz')
      piv_br = input_data_phi.pivot('Z','R','Br')
      piv_bphi = input_data_phi.pivot('Z','R','Bphi')

      R = piv_br.columns.values
      Z = piv_br.index.values
      Bz.append(piv_bz.values)
      Br.append(piv_br.values)
      Bphi.append(piv_bphi.values)
      RR_slice,ZZ_slice = np.meshgrid(R, Z)
      RR.append(RR_slice)
      ZZ.append(ZZ_slice)
      PP_slice = np.full_like(RR_slice,input_data_phi.Phi.unique()[0])
      PP_slice[:,PP_slice.shape[1]/2:]=input_data_phi.Phi.unique()[1]
      PP.append(PP_slice)

    ZZ = np.concatenate(ZZ)
    RR = np.concatenate(RR)
    PP = np.concatenate(PP)
    Bz = np.concatenate(Bz)
    Br = np.concatenate(Br)
    Bphi = np.concatenate(Bphi)

    brzphi_3d_fast = brzphi_3d_producer(ZZ,RR,PP,Reff,ns,ms)
    self.mod = Model(brzphi_3d_fast, independent_vars=['r','z','phi'])

    if use_pickle:
      self.params = pkl.load(open('result.p',"rb"))
    else:
      self.params = Parameters()

    if 'R' not in  self.params: self.params.add('R',value=Reff,vary=False)
    if 'ns' not in self.params: self.params.add('ns',value=ns,vary=False)
    if 'ms' not in self.params: self.params.add('ms',value=ms,vary=False)

    for n in range(ns):
      if 'delta_{0}'.format(n) not in self.params: self.params.add('delta_{0}'.format(n),value=(np.pi/ns)*n, min=0, max=np.pi)
      else: self.params['delta_{0}'.format(n)].vary=False
      for m in range(ms):
        if 'A_{0}_{1}'.format(n,m) not in self.params: self.params.add('A_{0}_{1}'.format(n,m),value=-100)
        else: self.params['A_{0}_{1}'.format(n,m)].vary=True
        if 'B_{0}_{1}'.format(n,m) not in self.params: self.params.add('B_{0}_{1}'.format(n,m),value=100)
        else: self.params['B_{0}_{1}'.format(n,m)].vary=True

    print 'fitting with n={0}, m={1}'.format(ns,ms)
    start_time=time()
    if use_pickle:
      self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
          #weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          r=RR, z=ZZ, phi=PP, params = self.params,method='leastsq',fit_kws={'maxfev':500})
    else:
      self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
          #weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          r=RR, z=ZZ, phi=PP, params = self.params,method='leastsq',fit_kws={'maxfev':200})

    self.params = self.result.params
    end_time=time()
    print("Elapsed time was %g seconds" % (end_time - start_time))
    report_fit(self.result, show_correl=False)
    self.pickle_results()

  def fit_3d_v2(self,ns=5,ms=10,use_pickle = False):
    Reff=9000
    phi_slice = np.pi/2
    input_data_phi = self.input_data[((abs(self.input_data.Phi)-phi_slice)<1e-6)|(self.input_data.Phi==0)]
    #input_data_phi = self.input_data[(abs(self.input_data.Phi)-phi_slice)<1e-6]
    input_data_phi.ix[input_data_phi.Phi<0, 'R']*=-1

    piv_bz = input_data_phi.pivot('Z','R','Bz')
    piv_br = input_data_phi.pivot('Z','R','Br')
    piv_bphi = input_data_phi.pivot('Z','R','Bphi')

    self.R = piv_br.columns.values
    self.Z = piv_br.index.values
    self.Bz = piv_bz.values
    self.Br = piv_br.values
    self.Bphi = piv_bphi.values
    self.RR,self.ZZ = np.meshgrid(self.R, self.Z)
    self.PP = np.full_like(self.RR,input_data_phi.Phi.unique()[0])
    if len(input_data_phi.Phi.unique())>1:
      self.PP[:,self.PP.shape[1]/2:]*=-1
    #print self.RR
    #print self.PP
    #raw_input()

    #piv_bz_err = input_data_phi.pivot('Z','R','Bzerr')
    #piv_br_err = input_data_phi.pivot('Z','R','Brerr')
    #piv_bphi_err = input_data_phi.pivot('Z','R','Bphierr')
    #self.Bzerr=piv_bz_err.values
    #self.Brerr=piv_br_err.values
    #self.Bphierr=piv_bphi_err.values

    brzphi_3d_fast = brzphi_3d_producer(self.ZZ,self.RR,self.PP,Reff,ns,ms)
    self.mod = Model(brzphi_3d_fast, independent_vars=['r','z','phi'])

    if use_pickle:
      self.params = pkl.load(open('result.p',"rb"))
    else:
      self.params = Parameters()

    if 'R' not in  self.params: self.params.add('R',value=Reff,vary=False)
    #self.params.add('delta',value=0.5,min=0,max=np.pi, vary=False)
    #self.params.add('delta',value=0.005,min=-0.1,max=0.1, vary=True)
    if 'ns' not in self.params: self.params.add('ns',value=ns,vary=False)
    if 'ms' not in self.params: self.params.add('ms',value=ms,vary=False)

    for n in range(ns):
      if 'delta_{0}'.format(n) not in self.params: self.params.add('delta_{0}'.format(n),value=(np.pi/ns)*n, min=0, max=np.pi)
      else: self.params['delta_{0}'.format(n)].vary=False
      for m in range(ms):
        if 'A_{0}_{1}'.format(n,m) not in self.params: self.params.add('A_{0}_{1}'.format(n,m),value=-100)
        else: self.params['A_{0}_{1}'.format(n,m)].vary=True
        if 'B_{0}_{1}'.format(n,m) not in self.params: self.params.add('B_{0}_{1}'.format(n,m),value=100)
        else: self.params['B_{0}_{1}'.format(n,m)].vary=True
      #print 'fitting with n={0}, m={1}'.format(n,ms)
      #if use_pickle:
      #  self.result = self.mod.fit(np.concatenate([self.Br,self.Bz,self.Bphi]).ravel(),
      #    r=self.RR, z=self.ZZ, phi=self.PP, params = self.params,method='powell')
      #else:
      #  self.result = self.mod.fit(np.concatenate([self.Br,self.Bz,self.Bphi]).ravel(),
      #    r=self.RR, z=self.ZZ, phi=self.PP, params = self.params,method='leastsq')
      #self.params = self.result.params

    print 'fitting with n={0}, m={1}'.format(ns,ms)
    start_time=time()
    if use_pickle:
      self.result = self.mod.fit(np.concatenate([self.Br,self.Bz,self.Bphi]).ravel(),
          #weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          r=self.RR, z=self.ZZ, phi=self.PP, params = self.params,method='leastsq',fit_kws={'maxfev':500})
    else:
      self.result = self.mod.fit(np.concatenate([self.Br,self.Bz,self.Bphi]).ravel(),
          #weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          r=self.RR, z=self.ZZ, phi=self.PP, params = self.params,method='leastsq',fit_kws={'maxfev':500})

    self.params = self.result.params
    end_time=time()
    print("Elapsed time was %g seconds" % (end_time - start_time))
    report_fit(self.result, show_correl=False)
    self.pickle_results()

  def fit_3d(self,ns=5,ms=10,use_pickle = False):

    Reff=9000
    phi_slice = 1.570796
    #input_data_phi = self.input_data[(abs(self.input_data.phi-phi_slice)<1e-6)|(self.input_data.phi==0)]
    input_data_phi = self.input_data[(abs(abs(self.input_data.Phi)-phi_slice)<1e-6)]
    piv_bz_pos = input_data_phi.query('Phi>=0').pivot('Z','R','Bz')
    piv_br_pos = input_data_phi.query('Phi>=0').pivot('Z','R','Br')
    piv_bphi_pos = input_data_phi.query('Phi>=0').pivot('Z','R','Bphi')

    piv_bz_neg = input_data_phi.query('Phi<0').pivot('Z','R','Bz')
    piv_br_neg = input_data_phi.query('Phi<0').pivot('Z','R','Br')
    piv_bphi_neg = input_data_phi.query('Phi<0').pivot('Z','R','Bphi')

    R=piv_br.columns.values
    Z=piv_br.index.values
    self.Bz=piv_bz.values
    self.Br=piv_br.values
    self.Bphi=piv_bphi.values
    self.R,self.Z = np.meshgrid(R, Z)
    self.Phi = np.full_like(self.R,input_data_phi.Phi.unique()[0])

    piv_bz_err = input_data_phi.pivot('Z','R','Bzerr')
    piv_br_err = input_data_phi.pivot('Z','R','Brerr')
    piv_bphi_err = input_data_phi.pivot('Z','R','Bphierr')
    self.Bzerr=piv_bz_err.values
    self.Brerr=piv_br_err.values
    self.Bphierr=piv_bphi_err.values

    #self.mod = Model(brzphi_3d, independent_vars=['r','z','phi'])
    brzphi_3d_fast = brzphi_3d_producer(self.Z,self.R,self.Phi,Reff,ns,ms)
    self.mod = Model(brzphi_3d_fast, independent_vars=['r','z','phi'])

    if use_pickle:
      #self.params = pkl.load(open('result.p',"rb"))
      #for param in self.params:
      #  self.params[param].vary = False
      #self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(), weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
      #    r=self.X,z=self.Y, params = self.params,method='leastsq')
      pass
    else:

      #b_zeros = []
      #for n in range(ns):
      #  b_zeros.append(special.jn_zeros(n,ms))
      #kms = [b/R for b in b_zeros]

      self.params = Parameters()
      self.params.add('R',value=Reff,vary=False)
      self.params.add('offset',value=0,vary=False)
      self.params.add('ns',value=ns,vary=False)
      self.params.add('ms',value=ms,vary=False)

      for n in range(ns):
        for m in range(ms):
          self.params.add('A_{0}_{1}'.format(n,m),value=0)
          self.params.add('B_{0}_{1}'.format(n,m),value=0)
      print 'fitting with n={0}, m={1}'.format(ns,ms)
      self.result = self.mod.fit(np.concatenate([self.Br,self.Bz,self.Bphi]).ravel(),
          weights = np.concatenate([self.Brerr,self.Bzerr,self.Bphierr]).ravel(),
          r=self.R, z=self.Z, phi=self.Phi, params = self.params,method='leastsq')

    self.params = self.result.params
    report_fit(self.result)

  def fit_2d_sim(self,B,C,nparams = 20,use_pickle = False):

    if B=='X':Br='Bx'
    elif B=='Y':Br='By'
    piv_bz = self.input_data.pivot(C,B,'Bz')
    piv_br = self.input_data.pivot(C,B,Br)
    X=piv_br.columns.values
    Y=piv_br.index.values
    self.Bz=piv_bz.values
    self.Br=abs(piv_br.values)
    self.X,self.Y = np.meshgrid(X, Y)

    piv_bz_err = self.input_data.pivot(C,B,'Bzerr')
    piv_br_err = self.input_data.pivot(C,B,Br+'err')
    self.Bzerr=piv_bz_err.values
    self.Brerr=piv_br_err.values

    self.mod = Model(brz_2d_trig, independent_vars=['r','z'])

    if use_pickle:
      self.params = pkl.load(open('result.p',"rb"))
      #for param in self.params:
      #  self.params[param].vary = False
      self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(), weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          r=self.X,z=self.Y, params = self.params,method='leastsq')
    else:
      self.params = Parameters()
      #self.params.add('R',value=1000,vary=False)
      #self.params.add('R',value=22000,vary=False)
      self.params.add('R',value=9000,vary=False)
      #self.params.add('offset',value=-14000,vary=False)
      self.params.add('offset',value=0,vary=False)
      #self.params.add('C',value=0)
      self.params.add('A0',value=0)
      self.params.add('B0',value=0)
      #self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq')

      for i in range(nparams):
        print 'refitting with params:',i+1
        self.params.add('A'+str(i+1),value=0)
        self.params.add('B'+str(i+1),value=0)
        #if (i+1)%10==0:
        #  self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq')
        #  self.params = self.result.params

      #    fit_kws={'xtol':1e-100,'ftol':1e-100,'maxfev':5000,'epsfcn':1e-40})
      self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(), weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          r=self.X,z=self.Y, params = self.params,method='leastsq')
      #self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(), weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          #r=self.X,z=self.Y, params = self.params,method='lbfgsb',fit_kws= {'options':{'factr':0.1}})

    self.params = self.result.params
    report_fit(self.result)


  def fit_2d(self,A,B,C,use_pickle = False):
    self.mag_field = A
    self.axis2 = B
    self.axis1 = C

    piv = self.input_data.pivot(C,B,A)
    X=piv.columns.values
    Y=piv.index.values
    self.Z=piv.values
    self.X,self.Y = np.meshgrid(X, Y)

    piv_err = self.input_data.pivot(C,B,A+'err')
    self.Zerr = piv_err.values
    if A == 'Bz':
      #self.mod = Model(bz_2d, independent_vars=['r','z'])
      self.mod = Model(bz_2d, independent_vars=['r','z'])
    elif A == 'Br':
      self.mod = Model(br_2d, independent_vars=['r','z'])
    else:
      raise KeyError('No function available for '+A)

    if use_pickle:
      self.params = pkl.load(open('result.p',"rb"))
      self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params, weights  = self.Zerr.ravel(), method='leastsq')
      #for param in self.params:
      #  self.params[param].vary = False
      self.params = self.result.params
    else:
      self.params = Parameters()
      #self.params.add('R',value=1000,vary=False)
      #self.params.add('R',value=22000,vary=False)
      self.params.add('R',value=11000,vary=False)
      #if A == 'Br':
      self.params.add('C',value=0)
      self.params.add('A0',value=0)
      self.params.add('B0',value=0)
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq')

      for i in range(60):
        print 'refitting with params:',i+1
        #self.params = self.result.params
        self.params.add('A'+str(i+1),value=0)
        self.params.add('B'+str(i+1),value=0)
        #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='nelder')

      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq',
      #    fit_kws={'xtol':1e-100,'ftol':1e-100,'maxfev':5000,'epsfcn':1e-40})
      self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params, weights  = self.Zerr.ravel(), method='leastsq')
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='powell')
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.result.params,method='lbfgsb')
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.result.params,method='lbfgsb')
      self.params = self.result.params
      #self.params['R'].vary=True
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='slsqp')

    report_fit(self.result)

  def fit_1d(self,A,B):

    if A == 'Bz':
      self.mod = Model(bz_2d)
    elif A == 'Br':
      self.mod = Model(br_2d, independent_vars=['r','z'])
    else:
      raise KeyError('No function available for '+A)


    data_1d = self.input_data.query('X==0 & Y==0')
    self.X=data_1d[B].values
    self.Z=data_1d[A].values
    self.mod = Model(bz_r0_1d)

    self.params = Parameters()
    self.params.add('R',value=100000,vary=False)
    self.params.add('A0',value=0)
    self.params.add('B0',value=0)

    self.result = self.mod.fit(self.Z,z=self.X, params = self.params,method='nelder')
    for i in range(9):
      print 'refitting with params:',i+1
      self.params = self.result.params
      self.params.add('A'+str(i+1),value=0)
      self.params.add('B'+str(i+1),value=0)
      self.result = self.mod.fit(self.Z,z=self.X, params = self.params,method='nelder')

    self.params = self.result.params
    report_fit(self.result)
    #report_fit(self.params)

  def pickle_results(self):
    pkl.dump( self.result.params, open( 'result.p', "wb" ),pkl.HIGHEST_PROTOCOL )

