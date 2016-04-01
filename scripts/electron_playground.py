#! /usr/bin/env python

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
from time import time

plt.close('all')

#generate a uniform cube grid, and a uniform mag field in the z direction
x = y = np.linspace(-25,25,51)
z = np.linspace(0,50,51)
bx=by=np.zeros(len(x))
bz = np.full(z.shape,3.0)
#bx=bz=np.zeros(len(x))
#by = np.full(z.shape,3.0)
xx,yy,zz = np.meshgrid(x,y,z)
bxx,byy,bzz = np.meshgrid(bx,by,bz)

#load the field into a dataframe
df = pd.DataFrame(np.array([xx,yy,zz,bxx,byy,bzz]).reshape(6,-1).T,columns = ['X','Y','Z','Bx','By','Bz'])

#reduce the number of datapoints for appropriate quiver plotting:
df_quiver = df.query('(X+5)%10==0 and (Y+5)%10==0 and Z%10==0')
#recreate 3d meshgrid by reshaping the df back into six 3d arrays
quiver_size = int(round(df_quiver.shape[0]**(1./3.)))
print 'quiver_size', quiver_size
qxx,qyy,qzz,qbxx,qbyy,qbzz = df_quiver.values.T.reshape(6,quiver_size,quiver_size,quiver_size)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X (length)')
ax.set_ylabel('Y (length)')
ax.set_zlabel('Z (length)')
ax.quiver(qxx,qyy,qzz, qbxx,qbyy,qbzz, length=3,linewidths=(2,),arrow_length_ratio=0.6,colors='r')

#now lets assume we have an electron v = 1 unit/s in z dir
#it starts at 0,0,0
#only affected by F=qvXb
#x = x0 + vt + 1/2at^2
#vf = vi+at

#natural units conversion:
# B: 1 MeV^2 = 1.4440271e9 T
# L: 1/MeV = 1.9732705e-7 m
# s: 1/MeV = 6.582122e-22 s

def gamma(v):
    return 1/np.sqrt(1-np.dot(v,v))
def calc_lorentz_accel(v_vec,b_vec):
    return -1*np.cross(v_vec,b_vec/1.4440271e-3)/(gamma(v_vec)*511e3)
    #return -1*np.cross(v_vec,b_vec/1.4440271e-3)/511e3
def add_vel(u,v):
    return 1/(1+np.dot(u,v))*(u+v/gamma(u)+(gamma(u)/(1+gamma(u)))*(np.dot(u,v)*u))

def update_kinematics(p_vec,v_vec,b_vec,dt):
#not sure how to approach this in incremental steps
    a_vec = calc_lorentz_accel(v_vec,b_vec)
    p_vec_new = p_vec+(v_vec*dt+0.5*a_vec*dt**2)*1.9732705e-4
    v_vec_new = add_vel(v_vec,a_vec*dt)
    return (p_vec_new,v_vec_new)

pos = np.array([10,-10,25])
init_pos = pos
mom = np.array([0,8e6,0]) #in eV
init_mom = mom
v = mom/(511e3*np.sqrt(1+np.dot(mom,mom)/511e3**2))
init_v = v
path = [pos]
dt = 5e-1
total_time = 0
start_time=time()
while (x[0]<=pos[0]<=x[-1] and y[0]<=pos[1]<=y[-1] and z[0]<=pos[2]<=z[-1] and total_time<dt*1000000):
    #pos,v = update_kinematics(pos,v,np.array(mag_field_function(pos[0],pos[1],pos[2],True)),dt)
    pos,v = update_kinematics(pos,v,np.array([0,0,3]),dt)
    #print pos
    path.append(pos)
    total_time+=dt
print total_time
end_time=time()
#if not cfg_pickle.recreate:
print("Elapsed time was %g seconds" % (end_time - start_time))


#ax.plot(path_z,path_x,zs=path_y,linewidth=2)
path = np.asarray(path)
ax.plot(path[:,0],path[:,1],zs=path[:,2],linewidth=2)
ax.set_title('Path of electron through magnetic field')


# these are matplotlib.patch.Patch properties
textstr = 'init pos={0}\ninit mom={1} (eV)\nB={2}'.format(init_pos, init_mom, 'ideal DS field map')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
print 'init_E', gamma(init_v)*0.511, 'MeV'
print 'final_E', gamma(v)*0.511, 'MeV'
print 'init_v', init_v, 'c'
print 'final_v', v, 'c'
print 'energy diff', gamma(v)*0.511 - gamma(init_v)*0.511, 'MeV'

# place a text box in upper left in axes coords
ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
plt.show()
plt.savefig('../plots/anim/electron_path_toy.pdf')
