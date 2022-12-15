from __future__ import division
import numpy as np
import matplotlib 
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec 
from matplotlib.colorbar import ColorbarBase 
from matplotlib import gridspec
from scipy.stats import pearsonr
from scipy import stats
from pirates import *
from astropy.cosmology import WMAP9 as cosmo #to calculate hubble parameter
import gc, os, time
import emcee
import corner
start = time.time()

#--------------------------------------------------------------------------------------------------------------
G = 4.302*10**(-6)

#my models to fit the observations:
#my models to fit the observations:
#Burkert halo
def Burkert_halo(r, r0, rho0):
	G = 4.302*10**(-6) # units (kpc/Msun)*(km2/s2)
	M0 = (1.6*rho0*(r0**3))*(0.014775*10**33) #units Msun
	Mbh = (4*M0)*( (np.log(1.+(r/r0))) - (np.arctan(r/r0)) + (0.5*np.log(1.+((r/r0)**2))) )
	Vbh = ((G*Mbh)/r)**0.5
	return Mbh, Vbh
	
#NFW Halo	
def NFW_halo(r, r0, rho0):
	Mdm = 4*np.pi*(rho0*(0.014775*10**33))*(r0**3)*((np.log(1+(r/r0)) - ((r/r0)/(1+(r/r0)))))
	Vdm = (G*Mdm/r)**0.5
	return Mdm, Vdm
	
def disk_velocity(Md, rd, r):
	G = 4.302*10**(-6) # units (kpc/Msun)*(km2/s2)
	x = r/(3.2*rd)
	# ~ x = r/(6.4*Rd)
	#bassel func~~~~~
	# ~ #first 
	v0 = 0  #order of bessel func
	K0 = kv(v0, 1.6*x) #BESSEL Function
	I0 = iv(v0, 1.6*x) #BESSEL Function
	#~~~second 
	v1 = 1  #order of bessel func
	K1 = kv(v1, 1.6*x) #BESSEL Function
	I1 = iv(v1, 1.6*x) #BESSEL Function

	c0 = (I0*K0)-(I1*K1)
	Vd = ((0.5)*(G*Md/rd)*((3.2*x)**2)*c0)**0.5
	return Vd

def H2_velocity(MH2, rH2, r):
	G = 4.302*10**(-6) # units (kpc/Msun)*(km2/s2)
	x = r/(3.2*rH2)
	# ~ x = r/(6.4*Rd)
	#bassel func~~~~~
	# ~ #first 
	v0 = 0  #order of bessel func
	K0 = kv(v0, 1.6*x) #BESSEL Function
	I0 = iv(v0, 1.6*x) #BESSEL Function
	#~~~second 
	v1 = 1  #order of bessel func
	K1 = kv(v1, 1.6*x) #BESSEL Function
	I1 = iv(v1, 1.6*x) #BESSEL Function

	c0 = (I0*K0)-(I1*K1)
	VH2 = ((0.5)*(G*MH2/rH2)*(3.2*(x)**2)*c0)**0.5
	return VH2
	
	
def HI_velocity(MHI, RHI, r):
	G = 4.302*10**(-6) # units (kpc/Msun)*(km2/s2)
	x = r/RHI
	#bassel func~~~~~
	#first 
	v0 = 0  #order of bessel func
	K0 = kv(v0, 0.53*x) #BESSEL Function
	I0 = iv(v0, 0.53*x) #BESSEL Function
	#~~~second 
	v1 = 1  #order of bessel func
	K1 = kv(v1, 0.53*x) #BESSEL Function
	I1 = iv(v1, 0.53*x) #BESSEL Function

	c0 = (I0*K0)-(I1*K1)
	VHI = ((0.5)*(G*MHI/(RHI))*((x**2))*c0)**0.5
	return VHI


def bulge_velocity(Mb, R):
	G = 4.302*10**(-6) # units (kpc/Msun)*(km2/s2)
	Vb = np.sqrt(G*Mb/R)
	return Vb

def find_nearest(array, value):
	'''
	for any given value, find the nearest value from an array
	'''
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx],idx
	
	
def Burkert_halo_parameters(rc, rho0):
	xref = np.linspace(0.01,1000, 5000)
	mcrit = 200*136.71*(4*np.pi*(xref**3))/3  #critical mass for each galaxy
	Mhalo , Vhalo = Burkert_halo(xref, 10**rc, 10**rho0) #halo mass upto large R
	rhocrit = 200*136.71
	rhohalo = Mhalo/((4*np.pi*(xref**3))/3)
	temp, index = find_nearest(rhohalo, rhocrit)
	Rvir = xref[index] #in kpc
	Mvir = Mhalo[index] #in Msun
	Vvir = np.sqrt(G*Mvir/Rvir) #in km/sec
	return np.log10(Rvir), np.log10(Mvir), np.log10(Vvir) 
	
def NFW_halo_parameters(rc, rho0):
	xref = np.linspace(0.01,1000, 5000)
	mcrit = 200*136.71*(4*np.pi*(xref**3))/3  #critical mass for each galaxy
	Mhalo , Vhalo = NFW_halo(xref, 10**rc, 10**rho0) #halo mass upto large R
	rhocrit = 200*136.71
	rhohalo = Mhalo/((4*np.pi*(xref**3))/3)
	temp, index = find_nearest(rhohalo, rhocrit)
	Rvir = xref[index] #in kpc
	Mvir = Mhalo[index] #in Msun
	Vvir = np.sqrt(G*Mvir/Rvir) #in km/sec
	return np.log10(Rvir), np.log10(Mvir), np.log10(Vvir) 	
	

