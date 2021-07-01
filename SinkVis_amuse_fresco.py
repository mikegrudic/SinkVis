#!/usr/bin/env python
#Used by SinkVis to create observation-like images using amuse-fresco (https://pypi.org/project/amuse-fresco/). See install.txt for instructions!

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from amuse.datamodel import Particles
from amuse.units import units, nbody_system
from amuse.community.sse.interface import SSE
#from amuse.ext.masc import make_a_star_cluster
from amuse.ext import masc
from amuse.ext.fresco import make_fresco_image
import h5py

def make_amuse_fresco_stars_only(x,mstar,age_yr,L,res=512,p=5e-4,mass_limits=[0,0],mass_rescale=1.,filename=None,vmax=None):
    number_of_stars = len(mstar)

    if (mass_limits[0]!=0.0) or (mass_limits[1]!=0.0):
        if (mass_limits[0]==0.0): mass_limits[0]=np.min(mstar)
        if (mass_limits[1]==0.0): mass_limits[1]=np.max(mstar)
        mstar_new = np.clip(mstar,mass_limits[0],mass_limits[1])
        #small correction to make them stand apart (e.g. instea of clipping both 100 and 50 msun to 50, we get 50 and 60  so they don't look identical)
        mstar_new = mstar_new + (mstar-mstar_new)*0.1
        ##limits masses of star
        #logm = np.log10(mstar); logm0 = np.max(logm) + np.min(logm);
        #mstar_new = 10**( (logm - logm0)/mass_limits + logm0 )
    mstar_new = mstar**mass_rescale
    new_stars = Particles(number_of_stars)
    new_stars.age = age_yr | units.yr
    new_stars.mass = mstar_new  | units.MSun
    new_stars.position = x | units.pc
    stars = new_stars
    gas = Particles()
    se = SSE()
    se.particles.add_particles(stars)
    from_se = se.particles.new_channel_to(stars)
    from_se.copy()
    image, _ = make_fresco_image( stars, gas, return_vmax=True,\
        image_width=[L | units.pc,L | units.pc], image_size=[res,res],percentile=1-p,vmax=vmax)
    #image: (2048,2048,3) RGB
    if not(filename is None):
        #Save image to file
        plt.imshow(image[::-1],extent=(-L/2.0,L/2.0,-L/2.0,L/2.0))
        plt.xlim(-L/2.0,L/2.0)
        plt.ylim(-L/2.0,L/2.0)
        plt.imsave(filename,image[::-1])
    
    return image[::-1]
