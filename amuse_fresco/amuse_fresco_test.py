#!/usr/bin/env python
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


filename="/scratch/05917/tg852163/GMC_sim/Runs/Physics_ladder/M2e4_C_M_J_RTH_2e7/output/snapshot_060.hdf5"

with h5py.File(filename,'r') as F:
    mstar = np.array(F["PartType5"]["Masses"])
    x = np.array(F["PartType5"]["Coordinates"]) - F["Header"].attrs["BoxSize"]/2

    

number_of_stars = len(mstar)
new_stars = Particles(number_of_stars)
new_stars.age = 0 | units.yr
new_stars.mass = mstar  | units.MSun
new_stars.position = x | units.pc
stars = new_stars
gas = Particles()
se = SSE()
se.particles.add_particles(stars)
from_se = se.particles.new_channel_to(stars)
from_se.copy()

for p in (1.0-np.logspace(-2,-7,6)):
    image, vmax = make_fresco_image(
        stars, gas,
        return_vmax=True,
        image_width=[20. | units.pc,20. | units.pc], image_size=[2048,2048],percentile=p,   

    )
        #mode=["stars"],
        #return_vmax=True,
    plt.imshow(image[::-1],extent=(-10,10,-10,10))
    #plt.scatter(x[:,0],x[:,1],s=1)
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.imsave("M2e4_stars_%g_log.png"%p,image[::-1])