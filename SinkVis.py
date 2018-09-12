#!/usr/bin/env python
"""
Usage:
SinkVis.py <files> ... [options]

Options:
    -h --help           Show this screen.
    --rmax=<pc>         Maximum radius of plot window; defaults to box size/10.
    --c=<cx,cy,cz>      Coordinates of plot window center relative to box center [default: 0.0,0.0,0.0]
    --limits=<min,max>  Dynamic range of surface density colormap [default: 10,1e5]
    --cmap=<name>       Name of colormap to use [default: viridis]
    --interp_fac=<N>    Number of interpolating frames per snapshot [default: 1]
    --np=<N>            Number of processors to run on [default: 1]
    --res=<N>           Image resolution [default: 400]
"""

import meshoid
from meshoid import GridSurfaceDensity
import h5py
from sys import argv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, rgb_to_hsv, hsv_to_rgb
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from joblib import Parallel, delayed
import aggdraw
from natsort import natsorted
from docopt import docopt
from glob import glob

arguments = docopt(__doc__)
filenames = natsorted(arguments["<files>"])
boxsize = h5py.File(filenames[0])["Header"].attrs["BoxSize"]

r = float(arguments["--rmax"]) if arguments["--rmax"] else boxsize/10
center = np.array([float(c) for c in arguments["--c"].split(',')])
limits = np.array([float(c) for c in arguments["--limits"].split(',')])
res = int(arguments["--res"])
nproc = int(arguments["--np"])
n_interp = int(arguments["--interp_fac"])
cmap = arguments["--cmap"]

L = r*2

i = 0

font = ImageFont.truetype("LiberationSans-Regular.ttf", res//12)

def TransformCoords(x, angle):
    return np.c_[x[:,0]*np.cos(angle) + x[:,1]*np.sin(angle), -x[:,0]*np.sin(angle) + x[:,1]*np.cos(angle), x[:,2]]

def MakeImage(i, rot=0):
    F1 = h5py.File(filenames[i])
    F2 = h5py.File(filenames[min(i+1,len(filenames)-1)])
    id1, id2 = np.array(F1["PartType0"]["ParticleIDs"]), np.array(F2["PartType0"]["ParticleIDs"])
    unique, counts = np.unique(id2, return_counts=True)
    doubles = unique[counts>1]
    id2[np.in1d(id2,doubles)]=-1

    t1, t2 = F1["Header"].attrs["Time"], F2["Header"].attrs["Time"]
    x1, x2 = np.array(F1["PartType0"]["Coordinates"])[id1.argsort()], np.array(F2["PartType0"]["Coordinates"])[id2.argsort()]
    x1 -= boxsize/2 + center
    x2 -= boxsize/2 + center
    u1, u2 = np.array(F1["PartType0"]["InternalEnergy"])[id1.argsort()], np.array(F2["PartType0"]["InternalEnergy"])[id2.argsort()]
    h1, h2 = np.array(F1["PartType0"]["SmoothingLength"])[id1.argsort()], np.array(F2["PartType0"]["SmoothingLength"])[id2.argsort()]
    m1, m2 = np.array(F1["PartType0"]["Masses"])[id1.argsort()], np.array(F2["PartType0"]["Masses"])[id2.argsort()]
    
    # take only the particles that are in both snaps
    
    common_ids = np.intersect1d(id1,id2)
    idx1 = np.in1d(np.sort(id1),common_ids)
    idx2 = np.in1d(np.sort(id2),common_ids)


    x1 = x1[idx1]
    u1 = u1[idx1]
    h1 = h1[idx1]
    m1 = m1[idx1]
    x2 = x2[idx2]
    u2 = u2[idx2]
    h2 = h2[idx2]
    m2 = m2[idx2]
    m = np.array(F2["PartType0"]["Masses"])[idx2]
    
    if "PartType5" in F1.keys():
        id1s, id2s = np.array(F1["PartType5"]["ParticleIDs"]), np.array(F2["PartType5"]["ParticleIDs"])
        unique, counts = np.unique(id2s, return_counts=True)
        doubles = unique[counts>1]
        id2s[np.in1d(id2s,doubles)]=-1

        x1s, x2s = np.array(F1["PartType5"]["Coordinates"])[id1s.argsort()], np.array(F2["PartType5"]["Coordinates"])[id2s.argsort()]
        #m1s, m2s = (np.array(F1["PartType5"]["Masses"])*np.array(F1["PartType5"]["OStarNumber"]))[id1s.argsort()], (np.array(F2["PartType5"]["Masses"])*np.array(F2["PartType5"]["OStarNumber"]))[id2s.argsort()]
        m1s, m2s = np.array(F1["PartType5"]["Masses"]), np.array(F2["PartType5"]["Masses"])
        # take only the particles that are in both snaps

        common_ids = np.intersect1d(id1s,id2s)
        idx1 = np.in1d(np.sort(id1s),common_ids)
        idx2 = np.in1d(np.sort(id2s),common_ids)

        x1s = x1s[idx1]
        m1s = m1s[idx1]
        x2s = x2s[idx2]
        m2s = m2s[idx2]
        m_star = m2s

    time = F1["Header"].attrs["Time"]
    for k in range(n_interp):
        x = float(k)/n_interp * x2 + (n_interp-float(k))/n_interp * x1
        
        logu = float(k)/n_interp * np.log10(u2) + (n_interp-float(k))/n_interp * np.log10(u1)
        u = 10**logu

        h = float(k)/n_interp * h2 + (n_interp-float(k))/n_interp * h1
        rho = 32*m1/(4*np.pi*h**3/3)
        ntot = len(m)
        sigma_gas = GridSurfaceDensity(m, x, h, res, L).T
        if "PartType5" in F1.keys():
            x_star = float(k)/n_interp * x2s + (n_interp-float(k))/n_interp * x1s
        fgas = (np.log10(sigma_gas)-np.log10(limits[0]))/np.log10(limits[1]/limits[0])
        fgas = np.clip(fgas,0,1)
        data = fgas[:,:,np.newaxis]*plt.get_cmap(cmap)(fgas)[:,:,:3] 
        data = np.clip(data,0,1)

        filename = "SurfaceDensity_%s.%s.png"%(str(i).zfill(3),k)
        plt.imsave(filename, data) #f.split("snapshot_")[1].split(".hdf5")[0], map)
        print(filename)
        F = Image.open(filename)
        draw = ImageDraw.Draw(F)
        gridres=res
        draw.line(((gridres/16, 7*gridres/8), (gridres*5/16, 7*gridres/8)), fill="#FFFFFF", width=6)
        draw.text((gridres/16, 7*gridres/8 + 5), "%gpc"%(r*500/1000), font=font)
        draw.text((gridres/16, gridres/24), "%3.2gMyr"%(time*979), font=font)
        if "PartType5" in F1.keys():
            d = aggdraw.Draw(F)
            pen = aggdraw.Pen("white",gridres/800)
            p = aggdraw.Brush((255, 0,0))
#            print(len(x_star))
            #for X in x_star:
#                coords = np.concatenate([(X[:2]+r)/(2*r)*gridres-gridres/800, (X[:2]+r)/(2*r)*gridres+gridres/800])
#                d.ellipse(coords, pen, p)#, fill=(155, 176, 255))
            p = aggdraw.Brush((155, 176, 255))
            for X in x_star[m_star>0]:
                X -= boxsize/2
                coords = np.concatenate([(X[:2]+r)/(2*r)*gridres-gridres/400, (X[:2]+r)/(2*r)*gridres+gridres/400])
                d.ellipse(coords, pen, p)#, fill=(155, 176, 255))
            d.flush()
        F.save(filename)
        F.close()

if nproc>1:
    Parallel(n_jobs=nproc)(delayed(MakeImage)(i) for i in range(len(filenames)))
else:
    [MakeImage(i) for i in range(len(filenames))]
