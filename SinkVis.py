#!/usr/bin/env python
"""
Usage:
SinkVis.py <files> ... [options]

Options:
    -h --help            Show this screen.
    --rmax=<pc>          Maximum radius of plot window; defaults to box size/10.
    --c=<cx,cy,cz>       Coordinates of plot window center relative to box center [default: 0.0,0.0,0.0]
    --limits=<min,max>   Dynamic range of surface density colormap [default: 10,1e4]
    --Tlimits=<min,max>  Dynamic range of temperature colormap in K [default: 10,1000]
    --Tcmap=<name>       Name of colormap to use for temperature [default: inferno]
    --cmap=<name>        Name of colormap to use [default: viridis]
    --interp_fac=<N>     Number of interpolating frames per snapshot [default: 1]
    --np=<N>             Number of processors to run on [default: 1]
    --res=<N>            Image resolution [default: 500]
    --only_movie         Only the movie is saved, the images are removed at the end
    --no_movie           Does not create a movie, only makes images
    --fps=<fps>          Frame per second for movie [default: 20]
    --movie_name=<name>  Filename of the output movie file without format [default: sink_movie]
    --sink_type=<N>      Particle type of sinks [default: 5]
    --sink_scale=<msun>  Sink particle mass such that apparent sink size is 1 pixel [default: 0.1]
    --center_on_star     Center image on the most massive sink particle
    --plot_T_map         Plots both surface density and average temperature maps
    --outputfolder       Specifies the folder to save the images and movies to
"""

#Example
# python SinkVis.py /panfs/ds08/hopkins/guszejnov/GMC_sim/Tests/200msun/MHD_isoT_2e6/output/snapshot*.hdf5 --np=24 --only_movie --movie_name=200msun_MHD_isoT_2e6

#import meshoid
from meshoid import GridSurfaceDensity, GridAverage
import h5py
from sys import argv
#import matplotlib
from matplotlib import pyplot as plt
#from matplotlib.colors import LogNorm, rgb_to_hsv, hsv_to_rgb
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from joblib import Parallel, delayed
import aggdraw
from natsort import natsorted
from docopt import docopt
from glob import glob
import os

arguments = docopt(__doc__)
filenames = natsorted(arguments["<files>"])
boxsize = h5py.File(filenames[0], 'r')["Header"].attrs["BoxSize"]

r = float(arguments["--rmax"]) if arguments["--rmax"] else boxsize/10
center = np.array([float(c) for c in arguments["--c"].split(',')])
limits = np.array([float(c) for c in arguments["--limits"].split(',')])
Tlimits = np.array([float(c) for c in arguments["--Tlimits"].split(',')])
res = int(arguments["--res"])
nproc = int(arguments["--np"])
n_interp = int(arguments["--interp_fac"])
cmap = arguments["--cmap"]
Tcmap = arguments["--Tcmap"]
only_movie = arguments["--only_movie"]
no_movie = arguments["--no_movie"]
plot_T_map = arguments["--plot_T_map"]
fps = float(arguments["--fps"])
movie_name = arguments["--movie_name"]
outputfolder = arguments["--outputfolder"]
sink_type = "PartType" + arguments["--sink_type"]
sink_scale = float(arguments["--sink_scale"])
center_on_star = arguments["--center_on_star"]
L = r*2

#i = 0

font = ImageFont.truetype("LiberationSans-Regular.ttf", res//12)

image_paths = []

file_numbers = [int(f.split("snapshot_")[1].split(".hdf5")[0]) for f in filenames]
#print(file_numbers)
filedict = dict(zip(file_numbers, filenames))

def TransformCoords(x, angle):
    return np.c_[x[:,0]*np.cos(angle) + x[:,1]*np.sin(angle), -x[:,0]*np.sin(angle) + x[:,1]*np.cos(angle), x[:,2]]

def StarColor(mass_in_msun):
    star_colors = np.array([[255, 203, 132],[255, 243, 233],[155, 176, 255]])
    colors = np.int_([np.interp(np.log10(mass_in_msun),[-1,0,1],star_colors[:,i]) for i in range(3)])
    return (colors[0],colors[1],colors[2])# if len(colors)==1 else colors)

def MakeImage(i):
#    print(i)
    F1 = h5py.File(filenames[i],'r')
    F2 = h5py.File(filenames[min(i+1,len(filenames)-1)],'r')

    t1, t2 = F1["Header"].attrs["Time"], F2["Header"].attrs["Time"]

    if not sink_type in F1.keys() and center_on_star: return
    
    if "PartType0" in F1.keys():
        id1, id2 = np.array(F1["PartType0"]["ParticleIDs"]), np.array(F2["PartType0"]["ParticleIDs"])
        unique, counts = np.unique(id2, return_counts=True)
        doubles = unique[counts>1]
        id2[np.in1d(id2,doubles)]=-1
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
        m = m2
#        print(len(idx2))
#        m = np.array(F2["PartType0"]["Masses"])[id[idx2]
     
    if sink_type in F1.keys():
        id1s, id2s = np.array(F1[sink_type]["ParticleIDs"]), np.array(F2[sink_type]["ParticleIDs"])
        unique, counts = np.unique(id2s, return_counts=True)
        doubles = unique[counts>1]
        id2s[np.in1d(id2s,doubles)]=-1

        x1s, x2s = np.array(F1[sink_type]["Coordinates"])[id1s.argsort()], np.array(F2[sink_type]["Coordinates"])[id2s.argsort()]
        #m1s, m2s = (np.array(F1[sink_type]["Masses"])*np.array(F1[sink_type]["OStarNumber"]))[id1s.argsort()], (np.array(F2[sink_type]["Masses"])*np.array(F2[sink_type]["OStarNumber"]))[id2s.argsort()]
        m1s, m2s = np.array(F1[sink_type]["Masses"])[id1s.argsort()], np.array(F2[sink_type]["Masses"])[id2s.argsort()]
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
        if sink_type in F1.keys():
            x_star = float(k)/n_interp * x2s + (n_interp-float(k))/n_interp * x1s
        star_center =  (x_star[m_star.argmax()]-boxsize/2 if (center_on_star and sink_type in F1.keys()) else np.zeros(3))
#            if center_on_star: star_center = x_star[m_star.argmax()] else: star_center = np.zeros(3)
        if "PartType0" in F1.keys():
            x = float(k)/n_interp * x2 + (n_interp-float(k))/n_interp * x1 - star_center
#            if center_on_star and sink_type in F1.keys(): x -= star_center
            
            logu = float(k)/n_interp * np.log10(u2) + (n_interp-float(k))/n_interp * np.log10(u1)
            u = 10**logu

            h = float(k)/n_interp * h2 + (n_interp-float(k))/n_interp * h1
#            print(m)
            sigma_gas = GridSurfaceDensity(m, x, h, res, L).T
            Tmap_gas = GridAverage(u, x, h, res, L).T/1.01e4 #should be similar to mass weighted average if partcile masses roughly constant, also converting to K
        else:
            sigma_gas = np.zeros((res,res))
            Tmap_gas = np.zeros((res,res))
        #Gas surface density
        fgas = (np.log10(sigma_gas)-np.log10(limits[0]))/np.log10(limits[1]/limits[0])
        fgas = np.clip(fgas,0,1)
        data = fgas[:,:,np.newaxis]*plt.get_cmap(cmap)(fgas)[:,:,:3] 
        data = np.clip(data,0,1)
        #Gas temperature map
        fTgas = (np.log10(Tmap_gas)-np.log10(Tlimits[0]))/np.log10(Tlimits[1]/Tlimits[0])
        fTgas = np.clip(fTgas,0,1)
        Tdata = fTgas[:,:,np.newaxis]*plt.get_cmap(Tcmap)(fTgas)[:,:,:3] 
        Tdata = np.clip(Tdata,0,1)

#        print(i)
        file_number = file_numbers[i]
        filename = "SurfaceDensity_%s.%s.png"%(str(file_number).zfill(4),k)
        Tfilename = "Temperature_%s.%s.png"%(str(file_number).zfill(4),k)
        if outputfolder:
            filename=outputfolder+'/'+filename
            Tfilename=outputfolder+'/'+Tfilename
        plt.imsave(filename, data) #f.split("snapshot_")[1].split(".hdf5")[0], map)
        print(filename)
        if plot_T_map:
            plt.imsave(Tfilename, Tdata) #f.split("snapshot_")[1].split(".hdf5")[0], map)
            print(Tfilename)
            flist = [filename, Tfilename]
        else:
            flist = [filename]
    
        for fname in flist:
            F = Image.open(fname)
            draw = ImageDraw.Draw(F)
            gridres=res
            draw.line(((gridres/16, 7*gridres/8), (gridres*5/16, 7*gridres/8)), fill="#FFFFFF", width=6)
            draw.text((gridres/16, 7*gridres/8 + 5), "%gpc"%(r*500/1000), font=font)
            draw.text((gridres/16, gridres/24), "%3.2gMyr"%(time*979), font=font)
            if sink_type in F1.keys():
#                p = aggdraw.Brush((255, 0,0))
#                print(len(x_star))
                #for X in x_star:
#                    coords = np.concatenate([(X[:2]+r)/(2*r)*gridres-gridres/800, (X[:2]+r)/(2*r)*gridres+gridres/800])
#                    d.ellipse(coords, pen, p)#, fill=(155, 176, 255))
                #(155, 176, 255))
                d = aggdraw.Draw(F)
                pen = aggdraw.Pen("white",gridres/800)
                for j in np.arange(len(x_star))[m_star>0]:
                    X = x_star[j] - star_center
                    ms = m_star[j]
                    star_size = gridres/400 * (ms/sink_scale)**(1./3)
                    star_size = max(1,star_size)
                    p = aggdraw.Brush(StarColor(ms))
                    X -= boxsize/2 + center
                    coords = np.concatenate([(X[:2]+r)/(2*r)*gridres-star_size, (X[:2]+r)/(2*r)*gridres+star_size])
                    d.ellipse(coords, pen, p)#, fill=(155, 176, 255))
                d.flush()
            F.save(fname)
            F.close()

def MakeMovie():
    #Plotting surface density
    #Find files
    if outputfolder:
        filenames=natsorted(glob(outputfolder+'/'+'SurfaceDensity_????.?.png'))
        framefile=outputfolder+'/'+"frames.txt"
        moviefilename=outputfolder+'/'+movie_name
    else:
        filenames=natsorted(glob('SurfaceDensity_????.?.png'))
        framefile="frames.txt"
        moviefilename=movie_name
    #Use ffmpeg to create movie
    file(framefile,'w').write('\n'.join(["file '%s'"%f for f in filenames]))
    os.system("ffmpeg -y -r " + str(fps) + " -f concat -i frames.txt  -vb 20M -pix_fmt yuv420p  -q:v 0 -vcodec mpeg4 " + moviefilename + ".mp4")
    #Erase files, leave movie only
    if only_movie:
        for i in filenames:
            os.remove(i)
    os.remove(framefile)
    #Plotting temperature
    if plot_T_map:
        #Find files
        if outputfolder:
            filenames=natsorted(glob(outputfolder+'/'+'Temperature_????.?.png'))
        else:
            filenames=natsorted(glob('Temperature_????.?.png'))
        #Use ffmpeg to create movie
        file(framefile,'w').write('\n'.join(["file '%s'"%f for f in filenames]))
        os.system("ffmpeg -y -r " + str(fps) + " -f concat -i frames.txt  -vb 20M -pix_fmt yuv420p  -q:v 0 -vcodec mpeg4 " + moviefilename + "_temp.mp4")
        #Erase files, leave movie only
        if only_movie:
            for i in filenames:
                os.remove(i)
        os.remove(framefile)

if nproc>1:
    Parallel(n_jobs=nproc)(delayed(MakeImage)(i) for i in range(len(filenames)))
else:
    [MakeImage(i) for i in range(len(filenames))]

if (len(filenames) > 1 and (not no_movie) ): 
    MakeMovie() # only make movie if plotting multiple files
