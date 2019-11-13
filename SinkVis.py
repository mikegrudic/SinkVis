#!/usr/bin/env python
"""
Usage:
SinkVis.py <files> ... [options]

Options:
    -h --help              Show this screen.
    --rmax=<pc>            Maximum radius of plot window; defaults to box size/10.
    --full_box             Sets the plot to the entire box, overrides rmax
    --c=<cx,cy,cz>         Coordinates of plot window center relative to box center [default: 0.0,0.0,0.0]
    --limits=<min,max>     Dynamic range of surface density colormap [default: 0,0]
    --Tlimits=<min,max>    Dynamic range of temperature colormap in K [default: 0,0]
    --Tcmap=<name>         Name of colormap to use for temperature [default: inferno]
    --cmap=<name>          Name of colormap to use [default: viridis]
    --interp_fac=<N>       Number of interpolating frames per snapshot [default: 1]
    --np=<N>               Number of processors to run on [default: 1]
    --res=<N>              Image resolution [default: 500]
    --only_movie           Only the movie is saved, the images are removed at the end
    --no_movie             Does not create a movie, only makes images
    --fps=<fps>            Frame per second for movie [default: 20]
    --movie_name=<name>    Filename of the output movie file without format [default: sink_movie]
    --sink_type=<N>        Particle type of sinks [default: 5]
    --sink_scale=<msun>    Sink particle mass such that apparent sink size is 1 pixel [default: 0.1]
    --center_on_star       Center image on the most massive sink particle
    --galunits             Use default GADGET units
    --plot_T_map           Plots both surface density and average temperature maps
    --outputfolder=<name>  Specifies the folder to save the images and movies to
    --name_addition=<name> Extra string to be put after the name of the ouput files, defaults to empty string       
"""

#Example
# python SinkVis.py /panfs/ds08/hopkins/guszejnov/GMC_sim/Tests/200msun/MHD_isoT_2e6/output/snapshot*.hdf5 --np=24 --only_movie --movie_name=200msun_MHD_isoT_2e6

#import meshoid
from Meshoid import GridSurfaceDensity, GridAverage
import h5py
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
#from joblib import Parallel, delayed
from multiprocessing import Pool
import aggdraw
from natsort import natsorted
from docopt import docopt
from glob import glob
import os
from sys import argv
from load_from_snapshot import load_from_snapshot
import re






def TransformCoords(x, angle):
    return np.c_[x[:,0]*np.cos(angle) + x[:,1]*np.sin(angle), -x[:,0]*np.sin(angle) + x[:,1]*np.cos(angle), x[:,2]]

def StarColor(mass_in_msun):
    star_colors = np.array([[255, 203, 132],[255, 243, 233],[155, 176, 255]])
    colors = np.int_([np.interp(np.log10(mass_in_msun),[-1,0,1],star_colors[:,i]) for i in range(3)])
    return (colors[0],colors[1],colors[2])# if len(colors)==1 else colors)

def MakeImage(i):
#    print(i)
    snapnum1=file_numbers[i]
    snapnum2=(file_numbers[min(i+1,len(filenames)-1)] if n_interp>1 else snapnum1)
    #keylist=load_from_snapshot("keys",0,datafolder,snapnum1)
    numpart_total=load_from_snapshot("NumPart_Total",0,datafolder,snapnum1)
    if not numpart_total[sink_type] and center_on_star: return
    if numpart_total[0]:
        id1, id2 = np.array(load_from_snapshot("ParticleIDs",0,datafolder,snapnum1)), np.array(load_from_snapshot("ParticleIDs",0,datafolder,snapnum2))
        unique, counts = np.unique(id2, return_counts=True)
        doubles = unique[counts>1]
        id2[np.in1d(id2,doubles)]=-1
        x1, x2 = length_unit*np.array(load_from_snapshot("Coordinates",0,datafolder,snapnum1))[id1.argsort()], length_unit*np.array(load_from_snapshot("Coordinates",0,datafolder,snapnum2))[id2.argsort()]
        if not galunits:
            x1 -= boxsize/2 + center
            x2 -= boxsize/2 + center
        u1, u2 = np.array(load_from_snapshot("InternalEnergy",0,datafolder,snapnum1))[id1.argsort()], np.array(load_from_snapshot("InternalEnergy",0,datafolder,snapnum2))[id2.argsort()]
        h1, h2 = length_unit*np.array(load_from_snapshot("SmoothingLength",0,datafolder,snapnum1))[id1.argsort()], length_unit*np.array(load_from_snapshot("SmoothingLength",0,datafolder,snapnum2))[id2.argsort()]
        m1, m2 = mass_unit*np.array(load_from_snapshot("Masses",0,datafolder,snapnum1))[id1.argsort()], mass_unit*np.array(load_from_snapshot("Masses",0,datafolder,snapnum2))[id2.argsort()]
        # take only the particles that are in both snaps
        common_ids = np.intersect1d(id1,id2)
        idx1 = np.in1d(np.sort(id1),common_ids)
        idx2 = np.in1d(np.sort(id2),common_ids)
        x1 = x1[idx1]; u1 = u1[idx1]; h1 = h1[idx1]; m1 = m1[idx1]
        x2 = x2[idx2]; u2 = u2[idx2]; h2 = h2[idx2]; m2 = m2[idx2]
        m = m2
    if numpart_total[sink_type]:
        id1s, id2s = np.array(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum1)), np.array(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum2))
        unique, counts = np.unique(id2s, return_counts=True)
        doubles = unique[counts>1]
        id2s[np.in1d(id2s,doubles)]=-1
        x1s, x2s = length_unit*np.array(load_from_snapshot("Coordinates",sink_type,datafolder,snapnum1))[id1s.argsort()], length_unit*np.array(load_from_snapshot("Coordinates",sink_type,datafolder,snapnum2))[id2s.argsort()]
        m1s, m2s = mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum1))[id1s.argsort()], mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum2))[id2s.argsort()]
        # take only the particles that are in both snaps
        common_ids = np.intersect1d(id1s,id2s)
        idx1 = np.in1d(np.sort(id1s),common_ids)
        idx2 = np.in1d(np.sort(id2s),common_ids)
        x1s = x1s[idx1]; m1s = m1s[idx1]
        x2s = x2s[idx2]; m2s = m2s[idx2]
        m_star = m2s
    time = load_from_snapshot("Time",0,datafolder,snapnum1)
    for k in range(n_interp):
        if numpart_total[sink_type]:
            x_star = float(k)/n_interp * x2s + (n_interp-float(k))/n_interp * x1s
        star_center =  (x_star[m_star.argmax()]-boxsize/2 if (center_on_star and numpart_total[sink_type]) else np.zeros(3)) 
        if numpart_total[0]:
            x = float(k)/n_interp * x2 + (n_interp-float(k))/n_interp * x1 - star_center

            logu = float(k)/n_interp * np.log10(u2) + (n_interp-float(k))/n_interp * np.log10(u1)
            u = 10**logu

            h = float(k)/n_interp * h2 + (n_interp-float(k))/n_interp * h1
            h = np.clip(h,L/res, 1e100)
            sigma_gas = GridSurfaceDensity(m, x, h, star_center*0, L, res=res).T
            Tmap_gas = GridAverage(u, x, h,star_center*0, L, res=res).T/1.01e4 #should be similar to mass weighted average if partcile masses roughly constant, also converting to K
        else:
            sigma_gas = np.zeros((res,res))
            Tmap_gas = np.zeros((res,res))
        #Adjust limits if not set
        if ((limits[0]==0) or (limits[1]==0)):
            limits[1]=1.2*np.percentile(sigma_gas,99)
            limits[0]=0.8*np.min([limits[1]*1e-2,np.max([limits[1]*1e-4,np.percentile(sigma_gas,5)])])
            print("Using surface density limits of %g and %g"%(limits[0],limits[1]))
        #Gas surface density
        fgas = (np.log10(sigma_gas)-np.log10(limits[0]))/np.log10(limits[1]/limits[0])
        fgas = np.clip(fgas,0,1)
        data = plt.get_cmap(cmap)(fgas)
        data = np.clip(data,0,1)
        #Adjust Tlimits if not set
        if ((Tlimits[0]==0) or (Tlimits[1]==0)):
            Tlimits[1]=np.percentile(Tmap_gas,95)
            Tlimits[0]=np.min([Tlimits[1]*1e-2,np.max([Tlimits[1]*1e-4,np.percentile(Tmap_gas,5)])])
            print("Using temperature limits of %g K and %g K"%(Tlimits[0],Tlimits[1]))
        #Gas temperature map
        fTgas = (np.log10(Tmap_gas)-np.log10(Tlimits[0]))/np.log10(Tlimits[1]/Tlimits[0])
        fTgas = np.clip(fTgas,0,1)
        Tdata = fTgas[:,:,np.newaxis]*plt.get_cmap(Tcmap)(fTgas)[:,:,:3] 
        Tdata = np.clip(Tdata,0,1)

        file_number = file_numbers[i]
        filename = "SurfaceDensity%s_%s.%s.png"%(name_addition,str(file_number).zfill(4),k)
        Tfilename = "Temperature%s_%s.%s.png"%(name_addition,str(file_number).zfill(4),k)
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
            if (r>1e-2):
                size_scale_text="%3.2gpc"%(r*500/1000)
                size_scale_ending=gridres/16+gridres*0.25
            else:
                new_scale_AU=10**np.round(np.log10(r*0.5*pc_to_AU))
                size_scale_text="%3.2gAU"%(new_scale_AU)
                size_scale_ending=gridres/16+gridres*(new_scale_AU)/(2*r*pc_to_AU)
            draw.line(((gridres/16, 7*gridres/8), (size_scale_ending, 7*gridres/8)), fill="#FFFFFF", width=6)
            draw.text((gridres/16, 7*gridres/8 + 5), size_scale_text, font=font)
            if (time*979>=1e-2):
                time_text="%3.2gMyr"%(time*979)
            elif(time*979>=1e-4):
                time_text="%3.2gkyr"%(time*979*1e3)
            else:
                time_text="%3.2gyr"%(time*979*1e6)
            draw.text((gridres/16, gridres/24), time_text, font=font)
            if numpart_total[sink_type]:
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
        filenames=natsorted(glob(outputfolder+'/'+'SurfaceDensity'+name_addition+'_????.?.png'))
        framefile=outputfolder+'/'+"frames.txt"
        moviefilename=outputfolder+'/'+movie_name
    else:
        filenames=natsorted(glob('SurfaceDensity'+name_addition+'_????.?.png'))
        framefile="frames.txt"
        moviefilename=movie_name
    #Use ffmpeg to create movie
    open(framefile,'w').write('\n'.join(["file '%s'"%f for f in filenames]))
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
            filenames=natsorted(glob(outputfolder+'/'+'Temperature'+name_addition+'_????.?.png'))
        else:
            filenames=natsorted(glob('Temperature'+name_addition+'_????.?.png'))
        #Use ffmpeg to create movie
        file(framefile,'w').write('\n'.join(["file '%s'"%f for f in filenames]))
        os.system("ffmpeg -y -r " + str(fps) + " -f concat -i frames.txt  -vb 20M -pix_fmt yuv420p  -q:v 0 -vcodec mpeg4 " + moviefilename + "_temp.mp4")
        #Erase files, leave movie only
        if only_movie:
            for i in filenames:
                os.remove(i)
        os.remove(framefile)
            

def Sinkvis_input(files="snapshot_000.hdf5", rmax=False, full_box=False, center=[0,0,0],limits=[0,0],Tlimits=[0,0],\
                interp_fac=1, np=1,res=500, only_movie=False, fps=20, movie_name="sink_movie",\
                center_on_star=False, Tcmap="inferno", cmap="viridis", no_movie=True, outputfolder="output",\
                plot_T_map=True, sink_scale=0.1, sink_type=5, galunits=False,name_addition=""):
    if (not isinstance(files, list)):
        files=[files]
    arguments={
        "<files>": files,
        "--rmax": rmax,
        "--full_box": full_box,
        "--c": str(center[0])+","+str(center[1])+","+str(center[2]),
        "--limits": str(limits[0])+","+str(limits[1]),
        "--Tlimits": str(Tlimits[0])+","+str(Tlimits[1]),
        "--interp_fac": interp_fac,
        "--np": np,
        "--res": res,
        "--only_movie": only_movie,
        "--fps": fps,
        "--movie_name": movie_name,
        "--sink_type": str(sink_type),
        "--sink_scale": sink_scale,
        "--galunits": galunits,
        "--center_on_star": center_on_star,
        "--Tcmap": Tcmap,
        "--cmap": cmap,
        "--no_movie": no_movie,
        "--outputfolder": outputfolder,
        "--plot_T_map": plot_T_map,
        "--name_addition": name_addition
        }
    return arguments

if __name__ == "__main__":
    arguments = docopt(__doc__)

    filenames = natsorted(arguments["<files>"])
    if os.path.isdir(filenames[0]):
        namestring="snapdir"
    else:
        namestring="snapshot"
    file_numbers = [int(re.search(namestring+'_\d*', f).group(0).replace(namestring+'_','')) for f in filenames]
    datafolder=(filenames[0].split(namestring+"_")[0])
    if not len(datafolder):
        datafolder="./"
    boxsize=load_from_snapshot("BoxSize",0,datafolder,file_numbers[0])
    full_box_flag = arguments["--full_box"]
    if full_box_flag:
        r = boxsize/2.0
    elif arguments["--rmax"]:
        r = float(arguments["--rmax"])
    else:
        r = boxsize/10
    name_addition = arguments["--name_addition"] if arguments["--name_addition"] else ""
    center = np.array([float(c) for c in arguments["--c"].split(',')])
    limits = np.array([float(c) for c in arguments["--limits"].split(',')])
    Tlimits = np.array([float(c) for c in arguments["--Tlimits"].split(',')])
    res = int(arguments["--res"])
    nproc = int(arguments["--np"])
    n_interp = int(arguments["--interp_fac"])
    cmap = arguments["--cmap"]
    Tcmap = arguments["--Tcmap"]
    only_movie = arguments["--only_movie"]
    galunits = arguments["--galunits"]
    no_movie = arguments["--no_movie"]
    plot_T_map = arguments["--plot_T_map"]
    fps = float(arguments["--fps"])
    movie_name = arguments["--movie_name"]
    outputfolder = arguments["--outputfolder"]
    sink_type = int(arguments["--sink_type"])
    sink_type_text="PartType" + str(sink_type)
    sink_scale = float(arguments["--sink_scale"])
    center_on_star = arguments["--center_on_star"]
    L = r*2
    length_unit = (1e3 if galunits else 1.)
    mass_unit = (1e10 if galunits else 1.)
    pc_to_AU = 206265.0
    #i = 0
    boxsize *= length_unit
    r *= length_unit
    L *= length_unit

    font = ImageFont.truetype("LiberationSans-Regular.ttf", res//12) 

    if nproc>1:
        Pool(nproc).map(MakeImage, (f for f in range(len(filenames))))
    else:
        [MakeImage(i) for i in range(len(filenames))]

    if (len(filenames) > 1 and (not no_movie) ): 
        MakeMovie() # only make movie if plotting multiple files
