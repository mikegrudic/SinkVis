#!/usr/bin/env python
"""
Usage:
SinkVis.py <files> ... [options]

Options:
    -h --help              Show this screen.
    --rmax=<pc>            Maximum radius of plot window; defaults to box size/10.
    --dir=<x,y,z>          Coordinate direction to orient the image along - x, y, or z [default: z]
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
    --center_on_star       Center image on the N_high most massive sink particles
    --center_on_densest    Center image on the N_high sinks with the densest gas nearby
    --N_high=<N>           Number of sinks to center on using the center_on_star or center_on_densest flags [default: 1]
    --center_on_ID=<ID>    Center image on sink particle with specific ID, does not center if zero [default: 0]
    --galunits             Use default GADGET units
    --plot_T_map           Plots both surface density and average temperature maps
    --outputfolder=<name>  Specifies the folder to save the images and movies to
    --name_addition=<name> Extra string to be put after the name of the ouput files, defaults to empty string       
    --no_pickle            Flag, if set no pickle file is created to make replots faster
    --no_timestamp         Flag, if set no timestamp will be put on the images
    --no_size_scale        Flag, if set no size scale will be put on the images
    --draw_axes            Flag, if set the coordinate axes are added to the figure
"""

#Example
# python SinkVis.py /panfs/ds08/hopkins/guszejnov/GMC_sim/Tests/200msun/MHD_isoT_2e6/output/snapshot*.hdf5 --np=24 --only_movie --movie_name=200msun_MHD_isoT_2e6

#import meshoid
from Meshoid import GridSurfaceDensityMultigrid, GridAverage
import Meshoid
from scipy.spatial import cKDTree
import h5py
import matplotlib
matplotlib.use('Agg')
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
import pickle

def find_sink_in_densest_gas(snapnum):
    #Find the N_high sinks in snapshot near the densest gas and return their ID, otherwise return 0
    #Check if we have looked for this one before
    filename = "Sinkvis_snap%d_gas_density_around_sinks.pickle"%(snapnum)
    if outputfolder:
        filename=outputfolder+'/'+filename
    if not os.path.exists(filename):
        print("Looking for the sink particle with the densest gas around it...")
        numpart_total=load_from_snapshot("NumPart_Total",0,datafolder,snapnum)
        if (numpart_total[0] and numpart_total[sink_type]):
            Ngb_target = 32
            #load_from_snapshot("keys",0,datafolder,snapnum)
            #load_from_snapshot("keys",5,datafolder,snapnum)
            #First,load the sinks 
            ids = np.array(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum))
            Nsink = len(ids)
            xs = length_unit*np.array(load_from_snapshot("Coordinates",sink_type,datafolder,snapnum))
            hs = length_unit*np.array(load_from_snapshot("SinkRadius",sink_type,datafolder,snapnum))
            ms = mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum))
            #Let's load the gas densities and pick out the ones that are densest
            rho = load_from_snapshot("Density",0,datafolder,snapnum)
            dense_ind = rho>np.percentile(rho,np.min([99.0,np.max([100*(1.0-Nsink*10000/len(rho)),0])])) #denser than 99% of the gas or less if few particles
            rho = rho[dense_ind] * mass_unit/(length_unit**3)
            xg = length_unit*np.array(load_from_snapshot("Coordinates",0,datafolder,snapnum))[dense_ind,:]
            gas_tree = cKDTree(xg)
            sink_gas_neighbors = gas_tree.query(xs,32)[1]
            max_neighbor_gas_density = np.max(rho[sink_gas_neighbors],axis=1)
            #Reorder sinks by neighbor gas density
            sink_order = max_neighbor_gas_density.argsort()[::-1]
            ids = ids[sink_order]
            xs = xs[sink_order,:]
            ms = ms[sink_order]
            max_neighbor_gas_density = max_neighbor_gas_density[sink_order]
            #Pick the ones we want 
            print("Sink particle with densest gas are:")
            for i in range(N_high):
                print("\t ID %d at %g %g %g with mass %g and %g neighboring gas density"%(ids[i],xs[i,0],xs[i,1],xs[i,2],ms[i],max_neighbor_gas_density[i]))
            print("Saving "+filename)
            outfile = open(filename, 'wb') 
            pickle.dump([ids, xs, ms, max_neighbor_gas_density], outfile)
            outfile.close()
            return ids[:N_high]
        else:
            print("No gas or sinks present")
            return [0]
    else:
        print("Loading data from "+filename)
        infile = open(filename, 'rb') 
        temp = pickle.load(infile)
        infile.close()
        ids = temp[0]; xs = temp[1]; ms = temp[2]; max_neighbor_gas_density = temp[3]; 
        for i in range(N_high):
            print("\t ID %d at %g %g %g with mass %g and %g neighboring gas density"%(ids[i],xs[i,0],xs[i,1],xs[i,2],ms[i],max_neighbor_gas_density[i]))
        return ids[-N_high:]
            
def CoordTransform(x):
    return np.roll(x, {'z': 0, 'y': 1, 'x': 2}[arguments["--dir"]], axis=1)

def StarColor(mass_in_msun,cmap):
    if cmap=='afmhot':
        star_colors = np.array([[255, 100, 60],[120, 200, 150],[75, 80, 255]]) #alternate colors, red-green-blue, easier to see on a bright color map
    else:
        star_colors = np.array([[255, 203, 132],[255, 243, 233],[155, 176, 255]]) #default colors, reddish for small ones, yellow-white for mid sized and blue for large
    colors = np.int_([np.interp(np.log10(mass_in_msun),[-1,0,1],star_colors[:,i]) for i in range(3)])
    return (colors[0],colors[1],colors[2])# if len(colors)==1 else colors)

def MakeImage(i):
    global center_on_ID
    global limits
    global Tlimits
#    print(i)
    snapnum1=file_numbers[i]
    snapnum2=(file_numbers[min(i+1,len(filenames)-1)] if n_interp>1 else snapnum1)
    
    sink_IDs_to_center_on = np.array([center_on_ID]) #default, will not center
    if center_on_densest:
        #Find the IDs of the sinks with the densest gas nearby 
        sink_IDs_to_center_on=find_sink_in_densest_gas(snapnum1)
    if center_on_star:
        #Find the IDs of the most massive sinks
        id1s = np.array(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum1))
        m1s = mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum1))
        sink_IDs_to_center_on=id1s[m1s.argsort()[-N_high:]] #choose the N_high most massive
    for sink_ID in sink_IDs_to_center_on:
        pickle_filename = "Sinkvis_snap%d_%d_%d_r%g_res%d_c%g_%g_%g_0_%d_%s.pickle"%(snapnum1,0,n_interp,r,res,center[0],center[1],center[2],sink_ID,arguments["--dir"])
        if outputfolder:
            pickle_filename=outputfolder+'/'+pickle_filename
        if not os.path.exists(pickle_filename):
            print("Loading snapshot data from "+filenames[i])
            #We don't have the data, must read it from snapshot
            #keylist=load_from_snapshot("keys",0,datafolder,snapnum1)
            numpart_total=load_from_snapshot("NumPart_Total",0,datafolder,snapnum1)
            if not numpart_total[sink_type] and (center_on_star or (sink_ID>0)): return
            if numpart_total[sink_type]:
                id1s, id2s = np.array(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum1)), np.array(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum2))
                unique, counts = np.unique(id2s, return_counts=True)
                doubles = unique[counts>1]
                id2s[np.in1d(id2s,doubles)]=-1
                x1s, x2s = length_unit*np.array(load_from_snapshot("Coordinates",sink_type,datafolder,snapnum1))[id1s.argsort()], length_unit*np.array(load_from_snapshot("Coordinates",sink_type,datafolder,snapnum2))[id2s.argsort()]
                x1s, x2s = CoordTransform(x1s), CoordTransform(x2d)
                m1s, m2s = mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum1))[id1s.argsort()], mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum2))[id2s.argsort()]
                # take only the particles that are in both snaps
                common_sink_ids = np.intersect1d(id1s,id2s)
                idx1 = np.in1d(np.sort(id1s),common_sink_ids)
                idx2 = np.in1d(np.sort(id2s),common_sink_ids)
                x1s = x1s[idx1]; m1s = m1s[idx1]
                x2s = x2s[idx2]; m2s = m2s[idx2]
                m_star = m2s
                if ((sink_ID>0) and (not np.any(common_sink_ids==sink_ID)) ): 
                    print("Sink ID %d not present in "%(sink_ID)+filenames[i])
                    print("Sink IDs present: ",np.int64(common_sink_ids))
                    print("Masses of present sinks: ",m_star)
                    print("Positions of present sinks: ",x1s-boxsize/2)
                    ids_m=np.int64(common_sink_ids[m_star>2])
                    ms_m=m_star[m_star>2]
                    dxs_m=x1s[m_star>2]-boxsize/2
                    #sort, for now by x-y radial distance
                    drs_m=np.sqrt(dxs_m[:,0]**2+dxs_m[:,1]**2)
                    sortind = np.argsort(drs_m)
                    ids_m=ids_m[sortind]; ms_m=ms_m[sortind]; dxs_m=dxs_m[sortind,:]
                    print("Massive sink IDs: ",ids_m)
                    print("Massive masses sinks: ",ms_m)
                    print("Positions of massive sinks: ",dxs_m)
                    sinkfilename = "Sinkvis_snap%d_massive_sinks.txt"%(snapnum1)
                    if outputfolder:
                        sinkfilename=outputfolder+'/'+sinkfilename
                    np.savetxt(sinkfilename,np.transpose(np.array([np.int64(ids_m),ms_m,dxs_m[:,0],dxs_m[:,1],dxs_m[:,2]])))
                    return
            if numpart_total[0]:
                id1, id2 = np.array(load_from_snapshot("ParticleIDs",0,datafolder,snapnum1)), np.array(load_from_snapshot("ParticleIDs",0,datafolder,snapnum2))
                unique, counts = np.unique(id2, return_counts=True)
                doubles = unique[counts>1]
                id2[np.in1d(id2,doubles)]=-1
                x1, x2 = length_unit*np.array(load_from_snapshot("Coordinates",0,datafolder,snapnum1))[id1.argsort()], length_unit*np.array(load_from_snapshot("Coordinates",0,datafolder,snapnum2))[id2.argsort()]
                x1, x2 = CoordTransform(x1), CoordTransform(x2)
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
                # unload stuff to save memory
                idx1=0; idx2=0; id1=0; id2=0;
            time = load_from_snapshot("Time",0,datafolder,snapnum1)
        for k in range(n_interp):
            pickle_filename = "Sinkvis_snap%d_%d_%d_r%g_res%d_c%g_%g_%g_0_%d_%s.pickle"%(snapnum1,k,n_interp,r,res,center[0],center[1],center[2],sink_ID,arguments["--dir"])
            if outputfolder:
                pickle_filename=outputfolder+'/'+pickle_filename
            if not os.path.exists(pickle_filename):
                if numpart_total[sink_type]:
                    x_star = float(k)/n_interp * x2s + (n_interp-float(k))/n_interp * x1s
                else:
                    x_star = []; m_star = [];
                #star_center =  (x_star[m_star.argmax()]-boxsize/2 if ((center_on_star or (sink_ID>0)) and numpart_total[sink_type]) else np.zeros(3))
                star_center = np.zeros(3)
                if sink_ID:
                    star_center = np.squeeze(x_star[common_sink_ids==sink_ID]-boxsize/2)
                if numpart_total[0]:
                    x = float(k)/n_interp * x2 + (n_interp-float(k))/n_interp * x1
                    #correct for periodic box
                    jump_ind1 = (x2 - x1) > (boxsize/2) #assuming no particle travels more than half of the the box in a single snapshot
                    jump_ind2 = (x1 - x2) > (boxsize/2)
                    if np.any(jump_ind1):
                        x[jump_ind1] = (float(k)/n_interp * (x2[jump_ind1]-boxsize) + (n_interp-float(k))/n_interp * x1[jump_ind1])%boxsize 
                    if np.any(jump_ind2):
                        x[jump_ind2] = (float(k)/n_interp * x2[jump_ind2] + (n_interp-float(k))/n_interp * (x1[jump_ind2]-boxsize))%boxsize 
                    x -= star_center

                    logu = float(k)/n_interp * np.log10(u2) + (n_interp-float(k))/n_interp * np.log10(u1)
                    u = 10**logu

                    h = float(k)/n_interp * h2 + (n_interp-float(k))/n_interp * h1
                    h = np.clip(h,L/res, 1e100)
                    sigma_gas = GridSurfaceDensity(m, x, h, star_center*0, L, res=res).T
                    if plot_T_map:
                        Tmap_gas = GridAverage(u, x, h,star_center*0, L, res=res).T/1.01e4 #should be similar to mass weighted average if partcile masses roughly constant, also converting to K
                        logTmap_gas = GridAverage(np.log10(u/1.01e4), x, h,star_center*0, L, res=res).T #average of log T so that it is not completely dominated by the warm ISM
                    else:
                        Tmap_gas = np.zeros((res,res))
                        logTmap_gas = np.zeros((res,res))
                else:
                    sigma_gas = np.zeros((res,res))
                    Tmap_gas = np.zeros((res,res))
                    logTmap_gas = np.zeros((res,res))
                #Save data
                if not no_pickle:
                    print("Saving "+pickle_filename)
                    outfile = open(pickle_filename, 'wb') 
                    pickle.dump([x_star,m_star,sigma_gas,Tmap_gas,logTmap_gas,time,numpart_total, star_center], outfile)
                    outfile.close()
            else:
                #Load data from pickle file
                print("Loading "+pickle_filename)
                infile = open(pickle_filename, 'rb') 
                temp = pickle.load(infile)
                infile.close()
                x_star = temp[0]; m_star = temp[1]; sigma_gas = temp[2]; Tmap_gas = temp[3]; logTmap_gas = temp[4]; time = temp[5]; numpart_total = temp[6];
                star_center = temp[7]
                temp = 0; #unload
            #Adjust limits if not set
            if ((limits[0]==0) or (limits[1]==0)):
                limits[1]=2.0*np.percentile(sigma_gas,99.9)
                if cmap=='afmhot':
                    limits[1]*=3.0
                limits[0]=0.5*np.min([limits[1]*1e-2,np.max([limits[1]*1e-4,np.percentile(sigma_gas,5)])])
                print("Using surface density limits of %g and %g"%(limits[0],limits[1]))
            #Gas surface density
            fgas = (np.log10(sigma_gas)-np.log10(limits[0]))/np.log10(limits[1]/limits[0])
            fgas = np.clip(fgas,0,1)
            data = plt.get_cmap(cmap)(fgas)
            data = np.clip(data,0,1)
            #Adjust Tlimits if not set
            if ((Tlimits[0]==0) or (Tlimits[1]==0)):
                Tlimits[1]=np.percentile(Tmap_gas,99)
                Tlimits[0]=np.min([Tlimits[1]*1e-2,np.max([Tlimits[1]*1e-4,np.percentile(Tmap_gas,5)])])
                print("Using temperature limits of %g K and %g K"%(Tlimits[0],Tlimits[1]))
                logTlimits[1]=np.percentile(logTmap_gas,99)
                logTlimits[0]=np.min([logTlimits[1]-2,np.max([logTlimits[1]-4,np.percentile(logTmap_gas,5)])])
                print("Using log temperature limits of %g and %g"%(logTlimits[0],logTlimits[1]))
            #Gas temperature map
            fTgas = (np.log10(Tmap_gas)-np.log10(Tlimits[0]))/np.log10(Tlimits[1]/Tlimits[0])
            fTgas = np.clip(fTgas,0,1)
            Tdata = fTgas[:,:,np.newaxis]*plt.get_cmap(Tcmap)(fTgas)[:,:,:3] 
            Tdata = np.clip(Tdata,0,1)
            #Gas log temperature map
            flogTgas = (logTmap_gas-logTlimits[0])/(logTlimits[1]-logTlimits[0])
            flogTgas = np.clip(flogTgas,0,1)
            logTdata = flogTgas[:,:,np.newaxis]*plt.get_cmap(Tcmap)(flogTgas)[:,:,:3] 
            logTdata = np.clip(logTdata,0,1)
            
            local_name_addition = name_addition
            if sink_ID and (len(sink_IDs_to_center_on)>1):
                local_name_addition = '_%d'%(sink_ID) + local_name_addition
            file_number = file_numbers[i]
            filename = "SurfaceDensity%s_%s.%s.png"%(local_name_addition,str(file_number).zfill(4),k)
            Tfilename = "Temperature%s_%s.%s.png"%(local_name_addition,str(file_number).zfill(4),k)
            logTfilename = "LogTemperature%s_%s.%s.png"%(local_name_addition,str(file_number).zfill(4),k)
            if outputfolder:
                filename=outputfolder+'/'+filename
                Tfilename=outputfolder+'/'+Tfilename
                logTfilename=outputfolder+'/'+logTfilename
            plt.imsave(filename, data) #f.split("snapshot_")[1].split(".hdf5")[0], map)
            print(filename)
            if plot_T_map:
                plt.imsave(Tfilename, Tdata) #f.split("snapshot_")[1].split(".hdf5")[0], map)
                print(Tfilename)
                plt.imsave(logTfilename, logTdata) #f.split("snapshot_")[1].split(".hdf5")[0], map)
                print(logTfilename)
                flist = [filename, Tfilename,logTfilename]
            else:
                flist = [filename]

            for fname in flist:
                F = Image.open(fname)
                draw = ImageDraw.Draw(F)
                gridres=res
                if not no_size_scale:
                    if (r>1000):
                        scale_kpc=10**np.round(np.log10(r*0.5/1000))
                        size_scale_text="%3.3gkpc"%(scale_kpc)
                        size_scale_ending=gridres/16+gridres*(scale_kpc*1000)/(2*r)
                    if (r>1e-2):
                        scale_pc=10**np.round(np.log10(r*0.5))
                        size_scale_text="%3.3gpc"%(scale_pc)
                        size_scale_ending=gridres/16+gridres*(scale_pc)/(2*r)
                        #size_scale_ending=gridres/16+gridres*0.25
                    else:
                        scale_AU=10**np.round(np.log10(r*0.5*pc_to_AU))
                        size_scale_text="%3.4gAU"%(scale_AU)
                        size_scale_ending=gridres/16+gridres*(scale_AU)/(2*r*pc_to_AU)
                    draw.line(((gridres/16, 7*gridres/8), (size_scale_ending, 7*gridres/8)), fill="#FFFFFF", width=6)
                    draw.text((gridres/16, 7*gridres/8 + 5), size_scale_text, font=font)
                if not no_timestamp:
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
                        p = aggdraw.Brush(StarColor(ms,cmap))
                        X -= boxsize/2 + center
                        coords = np.concatenate([(X[:2]+r)/(2*r)*gridres-star_size, (X[:2]+r)/(2*r)*gridres+star_size])
                        d.ellipse(coords, pen, p)#, fill=(155, 176, 255))
                    d.flush()
                F.save(fname)
                F.close()
                if draw_axes:
                    xlim = [boxsize/2.0+center[0]+star_center[0]-r,boxsize/2.0+center[0]+star_center[0]+r]
                    ylim = [boxsize/2.0+center[1]+star_center[1]-r,boxsize/2.0+center[1]+star_center[1]+r]
                    data = plt.imread(fname)
                    fig, ax = plt.subplots()
                    ax.imshow( data, extent=(xlim[0],xlim[1],ylim[0],ylim[1]) )
                    ax.set_xlabel("X [pc]")
                    ax.set_ylabel("Y [pc]")
                    fig.set_size_inches(6, 6)
                    #plt.figure(num=fig.number, figsize=(1.3*gridres/150, 1.2*gridres/150), dpi=550)
                    fig.savefig(fname,dpi=int(gridres/5))
            

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
                center_on_star=0, N_high=1, Tcmap="inferno", cmap="viridis", no_movie=True, outputfolder="output",\
                plot_T_map=True, sink_scale=0.1, sink_type=5, galunits=False,name_addition="",center_on_ID=0,no_pickle=False, no_timestamp=False,\
                no_size_scale=False, center_on_densest=False, draw_axes=False):
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
        "--no_pickle": no_pickle,
        "--fps": fps,
        "--movie_name": movie_name,
        "--sink_type": str(sink_type),
        "--sink_scale": sink_scale,
        "--galunits": galunits,
        "--center_on_star": center_on_star,
        "--center_on_ID": center_on_ID,
        "--center_on_densest": center_on_densest,
        "--N_high": N_high,
        "--Tcmap": Tcmap,
        "--cmap": cmap,
        "--no_movie": no_movie,
        "--outputfolder": outputfolder,
        "--plot_T_map": plot_T_map,
        "--name_addition": name_addition,
        "--no_timestamp": no_timestamp,
        "--no_size_scale": no_size_scale,
        "--draw_axes": draw_axes
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
    logTlimits = np.zeros(2)
    if Tlimits[0]:
        #used for log T plot
        logTlimits[:] = np.log10(Tlimits[:]) 
    res = int(arguments["--res"])
    nproc = int(arguments["--np"])
    n_interp = int(arguments["--interp_fac"])
    cmap = arguments["--cmap"]
    Tcmap = arguments["--Tcmap"]
    only_movie = arguments["--only_movie"]
    galunits = arguments["--galunits"]
    no_movie = arguments["--no_movie"]
    plot_T_map = arguments["--plot_T_map"]
    no_pickle = arguments["--no_pickle"]
    no_timestamp = arguments["--no_timestamp"]
    draw_axes = arguments["--draw_axes"]
    no_size_scale = arguments["--no_size_scale"]
    fps = float(arguments["--fps"])
    movie_name = arguments["--movie_name"]
    outputfolder = arguments["--outputfolder"]
    sink_type = int(arguments["--sink_type"])
    sink_type_text="PartType" + str(sink_type)
    sink_scale = float(arguments["--sink_scale"])
    center_on_star = 1 if arguments["--center_on_star"] else 0
    center_on_ID = int(arguments["--center_on_ID"]) if arguments["--center_on_ID"] else 0
    N_high = int(arguments["--N_high"])
    center_on_densest = 1 if arguments["--center_on_densest"] else 0
    L = r*2
    length_unit = (1e3 if galunits else 1.)
    mass_unit = (1e10 if galunits else 1.)
    pc_to_AU = 206265.0
    #i = 0
    boxsize *= length_unit
    r *= length_unit
    L *= length_unit

    font = ImageFont.truetype("LiberationSans-Regular.ttf", res//12) 
    
    if outputfolder:
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)
   
   #Try to guess surface density limits for multiple snap runs (can't guess within MakeImage routine due to parallelization, also the first image is unlikely to be useful as it is often just the IC) so we will run the last one first, without parallelization
    if ( ( (limits[0]==0) or (Tlimits[0]==0) ) and (len(filenames) > 1) ):
        print("Surface density or temperature limits not set, running final snapshot first to guess the appropriate values.")
        MakeImage(len(filenames)-1)
        
    if nproc>1:
        Pool(nproc).map(MakeImage, (f for f in range(len(filenames))))
    else:
        [MakeImage(i) for i in range(len(filenames))]

    if (len(filenames) > 1 and (not no_movie) ): 
        MakeMovie() # only make movie if plotting multiple files
