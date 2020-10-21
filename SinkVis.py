#!/usr/bin/env python
"""
Usage:
SinkVis.py <files> ... [options]

Options:
    -h --help                  Show this screen.
    --rmax=<pc>                Maximum radius of plot window; defaults to box size/10.
    --dir=<x,y,z>              Coordinate direction to orient the image along - x, y, or z [default: z]
    --full_box                 Sets the plot to the entire box, overrides rmax
    --c=<cx,cy,cz>             Coordinates of plot window center relative to box center [default: 0.0,0.0,0.0]
    --limits=<min,max>         Dynamic range of surface density colormap [default: 0,0]
    --Tlimits=<min,max>        Dynamic range of temperature colormap in K [default: 0,0]
    --energy_limits=<min,max>  Dynamic range of kinetic energy colormap in code units [default: 0,0]
    --ecmap=<name>             Name of colormap to use for kinetic energy [default: viridis]
    --Tcmap=<name>             Name of colormap to use for temperature [default: inferno]
    --cmap=<name>              Name of colormap to use [default: viridis]
    --cool_cmap=<name>         Name of colormap to use for plot_cool_map, defaults to same as cmap [default: same]
    --interp_fac=<N>           Number of interpolating frames per snapshot [default: 1]
    --np=<N>                   Number of processors to run on [default: 1]
    --res=<N>                  Image resolution [default: 512]
    --v_res=<N>                Resolution for overplotted velocity field if plot_v_map is on [default: 32]
    --velocity_scale=<f>       Scale for the quivers when using plot_v_map, in m/s [default: 1000]
    --arrow_color=<name>       Color of the velocity arrows if plot_v_map is enabled, [default: white]
    --slice_height=<pc>        Calculation is only done on particles within a box of 2*slice_height size around the center (mostly for zoom-ins), no slicing if set to zero [default: 0]
    --only_movie               Only the movie is saved, the images are removed at the end
    --no_movie                 Does not create a movie, only makes images (legacy, default behavior now is not to make a movie)
    --make_movie               Also makes movie
    --fps=<fps>                Frame per second for movie [default: 24]
    --movie_name=<name>        Filename of the output movie file without format [default: sink_movie]
    --sink_type=<N>            Particle type of sinks [default: 5]
    --sink_scale=<msun>        Sink particle mass such that apparent sink size is 1 pixel for that and all asses below [default: 0.1]
    --sink_relscale=<f>        Relative size scale of a sink particles at 10xsink_scale to the entire picture, e.g. 0.01 means these stars will be 1% of the entire plotting area, [default: 0.0025]
    --center_on_star           Center image on the N_high most massive sink particles
    --center_on_densest        Center image on the N_high sinks with the densest gas nearby
    --N_high=<N>               Number of sinks to center on using the center_on_star or center_on_densest flags [default: 1]
    --center_on_ID=<ID>        Center image on sink particle with specific ID, does not center if zero [default: 0]
    --galunits                 Use default GADGET units
    --plot_T_map               Plots both surface density and average temperature maps
    --plot_v_map               Overplots velocity map on plots
    --plot_energy_map          Plots kinetic energy map
    --plot_cool_map            Plots cool map that looks cool
    --energy_v_scale=<v0>      Scale in the weighting of kinetic energy (w=m*(1+(v/v0)^2)), [default: 1000.0]
    --outputfolder=<name>      Specifies the folder to save the images and movies to
    --name_addition=<name>     Extra string to be put after the name of the ouput files, defaults to empty string       
    --no_pickle                Flag, if set no pickle file is created to make replots faster
    --no_timestamp             Flag, if set no timestamp will be put on the images
    --no_size_scale            Flag, if set no size scale will be put on the images
    --draw_axes                Flag, if set the coordinate axes are added to the figure
    --remake_only              Flag, if set SinkVis will only used already calculated pickle files, used to remake plots
    --rescale_hsml=<f>         Factor by which the smoothing lengths of the particles are rescaled [default: 1]
    --highlight_wind=<f>       Factor by which to increase wind particle masses if you want to highlight them [default: 1]
    --smooth_center=<Ns>       If not 0 and SinkVis is supposed to center on a particle (e.g. with center_on_ID) then the center coordinates are smoothed across Ns snapshots, [default: 0]
    --disable_multigrid        Disables GridSurfaceDensityMultigrid froms meshoid, uses slower GridSurfaceDensity instead
"""

#Example
# python SinkVis.py /panfs/ds08/hopkins/guszejnov/GMC_sim/Tests/200msun/MHD_isoT_2e6/output/snapshot*.hdf5 --np=24 --only_movie --movie_name=200msun_MHD_isoT_2e6

from Meshoid import GridSurfaceDensityMultigrid,GridSurfaceDensity, GridAverage
import Meshoid
from scipy.spatial import cKDTree
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import LightSource
import numpy as np
from multiprocessing import Pool
import aggdraw
from natsort import natsorted
from docopt import docopt
from glob import glob
import os
from sys import argv
from load_from_snapshot import load_from_snapshot,check_if_filename_exists
import re
import pickle

wind_ids = np.array([1913298393, 1913298394])

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
    if cmap=='afmhot' or cmap=='inferno':
        star_colors = np.array([[255, 100, 60],[120, 200, 150],[75, 80, 255]]) #alternate colors, red-green-blue, easier to see on a bright color map
    else:
        star_colors = np.array([[255, 203, 132],[255, 243, 233],[155, 176, 255]]) #default colors, reddish for small ones, yellow-white for mid sized and blue for large
    colors = np.int_([np.interp(np.log10(mass_in_msun),[-1,0,1],star_colors[:,i]) for i in range(3)])
    return (colors[0],colors[1],colors[2])# if len(colors)==1 else colors)

def Star_Edge_Color(cmap):
    if cmap=='afmhot' or cmap=='inferno':
        return 'black'
    else:
        return 'white'


def MakeImage(i):
    global center_on_ID
    global limits
    global Tlimits
    global energy_limits
    global v_res
    
    if disable_multigrid:
        GridSurfaceDensity_func = GridSurfaceDensity
    else:
        GridSurfaceDensity_func = GridSurfaceDensityMultigrid
    
#    print(i)
    snapnum1=file_numbers[i]
    snapnum2=(file_numbers[min(i+1,len(filenames)-1)] if n_interp>1 else snapnum1)
    
    sink_IDs_to_center_on = np.array([center_on_ID]) #default, will not center
    if center_on_densest:
        #Find the IDs of the sinks with the densest gas nearby 
        sink_IDs_to_center_on=find_sink_in_densest_gas(snapnum1)
    if center_on_star:
        #Find the IDs of the most massive sinks
        id1s = np.int_(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum1))
        m1s = mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum1))
        sink_IDs_to_center_on=id1s[m1s.argsort()[-N_high:]] #choose the N_high most massive
    for sink_ID in sink_IDs_to_center_on:
        #Check if all relevant pickle files exist
        all_pickle_exist = True
        for k in range(n_interp):
            pickle_filename = "Sinkvis_snap%d_%d_%d_r%g_res%d_c%g_%g_%g_0_%d_%s"%(snapnum1,k,n_interp,r,res,center[0],center[1],center[2],sink_ID,arguments["--dir"])+rescale_text+slice_text+smooth_text+energy_v_scale_text+".pickle"
            if outputfolder:
                pickle_filename=outputfolder+'/'+pickle_filename
            all_pickle_exist = all_pickle_exist & os.path.exists(pickle_filename)
        if (all_pickle_exist and plot_T_map):
            #We have the files but we should check whether they have temperature data. If not we are redoing them
                infile = open(pickle_filename, 'rb') 
                temp = pickle.load(infile)
                infile.close()
                Tmap_present = np.max(np.abs(temp[3])) #it is zero if we saved an empty array
                all_pickle_exist = all_pickle_exist and Tmap_present
        if not all_pickle_exist:
            if remake_only:
                print(pickle_filename+" does not exist, returning...")
                return
            print("Loading snapshot data from "+filenames[i])
            #We don't have the data, must read it from snapshot
            #keylist=load_from_snapshot("keys",0,datafolder,snapnum1)
            numpart_total=load_from_snapshot("NumPart_Total",0,datafolder,snapnum1)
            if not numpart_total[sink_type] and (center_on_star or (sink_ID>0)): return
            if numpart_total[sink_type]:
                id1s, id2s = np.int_(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum1)), np.int_(load_from_snapshot("ParticleIDs",sink_type,datafolder,snapnum2))                
                unique, counts = np.unique(id2s, return_counts=True)
                doubles = unique[counts>1]
                id2s[np.in1d(id2s,doubles)]=-1
                x1s, x2s = length_unit*np.array(load_from_snapshot("Coordinates",sink_type,datafolder,snapnum1))[id1s.argsort()], length_unit*np.array(load_from_snapshot("Coordinates",sink_type,datafolder,snapnum2))[id2s.argsort()]
                x1s, x2s = CoordTransform(x1s), CoordTransform(x2s)
                m1s, m2s = mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum1))[id1s.argsort()], mass_unit*np.array(load_from_snapshot("Masses",sink_type,datafolder,snapnum2))[id2s.argsort()]
                v1s, v2s = velocity_unit*np.array(load_from_snapshot("Velocities",sink_type,datafolder,snapnum1))[id1s.argsort()], velocity_unit*np.array(load_from_snapshot("Velocities",sink_type,datafolder,snapnum2))[id2s.argsort()]
                v1s, v2s = CoordTransform(v1s), CoordTransform(v2s)
                # take only the particles that are in both snaps
                common_sink_ids = np.intersect1d(id1s,id2s)
                if slice_height:
                    star_center1 = np.zeros(3)
                    star_center2 = np.zeros(3)
                    if sink_ID:
                        star_center1 = np.squeeze(x1s[id1s==sink_ID]-boxsize/2)
                        star_center2 = np.squeeze(x2s[id2s==sink_ID]-boxsize/2)
                    #Find which particles are within the slice in at least on snapshot and keep those only
                    dxs = np.abs(x1s-star_center1-center-boxsize/2)
                    ids_in_slice1 = id1s[(dxs[:,2]<=slice_height)]
                    dxs = np.abs(x2s-star_center2-center-boxsize/2)
                    ids_in_slice2 = id2s[(dxs[:,2]<=slice_height)]
                    common_sink_ids = np.intersect1d(common_sink_ids,np.union1d(ids_in_slice1,ids_in_slice2))
                idx1 = np.in1d(np.sort(id1s),common_sink_ids)
                idx2 = np.in1d(np.sort(id2s),common_sink_ids)
                x1s = x1s[idx1]; m1s = m1s[idx1]; v1s = v1s[idx1];
                x2s = x2s[idx2]; m2s = m2s[idx2]; v2s = v2s[idx2];
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
                id1, id2 = np.int_(load_from_snapshot("ParticleIDs",0,datafolder,snapnum1)), np.int_(load_from_snapshot("ParticleIDs",0,datafolder,snapnum2))
                wind_idx1 = np.in1d(id1, wind_ids)
#                print(np.sum(id1==wind_ids[0]), np.sum(id1==wind_ids[1]), id1.min(), id1.max())
                if np.any(wind_idx1):
                    progenitor_ids = np.int_(load_from_snapshot("ParticleIDGenerationNumber",0,datafolder,snapnum1))[wind_idx1]
                    child_ids = np.int_(load_from_snapshot("ParticleChildIDsNumber",0,datafolder,snapnum1))[wind_idx1]                    
                    wind_particle_ids = -((progenitor_ids << 16) + child_ids) # bit-shift the progenitor ID outside the plausible range for particle count, then add child ids to get a unique new id
                    id1[wind_idx1] = wind_particle_ids
#                    print(np.sum(id1==wind_ids[0]), np.sum(id1==wind_ids[1]), id1.min(), id1.max())
#                    print(len(np.unique(id1)), len(id1))
                    
                wind_idx2 = np.in1d(id2, wind_ids)
                if np.any(wind_idx2):
                    progenitor_ids = np.int_(load_from_snapshot("ParticleIDGenerationNumber",0,datafolder,snapnum2))[wind_idx2]
                    child_ids = np.int_(load_from_snapshot("ParticleChildIDsNumber",0,datafolder,snapnum2))[wind_idx2]
                    wind_particle_ids = -((progenitor_ids << 16) + child_ids) # bit-shift the progenitor ID outside the plausible range for particle count, then add child ids to get a unique new id
                    id2[wind_idx2] = wind_particle_ids
#                    print(np.sum(id2==wind_ids[0]), np.sum(id2==wind_ids[1]), id2.min(), id2.max())
#                    print(len(np.unique(id2)), len(id2))

                unique, counts = np.unique(id2, return_counts=True)
                doubles = unique[counts>1]

                id2[np.in1d(id2,doubles)]=-1

                id1_order, id2_order = id1.argsort(), id2.argsort()
                x1, x2 = length_unit*np.array(load_from_snapshot("Coordinates",0,datafolder,snapnum1))[id1_order], length_unit*np.array(load_from_snapshot("Coordinates",0,datafolder,snapnum2))[id2_order]
                x1, x2 = CoordTransform(x1), CoordTransform(x2)
                if not galunits:
                    x1 -= boxsize/2 + center
                    x2 -= boxsize/2 + center
                v1, v2 = velocity_unit*np.array(load_from_snapshot("Velocities",0,datafolder,snapnum1))[id1_order], velocity_unit*np.array(load_from_snapshot("Velocities",0,datafolder,snapnum2))[id2_order]
                v1, v2 = CoordTransform(v1), CoordTransform(v2)
                u1, u2 = np.array(load_from_snapshot("InternalEnergy",0,datafolder,snapnum1))[id1_order], np.array(load_from_snapshot("InternalEnergy",0,datafolder,snapnum2))[id2_order]
                h1, h2 = length_unit*np.array(load_from_snapshot("SmoothingLength",0,datafolder,snapnum1))[id1_order], length_unit*np.array(load_from_snapshot("SmoothingLength",0,datafolder,snapnum2))[id2_order]
                m1, m2 = mass_unit*np.array(load_from_snapshot("Masses",0,datafolder,snapnum1))[id1_order], mass_unit*np.array(load_from_snapshot("Masses",0,datafolder,snapnum2))[id2_order]
                # take only the particles that are in both snaps
                common_ids = np.intersect1d(id1,id2)
                if slice_height:
                    if not sink_ID:
                        star_center1 = np.zeros(3)
                        star_center2 = np.zeros(3)
                    #Find which particles are within the slice in at least on snapshot and keep those only
                    max_hsml_dist=5
                    dx = np.abs(x1-star_center1)-max_hsml_dist*h1[:,None]
                    ids_in_slice1 = id1[(dx[:,2]<=slice_height)]
                    dx = np.abs(x2-star_center2)-max_hsml_dist*h2[:,None]
                    ids_in_slice2 = id2[(dx[:,2]<=slice_height)]
                    common_ids = np.intersect1d(common_ids,np.union1d(ids_in_slice1,ids_in_slice2))
                    ids_in_slice1=0; ids_in_slice2=0; dx=0 #unload
                idx1 = np.in1d(np.sort(id1),common_ids)
                idx2 = np.in1d(np.sort(id2),common_ids)
                x1 = x1[idx1]; u1 = u1[idx1]; h1 = h1[idx1]*rescale_hsml; m1 = m1[idx1]; id1 = np.sort(id1)[idx1]; v1 = v1[idx1];
                x2 = x2[idx2]; u2 = u2[idx2]; h2 = h2[idx2]*rescale_hsml; m2 = m2[idx2]; id2 = np.sort(id2)[idx2]; v2 = v2[idx2];
                m = m2 # mass to actually use in render
                if highlight_wind != 1:
                    m[id2 < 0] *= highlight_wind
                
                # unload stuff to save memory
                idx1=0; idx2=0; id1=0; id2=0;

                
            time = load_from_snapshot("Time",0,datafolder,snapnum1)
        for k in range(n_interp):
            if (snapnum1!=snapnum2): #this part is to avoid creating pickle files for interpolating frames for the last snapshot
                k_in_filename = k
            else:
                k_in_filename = 0
            pickle_filename = "Sinkvis_snap%d_%d_%d_r%g_res%d_c%g_%g_%g_0_%d_%s"%(snapnum1,k_in_filename,n_interp,r,res,center[0],center[1],center[2],sink_ID,arguments["--dir"])+rescale_text+slice_text+smooth_text+energy_v_scale_text+".pickle"
            if outputfolder:
                pickle_filename=outputfolder+'/'+pickle_filename
            if not os.path.exists(pickle_filename):
                if numpart_total[sink_type]:
                    x_star = float(k)/n_interp * x2s + (n_interp-float(k))/n_interp * x1s
                    v_star = float(k)/n_interp * v2s + (n_interp-float(k))/n_interp * v1s
                else:
                    x_star = []; m_star = []; v_star = [];
                star_center = np.zeros(3)
                star_v_center = np.zeros(3)
                if sink_ID:
                    star_center = np.squeeze(x_star[common_sink_ids==sink_ID,:]-boxsize/2)
                    star_v_center = np.squeeze(v_star[common_sink_ids==sink_ID,:])
                    if smooth_center:
                        #Try to get more sink data and use it to smooth
                        star_center_coords=[]; snap_vals=[];
                        for snum in (snapnum1+np.arange(-smooth_center,smooth_center)):
                            if check_if_filename_exists(datafolder,snum)[0] != 'NULL': #snap exists
                                ids_temp = np.array(load_from_snapshot("ParticleIDs",sink_type,datafolder,snum))
                                if np.any(ids_temp==sink_ID): #the sink we want to center on is present
                                    xs_temp = length_unit*np.array(load_from_snapshot("Coordinates",sink_type,datafolder,snum))
                                    star_center_temp = np.squeeze(xs_temp[ids_temp==sink_ID,:]-boxsize/2)
                                    star_center_coords.append(star_center_temp)
                                    snap_vals.append(snum)
                        star_center_coords = np.array(star_center_coords); snap_vals = np.array(snap_vals)
                        #Let's fit a line to the motion
                        xfit = np.poly1d(np.polyfit(snap_vals,star_center_coords[:,0],1))
                        yfit = np.poly1d(np.polyfit(snap_vals,star_center_coords[:,1],1))
                        zfit = np.poly1d(np.polyfit(snap_vals,star_center_coords[:,2],1))
                        #Let's estimate the new center coordinate
                        star_center_old = star_center+0
                        star_center[0] = xfit(snapnum1+float(k)/n_interp)
                        star_center[1] = yfit(snapnum1+float(k)/n_interp)
                        star_center[2] = zfit(snapnum1+float(k)/n_interp)
                        print("Smoothing changed centering from %g %g %g to %g %g %g"%(star_center_old[0],star_center_old[1],star_center_old[2],star_center[0],star_center[1],star_center[2]))
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
                    
                    v = float(k)/n_interp * v2 + (n_interp-float(k))/n_interp * v1
                    v -= star_v_center
                    

                    logu = float(k)/n_interp * np.log10(u2) + (n_interp-float(k))/n_interp * np.log10(u1)
                    u = (10**logu)/1.01e4 #converting to K

                    h = float(k)/n_interp * h2 + (n_interp-float(k))/n_interp * h1
                    h = np.clip(h,L/res, 1e100)
                    sigma_gas = GridSurfaceDensity_func(m, x, h, star_center*0, L, res=res).T
                    if plot_T_map:
                        #Tmap_gas = GridAverage(u, x, h,star_center*0, L, res=res).T #should be similar to mass weighted average if particle masses roughly constant, also converting to K
                        #logTmap_gas = GridAverage(np.log10(u), x, h,star_center*0, L, res=res).T #average of log T so that it is not completely dominated by the warm ISM
                        weight_map = GridSurfaceDensity_func(np.ones(len(u)), x, h,star_center*0, L, res=res) #sum of weights
                        Tmap_gas = (GridSurfaceDensity_func(u, x, h,star_center*0, L, res=res)/weight_map).T #should be similar to mass weighted average if particle masses roughly constant, also converting to K
                        logTmap_gas = (GridSurfaceDensity_func(np.log10(u), x, h,star_center*0, L, res=res)/weight_map).T #average of log T so that it is not completely dominated by the warm ISM
                    else:
                        Tmap_gas = np.zeros((res,res))
                        logTmap_gas = np.zeros((res,res))
                    
                    v_field = np.zeros( (res,res,2) )
                    if plot_v_map:
                        weight_map = GridSurfaceDensity_func(np.ones(len(v[:,0])), x, h,star_center*0, L, res=res) #sum of weights
                        v_field[:,:,0] = (GridSurfaceDensity_func(v[:,0], x, h,star_center*0, L, res=res)/weight_map).T
                        v_field[:,:,1] = (GridSurfaceDensity_func(v[:,1], x, h,star_center*0, L, res=res)/weight_map).T
                    if plot_cool_map:
                        sigma_1D = GridSurfaceDensity_func(m * v[:,2]**2, x, h,star_center*0, L, res=res).T/sigma_gas
                        v_avg = GridSurfaceDensity_func(m * v[:,2], x, h,star_center*0, L, res=res).T/sigma_gas
                        sigma_1D = np.sqrt(sigma_1D - v_avg**2) / 1e3
                    else:
                        sigma_1D = np.zeros((res,res))
                        
                    if plot_energy_map:
                        kin_energy_weighted = m*(1.0+np.sum(v**2,axis=1)/(energy_v_scale**2))
                        energy_map_gas = GridSurfaceDensity_func(kin_energy_weighted, x, h, star_center*0, L, res=res).T
                    else:
                        energy_map_gas = np.zeros((res,res))
                else:
                    sigma_gas = np.zeros((res,res))
                    Tmap_gas = np.zeros((res,res))
                    logTmap_gas = np.zeros((res,res))
                    energy_map_gas = np.zeros((res,res))
                #Save data
                if not no_pickle:
                    print("Saving "+pickle_filename)
                    outfile = open(pickle_filename, 'wb') 
                    pickle.dump([x_star,m_star,sigma_gas,Tmap_gas,logTmap_gas,time,numpart_total, star_center,v_field,energy_map_gas, sigma_1D], outfile)
                    outfile.close()
            else:
                #Load data from pickle file
                print("Loading "+pickle_filename)
                infile = open(pickle_filename, 'rb') 
                temp = pickle.load(infile)
                infile.close()
                x_star = temp[0]; m_star = temp[1]; sigma_gas = temp[2]; Tmap_gas = temp[3]; logTmap_gas = temp[4]; time = temp[5]; numpart_total = temp[6];
                star_center = temp[7]
                if (len(temp)>=9):
                    v_field = temp[8]
                else:
                    v_field = np.zeros( (res,res,2) )
                if (len(temp)>=10):
                    energy_map_gas = temp[9]
                else:
                    energy_map_gas = np.zeros((res,res))
                if len(temp)>= 11:
                    sigma_1D = temp[10]
                else:
                    sigma_1D = np.zeros((res,res))
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
            fgas = np.flipud(fgas)
            data = plt.get_cmap(cmap)(fgas)
            data = np.clip(data,0,1)
            if plot_T_map:
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
                
                
            if plot_energy_map:
                #Adjust energy_limits if not set
                if ((energy_limits[0]==0) or (energy_limits[1]==0)):
                    energy_limits = limits
#                    energy_limits[1]=np.percentile(energy_map_gas,99)
#                    energy_limits[0]=np.min([energy_limits[1]*1e-2,np.max([energy_limits[1]*1e-4,np.percentile(energy_map_gas,5)])])
                    print("Using energy limits of %g and %g"%(energy_limits[0],energy_limits[1]))
                #Gas temperature map
                fegas = (np.log10(energy_map_gas)-np.log10(energy_limits[0]))/np.log10(energy_limits[1]/energy_limits[0])
                fegas = np.clip(fegas,0,1)
                energy_data = fegas[:,:,np.newaxis]*plt.get_cmap(ecmap)(fegas)[:,:,:3] 
                energy_data = np.clip(energy_data,0,1)

            if plot_cool_map:
                fgas = (np.log10(sigma_gas)-np.log10(limits[0]))/np.log10(limits[1]/limits[0])
#                fgas = np.clip(fgas,0,1)

                ls = LightSource(azdeg=315, altdeg=45)
                #lightness = ls.hillshade(z, vert_exag=4)
                mapcolor = plt.get_cmap(cool_cmap)(np.log10(sigma_1D/0.1)/2)
                cool_data = ls.blend_hsv(mapcolor[:,:,:3], fgas[:,:,None])
                cool_data = np.flipud(cool_data)
                
                
            local_name_addition = name_addition
            if sink_ID and (len(sink_IDs_to_center_on)>1):
                local_name_addition = '_%d'%(sink_ID) + local_name_addition
            file_number = file_numbers[i]            
            filename = "SurfaceDensity%s_%s.%s.png"%(local_name_addition,str(file_number).zfill(4),k)
            Tfilename = "Temperature%s_%s.%s.png"%(local_name_addition,str(file_number).zfill(4),k)
            efilename = "KineticEnergy%s_%s.%s.png"%(local_name_addition,str(file_number).zfill(4),k)
            logTfilename = "LogTemperature%s_%s.%s.png"%(local_name_addition,str(file_number).zfill(4),k)
            coolfilename = "cool_%s_%s.%s.png"%(local_name_addition,str(file_number).zfill(4),k)
            if outputfolder:
                filename=outputfolder+'/'+filename
                Tfilename=outputfolder+'/'+Tfilename
                efilename=outputfolder+'/'+efilename
                logTfilename=outputfolder+'/'+logTfilename
                coolfilename=outputfolder+'/'+coolfilename
            plt.imsave(filename, data) #f.split("snapshot_")[1].split(".hdf5")[0], map)
            print(filename)
            flist = [filename]
            if plot_T_map:
                plt.imsave(Tfilename, Tdata) #f.split("snapshot_")[1].split(".hdf5")[0], map)
                print(Tfilename)
                flist.append(Tfilename)
                plt.imsave(logTfilename, logTdata) #f.split("snapshot_")[1].split(".hdf5")[0], map)
                print(logTfilename)
                flist.append(logTfilename)
            if plot_energy_map:
                plt.imsave(efilename, energy_data) #f.split("snapshot_")[1].split(".hdf5")[0], map)
                print(efilename)
                flist.append(efilename)
            if plot_cool_map:
                plt.imsave(coolfilename, cool_data)
                print(coolfilename)
                flist.append(coolfilename)
            for fname in flist:
                gridres=res
                #Adding velocity field  here for some reason messes up the timestamp and the scale
                #Add labels and scale
                F = Image.open(fname)
                gridres = F.size[0]
                draw = ImageDraw.Draw(F)
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
                    pen = aggdraw.Pen(Star_Edge_Color(cmap),1) #gridres/800
                    for j in np.arange(len(x_star))[m_star>0]:
                        X = x_star[j] - star_center
                        ms = m_star[j]
                        star_size = gridres*sink_relscale * (np.log10(ms/sink_scale) + 1)
                        star_size = max(1,star_size)
                        p = aggdraw.Brush(StarColor(ms,cmap))
                        X -= boxsize/2 + center
                        norm_coords = (X[:2]+r)/(2*r)*gridres
                        #Pillow puts the origin in th top left corner, so we need to flip the y axis
                        norm_coords[1] = gridres - norm_coords[1]
                        coords = np.concatenate([norm_coords-star_size, norm_coords+star_size])
                        d.ellipse(coords, pen, p)#, fill=(155, 176, 255))
                    d.flush()
                F.save(fname)
                F.close()
                #Add velocity field
                if plot_v_map:
                    if v_res>res:
                        print("v_res too high, resetting to %d"%(res))
                        v_res=res
                    xlim = [boxsize/2.0+center[0]+star_center[0]-r,boxsize/2.0+center[0]+star_center[0]+r]
                    ylim = [boxsize/2.0+center[1]+star_center[1]-r,boxsize/2.0+center[1]+star_center[1]+r]
                    data = plt.imread(fname)
                    fig, ax = plt.subplots()
                    ax.imshow( data, extent=(xlim[0],xlim[1],ylim[0],ylim[1]) )
                    if not ('vx_field' in locals()):
                        #quiver_scale=v_res/4*np.mean(np.linalg.norm(v_field,axis=2))
                        quiver_scale=v_res/4*velocity_scale
                        x = np.linspace(xlim[0],xlim[1],num=v_res)
                        y = np.linspace(ylim[0],ylim[1],num=v_res)
                        #Reduce v_field resolution
                        vx_smoothed = gaussian_filter(v_field[:,:,0], sigma=res/v_res)
                        vy_smoothed = gaussian_filter(v_field[:,:,1], sigma=res/v_res)
                        #Interolate v_field
                        vx_interpolfunc = interp2d(np.arange(res)/(res-1), np.arange(res)/(res-1), vx_smoothed )
                        vy_interpolfunc = interp2d(np.arange(res)/(res-1), np.arange(res)/(res-1), vy_smoothed )
                        vx_field = vx_interpolfunc( np.arange(v_res)/(v_res-1), np.arange(v_res)/(v_res-1) )
                        vy_field = vy_interpolfunc( np.arange(v_res)/(v_res-1), np.arange(v_res)/(v_res-1) )
                        #Rescale v_field
                        v_min_scale = 0.3 * (8/v_res) #prefactor times the space between velocity grid points
                        vx_field = vx_field/quiver_scale; vy_field = vy_field/quiver_scale
                        #Correct for too small arrows
                        vlength = np.sqrt(vx_field**2 + vy_field**2); 
                        vlength_corrections = np.clip(v_min_scale/vlength,1.0,None)
                        vx_field = vx_field*vlength_corrections; vy_field = vy_field*vlength_corrections
                        #Correction to align the arrows and the background
                        vx_field = np.fliplr(-vx_field); vy_field = np.fliplr(vy_field)
                    ax.quiver(x,y,vx_field,vy_field,color=arrow_color,scale=1.0,scale_units='inches',units='xy',angles='xy')
                    ax.axis('off')
                    fig.set_size_inches(8, 8)
                    fig.savefig(fname,dpi=int(gridres/8))
                    plt.close(fig)
                    
                if draw_axes:
                    xlim = [boxsize/2.0+center[0]+star_center[0]-r,boxsize/2.0+center[0]+star_center[0]+r]
                    ylim = [boxsize/2.0+center[1]+star_center[1]-r,boxsize/2.0+center[1]+star_center[1]+r]
                    data = plt.imread(fname)
                    fig, ax = plt.subplots()
                    ax.imshow( data, extent=(xlim[0],xlim[1],ylim[0],ylim[1]) )
                    axes_dirs = np.roll(['X','Y','Z'], {'z': 0, 'y': 1, 'x': 2}[arguments["--dir"]])
                    ax.set_xlabel(axes_dirs[0]+" [pc]")
                    ax.set_ylabel(axes_dirs[1]+" [pc]")
                    fig.set_size_inches(6, 6)
                    #plt.figure(num=fig.number, figsize=(1.3*gridres/150, 1.2*gridres/150), dpi=550)
                    fig.savefig(fname,dpi=int(gridres/5))
                    plt.close(fig)
            

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
    open(framefile,'w').write('\n'.join(["file '%s'"%os.path.basename(f) for f in filenames]))
    os.system("ffmpeg -y -r " + str(fps) + " -f concat -i "+framefile+" -vb 20M -pix_fmt yuv420p  -q:v 0 -vcodec h264 -acodec aac -strict -2 -preset slow " + moviefilename + ".mp4")
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
        open(framefile,'w').write('\n'.join(["file '%s'"%f for f in filenames]))
        os.system("ffmpeg -y -r " + str(fps) + " -f concat -i frames.txt -vb 20M -pix_fmt yuv420p -q:v 0 -vcodec h264 -acodec aac -strict -2 -preset slow " + moviefilename + "_temp.mp4")
        #Erase files, leave movie only
        if only_movie:
            for i in filenames:
                os.remove(i)
        os.remove(framefile)
    #Plotting coolness
    if plot_cool_map:
        #Find files
        if outputfolder:
            filenames=natsorted(glob(outputfolder+'/'+'cool'+name_addition+'_????.?.png'))
        else:
            filenames=natsorted(glob('cool'+name_addition+'_????.?.png'))
        #Use ffmpeg to create movie
        open(framefile,'w').write('\n'.join(["file '%s'"%f for f in filenames]))
        os.system("ffmpeg -y -r " + str(fps) + " -f concat -i frames.txt -vb 20M -pix_fmt yuv420p -q:v 0 -vcodec h264 -acodec aac -strict -2 -preset slow " + moviefilename + "_cool.mp4")
        #Erase files, leave movie only
        if only_movie:
            for i in filenames:
                os.remove(i)
        os.remove(framefile)
            

def make_input(files=["snapshot_000.hdf5"], rmax=False, full_box=False, center=[0,0,0],limits=[0,0],Tlimits=[0,0],energy_limits=[0,0],\
                interp_fac=1, np=1,res=512,v_res=32, only_movie=False, fps=20, movie_name="sink_movie",dir='z',\
                center_on_star=0, N_high=1, Tcmap="inferno", cmap="viridis",ecmap="viridis", no_movie=True,make_movie=False, outputfolder="output",cool_cmap='same',\
                plot_T_map=True,plot_v_map=False,plot_energy_map=False, sink_scale=0.1, sink_relscale=0.0025, sink_type=5, galunits=False,name_addition="",center_on_ID=0,no_pickle=False, no_timestamp=False,slice_height=0,velocity_scale=1000,arrow_color='white',energy_v_scale=1000,\
                no_size_scale=False, center_on_densest=False, draw_axes=False, remake_only=False, rescale_hsml=1.0, smooth_center=False, highlight_wind=1.0,\
                disable_multigrid=False):
    if (not isinstance(files, list)):
        files=[files]
    arguments={
        "<files>": files,
        "--rmax": rmax,
        "--full_box": full_box,
        "--c": str(center[0])+","+str(center[1])+","+str(center[2]),
        "--dir": dir,
        "--limits": str(limits[0])+","+str(limits[1]),
        "--Tlimits": str(Tlimits[0])+","+str(Tlimits[1]),
        "--energy_limits": str(energy_limits[0])+","+str(energy_limits[1]),
        "--interp_fac": interp_fac,
        "--np": np,
        "--res": res,
        "--v_res": res,
        "--velocity_scale": velocity_scale,
        "--energy_v_scale": energy_v_scale,
        "--arrow_color": arrow_color,
        "--only_movie": only_movie,
        "--slice_height": slice_height,
        "--no_pickle": no_pickle,
        "--fps": fps,
        "--movie_name": movie_name,
        "--sink_type": str(sink_type),
        "--sink_scale": sink_scale,
        "--sink_relscale": sink_relscale,
        "--galunits": galunits,
        "--center_on_star": center_on_star,
        "--center_on_ID": center_on_ID,
        "--center_on_densest": center_on_densest,
        "--N_high": N_high,
        "--Tcmap": Tcmap,
        "--cmap": cmap,
        "--cool_cmap": cool_cmap,
        "--ecmap": ecmap,
        "--no_movie": no_movie,
        "--make_movie": make_movie,
        "--outputfolder": outputfolder,
        "--plot_T_map": plot_T_map,
        "--plot_cool_map": plot_cool_map,
        "--plot_energy_map": plot_energy_map,
        "--plot_v_map": plot_v_map,
        "--name_addition": name_addition,
        "--no_timestamp": no_timestamp,
        "--no_size_scale": no_size_scale,
        "--draw_axes": draw_axes,
        "--remake_only": remake_only,
        "--rescale_hsml": rescale_hsml,
        "--smooth_center": smooth_center,
        "--disable_multigrid": disable_multigrid,
        "--highlight_wind": highlight_wind
        }
    return arguments

def main(input):
    global arguments; arguments=input
    global filenames; filenames = natsorted(arguments["<files>"])
    if os.path.isdir(filenames[0]):
        namestring="snapdir"
    else:
        namestring="snapshot"
    global file_numbers; file_numbers = [int(re.search(namestring+'_\d*', f).group(0).replace(namestring+'_','')) for f in filenames]
    global datafolder; datafolder=(filenames[0].split(namestring+"_")[0])
    if not len(datafolder):
        datafolder="./"
    global boxsize; boxsize=load_from_snapshot("BoxSize",0,datafolder,file_numbers[0])
    full_box_flag = arguments["--full_box"]
    global r;
    if full_box_flag:
        r = boxsize/2.0
    elif arguments["--rmax"]:
        r = float(arguments["--rmax"])
    else:
        r = boxsize/10
    global name_addition; name_addition = arguments["--name_addition"] if arguments["--name_addition"] else ""
    global center; center = np.array([float(c) for c in arguments["--c"].split(',')])
    #Cycle coordinates to match projection
    center = np.roll(center, {'z': 0, 'y': 1, 'x': 2}[arguments["--dir"]], axis=0)
    global limits; limits = np.array([float(c) for c in arguments["--limits"].split(',')])
    global Tlimits; Tlimits = np.array([float(c) for c in arguments["--Tlimits"].split(',')])
    global energy_limits; energy_limits = np.array([float(c) for c in arguments["--energy_limits"].split(',')])
    global logTlimits; logTlimits = np.zeros(2)
    if Tlimits[0]:
        #used for log T plot
        logTlimits[:] = np.log10(Tlimits[:]) 
    global res; res = int(arguments["--res"])
    global v_res; v_res = int(arguments["--v_res"])
    nproc = int(arguments["--np"])
    global n_interp; n_interp = int(arguments["--interp_fac"])
    global cmap; cmap = arguments["--cmap"]
    global cool_cmap; cool_cmap = arguments["--cool_cmap"]
    if cool_cmap=='same':
        cool_cmap = cmap
    global ecmap; ecmap = arguments["--ecmap"]
    global Tcmap; Tcmap = arguments["--Tcmap"]
    global only_movie; only_movie = arguments["--only_movie"]
    global galunits; galunits = arguments["--galunits"]
    global no_movie
    #no_movie = arguments["--no_movie"]
    make_movie = arguments["--make_movie"]
    global slice_height; slice_height = float(arguments["--slice_height"])
    if make_movie:
        no_movie = False
    else:
        no_movie = True
    global disable_multigrid; disable_multigrid = arguments["--disable_multigrid"]
    global plot_T_map; plot_T_map = arguments["--plot_T_map"]
    global plot_energy_map; plot_energy_map = arguments["--plot_energy_map"]
    global plot_v_map; plot_v_map = arguments["--plot_v_map"]
    global plot_cool_map; plot_cool_map = arguments["--plot_cool_map"]
    global no_pickle; no_pickle = arguments["--no_pickle"]
    global remake_only; remake_only = arguments["--remake_only"]
    global no_timestamp; no_timestamp = arguments["--no_timestamp"]
    global draw_axes; draw_axes = arguments["--draw_axes"]
    global no_size_scale; no_size_scale = arguments["--no_size_scale"]
    global fps; fps = float(arguments["--fps"])
    global velocity_scale; velocity_scale = float(arguments["--velocity_scale"]) 
    global energy_v_scale; energy_v_scale = float(arguments["--energy_v_scale"])
    global energy_v_scale_text; energy_v_scale_text=''
    if plot_energy_map:
        energy_v_scale_text = '_v0%g'%(energy_v_scale)
    global rescale_hsml; rescale_hsml = float(arguments["--rescale_hsml"])
    global highlight_wind; highlight_wind = float(arguments["--highlight_wind"])
    global movie_name; movie_name = arguments["--movie_name"]
    global outputfolder; outputfolder = arguments["--outputfolder"]
    global sink_type; sink_type = int(arguments["--sink_type"])
    global sink_type_text; sink_type_text="PartType" + str(sink_type)
    global sink_scale; sink_scale = float(arguments["--sink_scale"])
    global sink_relscale; sink_relscale = float(arguments["--sink_relscale"])
    global center_on_star; center_on_star = 1 if arguments["--center_on_star"] else 0
    global center_on_ID; center_on_ID = int(arguments["--center_on_ID"]) if arguments["--center_on_ID"] else 0
    global smooth_center; smooth_center = int(arguments["--smooth_center"])
    global smooth_text; smooth_text=''
    if smooth_center:
        smooth_text = '_smoothed%d'%(smooth_center)
    global N_high; N_high = int(arguments["--N_high"])
    global center_on_densest; center_on_densest = 1 if arguments["--center_on_densest"] else 0
    global L; L = r*2
    global length_unit; length_unit = (1e3 if galunits else 1.) #in pc
    global velocity_unit; velocity_unit = (1e3 if galunits else 1.) #in m/s
    global arrow_color; arrow_color = arguments["--arrow_color"]
    global mass_unit; mass_unit = (1e10 if galunits else 1.) #in Msun
    global pc_to_AU; pc_to_AU = 206265.0
    #i = 0
    boxsize *= length_unit
    r *= length_unit
    L *= length_unit

    global font; font = ImageFont.truetype("LiberationSans-Regular.ttf", res//12) 
    
    #Only change the pickle filename if we rescale
    global rescale_text
    if (rescale_hsml!=1.0):
        rescale_text='_%g'%(rescale_hsml)
    else:
        rescale_text=''
    global slice_text; slice_text=''
    if slice_height:
        slice_text='_slice%g'%(slice_height)
    
    if outputfolder:
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)
   
   #Try to guess surface density limits for multiple snap runs (can't guess within MakeImage routine due to parallelization, also the first image is unlikely to be useful as it is often just the IC) so we will run the last one first, without parallelization
    if ( ( (limits[0]==0) or (Tlimits[0]==0 and plot_T_map) ) and (len(filenames) > 1) ):
        print("Surface density or temperature limits not set, running final snapshot first to guess the appropriate values.")
        MakeImage(len(filenames)-1)
        
    if nproc>1:
        Pool(nproc).map(MakeImage, (f for f in range(len(filenames))), chunksize=1)
    else:
        [MakeImage(i) for i in range(len(filenames))]
    if (len(filenames) > 1 and (not no_movie) ): 
        MakeMovie() # only make movie if plotting multiple files
        
if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
