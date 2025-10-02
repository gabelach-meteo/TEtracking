#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import cartopy.feature as cfeature
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
import time
import importlib
import stormcoupling as sc
import glob
from scipy import stats
import scipy
from geopy import distance
from scipy import interpolate
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import cm
import xlrd
import csv
import os
import math
import matplotlib.colors
import metpy
from metpy.interpolate import interpolate_to_isosurface
from metpy.calc import dewpoint_from_specific_humidity
from metpy.calc import equivalent_potential_temperature
from matplotlib.ticker import LogFormatterSciNotation
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from metpy.units import units 
import metpy.calc as mpcalc
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import imageio.v2 as imageio
from PIL import Image, ImageOps
importlib.reload(sc)
#### takes output data from StitchNodes (tracks) and organizes it into datafram with individual track IDs

def parse_te_tracks(filename):
    tracks = []
    trackid = -1
    with open(filename) as f:
        current_id = None
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "start":
                length = int(parts[1])  # track ID
                trackid = trackid+1
            else:
                lon, lat, msl, year, month, day, hour = parts[-7:]
                tracks.append({
                    "id": trackid,
                    "length": length,
                    "lon": float(lon),
                    "lat": float(lat),
                    "msl": float(msl),
                    "time": pd.Timestamp(int(year), int(month), int(day), int(hour))
                })
    return pd.DataFrame(tracks)

#### locates points from track dataframe at specific time

def findDataAtTime(year,month,day,hour,var,tracks,data):
    timestamp = f"{year}-{month}-{day}T{hour}:00"
    indices_time1 = tracks[tracks['time'] == timestamp]
    data_time1 = data[var].sel(valid_time=timestamp)
    return indices_time1, data_time1, timestamp

#### locates points from track dataframe at specific time (returns both cyclones (data1) and anticyclones (data2))


def findDataAtTime2(year,month,day,hour,var,tracks1,tracks2,data):
    timestamp = f"{year}-{month}-{day}T{hour}:00"
    indices_time1 = tracks1[tracks1['time'] == timestamp]
    indices_time2 = tracks2[tracks2['time'] == timestamp]
    data_time1 = data[var].sel(valid_time=timestamp)
    return indices_time1, indices_time2, data_time1, timestamp

#### locates points from track dataframe at specific time (returns both cyclones (data1) and anticyclones (data2)) but takes timestamp object


def findDataAtTime2_timestamp(timestamp,var,tracks1,tracks2,data):
    indices_time1 = tracks1[tracks1['time'] == timestamp]
    indices_time2 = tracks2[tracks2['time'] == timestamp]
    data_time1 = data[var].sel(valid_time=timestamp)
    return indices_time1, indices_time2, data_time1, timestamp

## plots contour map of MSL data (takes output of findDataAtTime functions)

def plotMSLAtTime2(output_FindData, bounds=[-175, -30, 20, 75]):
    indices1 = output_FindData[0]
    indices2 = output_FindData[1]
    data1 = output_FindData[2]
    fig=plt.figure(figsize=(24,16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.stock_img()
    ax.coastlines(color='gray')
    ax.gridlines(draw_labels=True)
    ax.add_feature(
        cfeature.LAND, 
        facecolor='palegoldenrod',   # any Matplotlib color
        zorder=0                 
    )

    ax.add_feature(
        cfeature.OCEAN, 
        facecolor='lightblue',
        zorder=0
    )
    ax.add_feature(
        cfeature.LAKES, 
        facecolor='lightblue',
        zorder=0
    )
    ax.add_feature(
        cfeature.BORDERS, 
        facecolor='gray',
        zorder=0
    )
    ax.add_feature(
        cfeature.RIVERS, 
        facecolor='lightblue',
        zorder=0
    )
    # basic stats
    msl_hpa = (data1 / 100.).values
    lon = data1['longitude'].values
    lat = data1['latitude'].values
    vmin = float(np.nanmin(msl_hpa))
    vmax = float(np.nanmax(msl_hpa))

    
    coarse_step = 4.0    # coarse contour spacing (hPa)
    fine_step   = 4.0    # fine contour spacing near minima (hPa)
    fine_range  = 12.0   # create finer spacing for values within (vmin, vmin+fine_range)

    # make coarse levels (descending or ascending doesn't matter for list)
    coarse_levels = np.arange(np.floor(vmin - 0.5*coarse_step),
                              np.ceil(vmax + 0.5*coarse_step),
                              coarse_step)

    # make fine levels near minima
    fine_levels = np.arange(np.floor(vmin - 0.5*fine_step),
                            vmin + fine_range + 0.0001,
                            fine_step)

    # combine, remove duplicates and sort
    levels = np.unique(np.concatenate([coarse_levels, fine_levels]))
    levels = np.sort(levels)
    # Plot center of each track at time t
    for index in zip(indices1.lon, indices1.lat, indices1.msl):
        mslval = round(index[2]/100)
        ax.text(index[0], index[1], 'L', c='red', fontsize=15, ha='center', va='center', fontweight='bold', zorder=4)
        ax.text(index[0], index[1]-1.2, mslval, c='red', fontsize=10, ha='center', va='top', fontweight='bold', zorder=4)
    for index in zip(indices2.lon, indices2.lat, indices2.msl):
        mslval = round(index[2]/100)
        ax.text(index[0], index[1], 'H', c='blue', fontsize=15, ha='center', va='center', fontweight='bold', zorder=4)
        ax.text(index[0], index[1]-1.2, mslval, c='blue', fontsize=10, ha='center', va='top', fontweight='bold', zorder=4)
    contour = ax.contour(
        data1['longitude'],
        data1['latitude'],
        data1 / 100,  # convert from Pa to hPa 
        levels=levels,
        transform=ccrs.PlateCarree(),
        colors='black',
        linewidths=1,
        zorder=1
    )

    ax.clabel(contour, inline=True, fontsize=8, fmt="%.0f",zorder=3)
    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    ax.set_title(f'msl, time = {output_FindData[3]}')
    #plt.savefig('test.jpg', bbox_inches='tight')
    plt.savefig(f'plots/random/msl_{output_FindData[3]}.png', bbox_inches='tight')
    plt.show()
    

## same as previous but gradient plot for meteorological variables

def plotGRADAtTime2(output_FindData, cmap1, vmin1=None, vmax1=None):
    indices1 = output_FindData[0]
    indices2 = output_FindData[1]
    data1 = output_FindData[1]
    p = None
    if 'pressure_level' in list(data1.coords):
        first_p = data1.pressure_level.data[0]
        print(f'Pressure level data found, defaulting to {first_p}')
        data1 = data1.sel(pressure_level=first_p)
        p = first_p
    var = data1.name
    fig=plt.figure(figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(color='black')
    ax.gridlines(draw_labels=True)
    ax.add_feature(
        cfeature.LAND, 
        facecolor='palegoldenrod',
        zorder=0                 # draw behind contours
    )

    ax.add_feature(
        cfeature.OCEAN, 
        facecolor='lightblue',
        zorder=0
    )
    # Plot each cyclone track
    for index in zip(indices1.lon, indices1.lat, indices1.msl):
        mslval = round(index[2]/100)
        ax[0].text(index[0], index[1], 'L', c='red', fontsize=15, ha='center', va='center', fontweight='bold', zorder=4)
        ax[0].text(index[0], index[1]-1.2, mslval, c='red', fontsize=10, ha='center', va='top', fontweight='bold', zorder=4)
    for index in zip(indices2.lon, indices2.lat, indices2.msl):
        mslval = round(index[2]/100)
        ax[0].text(index[0], index[1], 'H', c='blue', fontsize=15, ha='center', va='center', fontweight='bold', zorder=4)
        ax[0].text(index[0], index[1]-1.2, mslval, c='blue', fontsize=10, ha='center', va='top', fontweight='bold', zorder=4)
    im = ax.pcolormesh(
        data1['longitude'],
        data1['latitude'],
        data1,  
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        linewidths=1,
        zorder=1,
        vmin=vmin1,
        vmax=vmax1
    )
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(f'{var}, {data1.units}')
    #ax.clabel(im, inline=True, fontsize=8, fmt="%.0f")
    ax.set_extent([-175, -30, 20, 75], crs=ccrs.PlateCarree())
    ax.set_title(f'{var}, time = {output_FindData[2]}')
    #plt.savefig('test.jpg', bbox_inches='tight')
    plt.savefig(f'plots/random/{var}{p}_{output_FindData[2]}.png', bbox_inches='tight')
    plt.show()
    
## plots two panel contour and gradient map of MSL & variable of choice data data (takes output of findDataAtTime functions for both msl and the variable)

def plotMSLAndGradAtTime_twopanel(output_FindData, output_FindData1, cmap1, vmin1=None, vmax1=None, bounds=[-175, -30, 20, 75], out='def', hspace=-0.4):
    indices1 = output_FindData[0]
    indices2 = output_FindData[1]
    data1 = output_FindData[2]
    data2 = output_FindData1[2]
    p = 'sfc'
    if 'pressure_level' in list(data2.coords):
        first_p = data2.pressure_level.data[0]
        print(f'Pressure level data found, defaulting to {first_p}')
        data2 = data2.sel(pressure_level=first_p)
        p = first_p
    var = data2.name
    fig, axes = plt.subplots(
    nrows=2, ncols=1,
    subplot_kw={'projection': ccrs.PlateCarree()},
    figsize=(18, 14)
    )
    axes[0].set_extent(bounds, crs=ccrs.PlateCarree())
    axes[1].set_extent(bounds, crs=ccrs.PlateCarree())
    #ax.stock_img()
    axes[0].coastlines(color='gray')
    axes[0].gridlines(draw_labels=True)
    axes[1].coastlines(color='gray')
    axes[1].gridlines(draw_labels=True)
    axes[0].add_feature(
        cfeature.LAND, 
        facecolor='palegoldenrod',   # any Matplotlib color
        zorder=0                 
    )
    axes[1].add_feature(
        cfeature.LAND, 
        facecolor='palegoldenrod',   # any Matplotlib color
        zorder=0                 
    )
    axes[0].add_feature(
        cfeature.OCEAN, 
        facecolor='lightblue',
        zorder=0
    )
    axes[0].add_feature(
        cfeature.LAKES, 
        facecolor='lightblue',
        zorder=0
    )
    axes[1].add_feature(
        cfeature.OCEAN, 
        facecolor='lightblue',
        zorder=0
    )
    axes[1].add_feature(
        cfeature.LAKES, 
        facecolor='lightblue',
        zorder=0
    )
    axes[0].add_feature(
        cfeature.BORDERS, 
        facecolor='gray',
        zorder=0
    )
    axes[1].add_feature(
        cfeature.BORDERS, 
        facecolor='gray',
        zorder=0
    )
    axes[0].add_feature(
        cfeature.RIVERS, 
        facecolor='lightblue',
        zorder=0
    )
    axes[1].add_feature(
        cfeature.RIVERS, 
        facecolor='lightblue',
        zorder=0
    )
    
    # basic stats
    msl_hpa = (data1 / 100.).values
    lon = data1['longitude'].values
    lat = data1['latitude'].values
    vmin = float(np.nanmin(msl_hpa))
    vmax = float(np.nanmax(msl_hpa))

    
    coarse_step = 4.0    # coarse contour spacing (hPa)
    fine_step   = 4.0    # fine contour spacing near minima (hPa)
    fine_range  = 12.0   # create finer spacing for values within (vmin, vmin+fine_range)

    # make coarse levels (descending or ascending doesn't matter for list)
    coarse_levels = np.arange(np.floor(vmin - 0.5*coarse_step),
                              np.ceil(vmax + 0.5*coarse_step),
                              coarse_step)

    # make fine levels near minima
    fine_levels = np.arange(np.floor(vmin - 0.5*fine_step),
                            vmin + fine_range + 0.0001,
                            fine_step)

    # combine, remove duplicates and sort
    levels = np.unique(np.concatenate([coarse_levels, fine_levels]))
    levels = np.sort(levels)
    # Plot center of each track at time t
    for index in zip(indices1.lon, indices1.lat, indices1.msl):
        mslval = round(index[2]/100)
        axes[0].text(index[0], index[1], 'L', c='red', fontsize=15, ha='center', va='center', fontweight='bold', zorder=4, clip_on=True)
        axes[0].text(index[0], index[1]-1.2, mslval, c='red', fontsize=10, ha='center', va='top', fontweight='bold', zorder=4, clip_on=True)
    for index in zip(indices2.lon, indices2.lat, indices2.msl):
        mslval = round(index[2]/100)
        axes[0].text(index[0], index[1], 'H', c='blue', fontsize=15, ha='center', va='center', fontweight='bold', zorder=4, clip_on=True)
        axes[0].text(index[0], index[1]-1.2, mslval, c='blue', fontsize=10, ha='center', va='top', fontweight='bold', zorder=4, clip_on=True)
    for index in zip(indices1.lon, indices1.lat, indices1.msl):
        mslval = round(index[2]/100)
        axes[1].text(index[0], index[1], 'L', c='red', fontsize=15, ha='center', va='center', fontweight='bold', zorder=4, clip_on=True)
        axes[1].text(index[0], index[1]-1.2, mslval, c='red', fontsize=10, ha='center', va='top', fontweight='bold', zorder=4, clip_on=True)
    for index in zip(indices2.lon, indices2.lat, indices2.msl):
        mslval = round(index[2]/100)
        axes[1].text(index[0], index[1], 'H', c='blue', fontsize=15, ha='center', va='center', fontweight='bold', zorder=4, clip_on=True)
        axes[1].text(index[0], index[1]-1.2, mslval, c='blue', fontsize=10, ha='center', va='top', fontweight='bold', zorder=4, clip_on=True)
    contour = axes[0].contour(
        data1['longitude'],
        data1['latitude'],
        data1 / 100,  # convert from Pa to hPa 
        levels=levels,
        transform=ccrs.PlateCarree(),
        colors='black',
        linewidths=1,
        zorder=1
    )
    im = axes[1].pcolormesh(
        data2['longitude'],
        data2['latitude'],
        data2,  
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        linewidths=1,
        zorder=1,
        vmin=vmin1,
        vmax=vmax1
    )
    axes[0].clabel(contour, inline=True, fontsize=8, fmt="%.0f",zorder=3)
    axes[0].set_title(f'msl, time = {output_FindData[3]}')
    #cbar = fig.colorbar(im, ax=axes[1], orientation='horizontal', pad=0.05, aspect=50, fraction=0.03)
    cb_ax = inset_axes(axes[1], width="100%", height="5%", loc='lower center', borderpad=-3.5)
    cbar = plt.colorbar(im, cax=cb_ax, orientation='horizontal')
    cbar.set_label(f'{var}_{p}, {data2.units}')
    #ax.clabel(im, inline=True, fontsize=8, fmt="%.0f")
    #axes[1].set_title(f'{var}, time = {output_FindData[3]}')
    #plt.savefig('test.jpg', bbox_inches='tight')
    plt.subplots_adjust(hspace=hspace)
    plt.savefig(f'plots/random/msl_{var}_2panel_{output_FindData[3]}_{out}.png', bbox_inches='tight')
    plt.show()

    
## creates and saves png frames using twopanel function starting at start and ending at end in 3 hour increments

def createGifFrames_twopanel(start, end, tracks_c, tracks_ac, data1, data2, var, cmap1, vmin1=None, vmax1=None, bounds=[-175, -30, 20, 75], out='def', hspace=-0.4):
    for time in pd.date_range(start=start, end=end, freq='3h'):
        print(time)
        output1 = findDataAtTime2_timestamp(time,'msl', tracks_c, tracks_ac, data1)
        output2 = findDataAtTime2_timestamp(time, var, tracks_c, tracks_ac, data2)
        plotMSLAndGradAtTime_twopanel(output1, output2, cmap1=cmap1, vmin1=vmin1, vmax1=vmax1, bounds=bounds, hspace=hspace, out=out)
        
        
## same as previous but for just msl

def createGifFrames_msl(start, end, tracks_c, tracks_ac, data1, bounds=[-175, -30, 20, 75]):
    for time in pd.date_range(start=start, end=end, freq='3h'):
        print(time)
        output1 = findDataAtTime2_timestamp(time,'msl', tracks_c, tracks_ac, data1)
        plotMSLAtTime2(output1, bounds=bounds)