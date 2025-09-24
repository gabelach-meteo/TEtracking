#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import pandas as pd
import matplotlib.patches as mpatches
import cartopy.feature as cfeature
import geopy as geo
from collections import Counter
import csv
from geopy import distance


# ver: 1.2
# 
# Changes from previous version:
# - added compatibility for new mcs dataset (including all of NH)
# 
# 
# ## Notes ##
# 
# The only functions which should be ran externally are createCoupledPairs and save_data, and mcsIndexToTrack (synIndexToTrack) if you wish to convert a global index to a dataset (dataframe). The rest of the functions are used internally. 
# 
# General comment: This module heavily relies on csv files which contain superficial data about MCS and Synoptic tracks. Note that if the file path to the original data changes, these functions will not run properly unless you manually change the path variables. 
# 
# The csv files contain custom global indices for each track in the format 'MCS-YYYY-XXXX' or 'SYN-YYYY-XXXX' where YYYY corresponds to the year and XXXX corresponds to a unique 4-digit index, resetting at 0000 for each year. It additionally contains the paths to the data files containing the tracks, as well as the start and end times of each tracks. The MCS csv file contains a local index which corresponds to the 'tracks' value in the dataset from which the file was created. These files were created iteratively by looking at every track in the data and individually creating a row for it. 
# 
# MCS tracks:
# 
# Feng, Z., Leung, L. R., Liu, N., Wang, J., Houze, R. A., Li, J., et al. (2021). A global high-resolution mesoscale convective system database using satellite-derived cloud tops, surface precipitation, and tracking. Journal of Geophysical Research: Atmospheres, 126, e2020JD034202. https://doi.org/10.1029/2020JD034202
# 
# Synoptic tracks:
# 
# Crawford, Alex D. et al. "Sensitivity of Northern Hemisphere Cyclone Detection and Tracking Results to Fine Spatial and Temporal Resolution using ERA5" vol. 149, no. 8, 2021, https://doi.org/10.1175/MWR-D-20-0417.1
# 
# 
# 
# Planned Changes:
# - Create alternate createCoupledPairs function which uses varying distances or alternatively use distance as an input and see how this affects data 
# - Parallelize createCoupledPairs function
# - add input for output path of save_data
# 
# Expected inputs: 
# 
# - Most functions take as input a pandas DataFrame object which represents a CSV file containing a list of tracks and track file paths as described previously
# 
# Expected outputs: 
# 
# - If you are just saving the data with save_data, there will be no output and the module will automatically save each found pair as an individual csv file containing information on the indices, lats, lons, times and distances of the coupled tracks. 
# - The output of createCoupledPairs is a raw array containing n rows (where n corresponds to how many coupled points were found in total) and 9 columns: 
#             0: synoptic global index
#             1: MCS global index
#             2: number of indices which were coupled 
#             3: synoptic time 
#             4: MCS time
#             5: synoptic latitude
#             6: synoptic longitude
#             7: MCS latitude
#             8: MCS longitude
#             9: distance between points. This also corresponds to the data which is saved in the final pairs' csv files. 

# ## Helper Functions ##

# In[2]:


mcs_tracks = pd.read_csv('/home/glach/projects/def-rfajber/shared/gabe-final-notebooks/MCS_Tracks_3.csv')
mcs_tuples = mcs_tracks.to_numpy(dtype=str)


# In[3]:


def createMcsDatetimeStart(mcs_csv):
    """
    Creates an array of start times for given MCS tracks. 
 
    ---------
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks in which to create start times for
 
    Returns
    ---------
    numpy.array
        Array of Timestamp start times for each MCS track.
    """
    mcs_datetime64_start = np.array([])
    for index in mcs_csv.Global_Index:
        mcs = mcsIndexToTrack(index)
        time = mcs.base_time.data[0] # 0th index corresponds to first base_time index in the track, i.e. the time of the first point of the track
        btime = pd.to_datetime(time)
        mcs_datetime64_start = np.append(mcs_datetime64_start, btime)
    return mcs_datetime64_start


# In[4]:


def createMcsDatetimeEnd(mcs_csv):
    """
    Creates an array of end times for given MCS tracks. 
 
    ---------
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks in which to create end times for
 
    Returns
    ---------
    numpy.array
        Array of Timestamp end times for each MCS track.
    """
    mcs_datetime64_end = np.array([])
    for index in mcs_csv.Global_Index:
        mcs = mcsIndexToTrack(index)
        ind = (~np.isnan(mcs.base_time.data)).cumsum().argmax()-1 
        # ind corresponds to the final non-NaN base_time index, i.e. the index of the last point in the track
        # this is necessary because MCS tracks have a standard length of 200 indices when in reality this is unrealistic, so after the track ends, the indices are filled with NaN which cannot be worked with
        time = mcs.base_time.data[ind]
        btime = pd.to_datetime(time)
        mcs_datetime64_end = np.append(mcs_datetime64_end, btime)
    return mcs_datetime64_end


# In[5]:


## note : Not used in later versions. Replaced by mcsIndexToTrack or simply xr.open_dataset()
mes_dir='/home/glach/projects/def-rfajber/shared/tracks/Feng2020JGR_data/data/nam'
def importMCS(year, mes_idir = mes_dir):
    """
    Imports a North America MCS track dataset for a given year. 
 
    ---------
    year : int
        Year in which dataset is imported for
    mes_idir : str
        '/home/glach/projects/def-rfajber/shared/tracks/Feng2020JGR_data/data/nam'
 
    Returns
    ---------
    xarray.Dataset
        Full dataset of MCS tracks for the year. 
    """
    mcs_ds=xr.open_dataset(f'{mes_idir}/robust_mcs_tracks_extc_{year}0101_{year}1231.nc')
    mcs_ds['datetimestart'] = (('tracks'), createMcsDatetimeStart(mcs_ds))
    mcs_ds['datetimeend'] = (('tracks'), createMcsDatetimeEnd(mcs_ds))
    return mcs_ds


# In[6]:


def synToTimestamp(df, i_time):
    """
    Converts synoptic track data to Timestamp object.
 
    ---------
    df : pandas CSV DataFrame
        DataFrame of single synoptic track
    i_time : int
        index of row to return time from 
 
    Returns
    ---------
    pandas.Timestamp
        Timestamp object of given synoptic track index. 
    """
    year = str(df.year.iloc[i_time])
    month = str(df.month.iloc[i_time]).zfill(2)
    day = str(df.day.iloc[i_time]).zfill(2)
    hour = str(df.hour.iloc[i_time]).zfill(2)+':00:00'
    time = f'{year}-{month}-{day} {hour}'
    time_timestamp = pd.to_datetime(time, 
                                     format = '%Y-%m-%d %H:%M:%S')
    return time_timestamp


# In[7]:


def createSynDatetimeStart(syn_csv):
    """
    Creates an array of start times for given synoptic tracks. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks in which to create start times for
 
    Returns
    ---------
    numpy.array
        Array of Timestamp start times for each synoptic track.
    """
    syn_datetime64_start = np.array([])
    for index in syn_csv.Global_Index:
        df = synIndexToTrack(index)
        btime = synToTimestamp(df, 0) # 0th index corresponds to first index in the track, i.e. the row of the first point of the track
        syn_datetime64_start = np.append(syn_datetime64_start, btime)
    return syn_datetime64_start


# In[8]:


def createSynDatetimeEnd(syn_csv):
    """
    Creates an array of end times for given synoptic tracks. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks in which to create end times for
 
    Returns
    ---------
    numpy.array
        Array of Timestamp end times for each synoptic track.
    """
    syn_datetime64_end = np.array([])
    for index in syn_csv.Global_Index:
        df = synIndexToTrack(index)
        btime = synToTimestamp(df, -1) # -1th index corresponds to last index in the track, i.e. the row of the last point of the track
        syn_datetime64_end = np.append(syn_datetime64_end, btime)
    return syn_datetime64_end


# In[9]:


def createPairs1(syn_csv, mcs_csv, pairs_distance):
    """
    Creates an array of initial possible pairs of synoptic tracks and MCS tracks. Condition: MCS end is after synoptic start, MCS end is before synoptic end, first MCS & synoptic point within distance. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks to compare
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks to compare
    distance : integer
        distance (km) to which the initial points should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
 
    Returns
    ---------
    numpy.array
        2-dimensional array in form [synoptic_track, mcs_track] of the global index (from CSV file) of each potential pair. 
    """
    pairs_1 = np.array([0,0], dtype=str)
    ## all of the 0 indices in this function are used to obtain the 0th (first) lat/lon of the respective tracks in order to test their initial distance and filter based on such 
    first_mcs = mcs_csv.index[0]
    first_syn = syn_csv.index[0]
    M_e_ARR = pd.to_datetime(mcs_csv.End)
    for i, syn_index in enumerate(syn_csv.Global_Index):
        syn=synIndexToTrack(syn_index)
        C_s = pd.to_datetime(syn_csv.Start[i + first_syn])
        C_e = pd.to_datetime(syn_csv.End[i + first_syn])
        syn_lat_0 = syn.lat.to_numpy()[0]
        syn_lon_0 = syn.lon.to_numpy()[0]
        syn_coords_0 = (syn_lat_0, syn_lon_0)
        matchingInds = np.where(np.logical_and(M_e_ARR>C_s, M_e_ARR<=C_e))[0]
        for ind in matchingInds:
            mcs_index = mcs_csv.Global_Index[ind + first_mcs]
            mcs = mcsIndexToTrack(mcs_index)
            mcs_lat_0 = (mcs.meanlat.data[0])
            mcs_lon_0 = (mcs.meanlon.data[0])
            mcs_coords_0 = (mcs_lat_0, mcs_lon_0)
            dist = distance.distance(syn_coords_0, mcs_coords_0).km
            if(dist < pairs_distance):
                pairs_1 = np.vstack([pairs_1, np.array([syn_index, mcs_index])])
    pairs_1 = pairs_1[1:]
    return pairs_1


# In[11]:


def createPairs1_1(syn_array, mcs_array=mcs_tuples, pairs_distance=4000):
    ## meant for array inputs
    """
    Creates an array of initial possible pairs of synoptic tracks and MCS tracks. Condition: MCS end is after synoptic start, MCS end is before synoptic end, first MCS & synoptic point within distance. 
 
    ---------
    syn_array : 1d numpy array
        1d array of a synoptic track (same format as csv)
    mcs_array : 2d numpy array
        2d array of all MCS tracks to test against synoptic track
    distance : integer
        distance (km) to which the initial points should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
 
    Returns
    ---------
    numpy.array
        2-dimensional array in form [synoptic_track, mcs_track] of the global index (from CSV file) of each potential pair. 
    """
    pairs_1 = np.array([0,0], dtype=str)
    ## all of the 0 indices in this function are used to obtain the 0th (first) lat/lon of the respective tracks in order to test their initial distance and filter based on such 
    M_e_ARR = pd.to_datetime(mcs_array[:,4])
    syn_index = syn_array[0] # global index
    syn=synIndexToTrack(syn_index)
    C_s = pd.to_datetime(syn_array[2]) # start time
    C_e = pd.to_datetime(syn_array[3]) # end time
    syn_lat_0 = syn.lat.to_numpy()[0]
    syn_lon_0 = syn.lon.to_numpy()[0]
    syn_coords_0 = (syn_lat_0, syn_lon_0)
    matchingInds = np.where(np.logical_and(M_e_ARR>C_s, M_e_ARR<=C_e))[0]
    for ind in matchingInds:
        mcs_index = mcs_array[ind][0]
        mcs = mcsIndexToTrack(mcs_index)
        mcs_lat_0 = (mcs.meanlat.data[0])
        mcs_lon_0 = (mcs.meanlon.data[0])
        mcs_coords_0 = (mcs_lat_0, mcs_lon_0)
        dist = distance.distance(syn_coords_0, mcs_coords_0).km
        if(dist < pairs_distance):
            pairs_1 = np.vstack([pairs_1, np.array([syn_index, mcs_index])])
    pairs_1 = pairs_1[1:]
    return pairs_1


# In[12]:


# In[12]:


def createPairs2(syn_csv, mcs_csv, pairs_distance):
    """
    Creates an array of initial possible pairs of synoptic tracks and MCS tracks. Condition: MCS end is after synoptic start, MCS end is before synoptic end, first MCS & synoptic point within distance. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks to compare
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks to compare
    distance : integer
        distance (km) to which the initial points should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
 
    Returns
    ---------
    numpy.array
        2-dimensional array in form [synoptic_track, mcs_track] of the global index (from CSV file) of each potential pair. 
    """
    pairs_2 = np.array([0,0], dtype=str)
    ## all of the 0 indices in this function are used to obtain the 0th (first) lat/lon of the respective tracks in order to test their initial distance and filter based on such 
    first_mcs = mcs_csv.index[0]
    first_syn = syn_csv.index[0]
    M_s_ARR = pd.to_datetime(mcs_csv.Start)
    for i, syn_index in enumerate(syn_csv.Global_Index):
        syn=synIndexToTrack(syn_index)
        C_s = pd.to_datetime(syn_csv.Start[i + first_syn])
        C_e = pd.to_datetime(syn_csv.End[i + first_syn])
        syn_lat_0 = syn.lat.to_numpy()[0]
        syn_lon_0 = syn.lon.to_numpy()[0]
        syn_coords_0 = (syn_lat_0, syn_lon_0)
        matchingInds = np.where(np.logical_and(M_s_ARR<C_e, M_s_ARR>=C_s))[0]
        for ind in matchingInds:
            mcs_index = mcs_csv.Global_Index[ind + first_mcs]
            mcs = mcsIndexToTrack(mcs_index)
            mcs_lat_0 = (mcs.meanlat.data[0])
            mcs_lon_0 = (mcs.meanlon.data[0])
            mcs_coords_0 = (mcs_lat_0, mcs_lon_0)
            dist = distance.distance(syn_coords_0, mcs_coords_0).km
            if(dist < pairs_distance):
                pairs_2 = np.vstack([pairs_2, np.array([syn_index, mcs_index])])
    pairs_2 = pairs_2[1:]
    return pairs_2


# In[13]:


def createPairs2_1(syn_array, mcs_array = mcs_tuples, pairs_distance = 4000):
    ## meant for array inputs
    """
    Creates an array of initial possible pairs of synoptic tracks and MCS tracks. Condition: MCS end is after synoptic start, MCS end is before synoptic end, first MCS & synoptic point within distance. 
 
    ---------
    syn_array : 1d numpy array
        1d array of a synoptic track (same format as csv)
    mcs_array : 2d numpy array
        2d array of all MCS tracks to test against synoptic track
    distance : integer
        distance (km) to which the initial points should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
 
    Returns
    ---------
    numpy.array
        2-dimensional array in form [synoptic_track, mcs_track] of the global index (from CSV file) of each potential pair. 
    """
    pairs_1 = np.array([0,0], dtype=str)
    ## all of the 0 indices in this function are used to obtain the 0th (first) lat/lon of the respective tracks in order to test their initial distance and filter based on such 
    M_s_ARR = pd.to_datetime(mcs_array[:,3])
    syn_index = syn_array[0]
    syn=synIndexToTrack(syn_index)
    C_s = pd.to_datetime(syn_array[2])
    C_e = pd.to_datetime(syn_array[3])
    syn_lat_0 = syn.lat.to_numpy()[0]
    syn_lon_0 = syn.lon.to_numpy()[0]
    syn_coords_0 = (syn_lat_0, syn_lon_0)
    matchingInds = np.where(np.logical_and(M_s_ARR<C_e, M_s_ARR>=C_s))[0]
    for ind in matchingInds:
        mcs_index = mcs_array[ind][0]
        mcs = mcsIndexToTrack(mcs_index)
        mcs_lat_0 = (mcs.meanlat.data[0])
        mcs_lon_0 = (mcs.meanlon.data[0])
        mcs_coords_0 = (mcs_lat_0, mcs_lon_0)
        dist = distance.distance(syn_coords_0, mcs_coords_0).km
        if(dist < pairs_distance):
            pairs_1 = np.vstack([pairs_1, np.array([syn_index, mcs_index])])
    pairs_1 = pairs_1[1:]
    return pairs_1




# 
# defaultMCS = '/home/glach/projects/def-rfajber/shared/tracks-gabe/MCS_Tracks.csv'
# mcs_file = pd.read_csv(defaultMCS)
# def mcsIndexToTrack(Index, mcs_csv = mcs_file):
#     """
#     Creates dataset for MCS track from global index.
#  
#     ---------
#     Index : str
#         Global index of MCS track found in CSV file
#     csv_file_path : str
#         '/home/glach/projects/def-rfajber/shared/tracks-gabe/MCS_Tracks.csv'
#     Returns
#     ---------
#     xarray.dataset
#         Dataset of singular MCS track which corresponds to given global index. 
#     """
#     match = mcs_csv.index[mcs_csv['Global_Index'] == Index].tolist()[0] ## returns where the inputted global index is found in the csv file - the 0th index here takes out the specific row index to get the other information from
#     track = mcs_csv.Local_Index[match]
#     path = mcs_csv.path[match]
#     dataset = xr.open_dataset(path)
#     dataset = dataset.sel(tracks=track)
#     return dataset
# 

# In[ ]:


defaultMCS = '/home/glach/projects/def-rfajber/shared/tracks-summer-2024/MCS_Tracks_3.csv'
mcs_file = pd.read_csv(defaultMCS)
default_2014_netcdf_path = '/home/glach/projects/def-rfajber/shared/tracks-summer-2024/new datasets/2014.nc'
default_2015_netcdf_path = '/home/glach/projects/def-rfajber/shared/tracks-summer-2024/new datasets/2015.nc'
default_2016_netcdf_path = '/home/glach/projects/def-rfajber/shared/tracks-summer-2024/new datasets/2016.nc'
default_2017_netcdf_path = '/home/glach/projects/def-rfajber/shared/tracks-summer-2024/new datasets/2017.nc'
default_2018_netcdf_path = '/home/glach/projects/def-rfajber/shared/tracks-summer-2024/new datasets/2018.nc'
default_2019_netcdf_path = '/home/glach/projects/def-rfajber/shared/tracks-summer-2024/new datasets/2019.nc'
dataset2014 = xr.open_dataset(default_2014_netcdf_path)
dataset2015 = xr.open_dataset(default_2015_netcdf_path)
dataset2016 = xr.open_dataset(default_2016_netcdf_path)
dataset2017 = xr.open_dataset(default_2017_netcdf_path)
dataset2018 = xr.open_dataset(default_2018_netcdf_path)
dataset2019 = xr.open_dataset(default_2019_netcdf_path)
def mcsIndexToTrack(Index, mcs_csv=mcs_file, dataset2014 = dataset2014, dataset2015 = dataset2015, dataset2016 = dataset2016, dataset2017 = dataset2017, dataset2018 = dataset2018, dataset2019 = dataset2019):
    """
    Creates dataset for MCS track from global index.
 
    ---------
    Index : str
        Global index of MCS track found in CSV file
        
    Returns
    ---------
    xarray.dataset
        Dataset of singular MCS track which corresponds to given global index. 
    """
    match = mcs_csv.index[mcs_csv['Global_Index'] == Index].tolist()[0] ## returns where the inputted global index is found in the csv file - the 0th index here takes out the specific row index to get the other information from
    track = mcs_csv.Local_Index[match]
    year = Index[4:8]
    if(year == '2014'):
        dataset = dataset2014
    if(year == '2015'):
        dataset = dataset2015
    if(year == '2016'):
        dataset = dataset2016
    if(year == '2017'):
        dataset = dataset2017
    if(year == '2018'):
        dataset = dataset2018
    if(year == '2019'):
        dataset = dataset2019
    dataset = dataset.sel(tracks=track)
    return dataset


# In[15]:


defaultSYN = '/home/glach/projects/def-rfajber/shared/tracks-gabe/SYN_Tracks.csv'
def synIndexToTrack(Index, csv_file_path=defaultSYN):
    """
    Creates dataframe for synoptic track from global index.
 
    ---------
    Index : str
        Global index of synoptic track found in CSV file
    Returns
    ---------
    pandas.DataFrame
        DataFrame of singular synoptic track which corresponds to given global index.
    """
    syn_csv = pd.read_csv(csv_file_path)
    match = syn_csv.index[syn_csv['Global_Index'] == Index].tolist()[0] ## returns where the inputted global index is found in the csv file - the 0th index here takes out the specific row index to get the other information from
    path = syn_csv.path[match]
    dataset = pd.read_csv(path)
    return dataset


# In[16]:


def count_duplicate_rows(arr):
    """
    Helper function to count the number of rows with duplicate 0,1 columns in an array. Used internally.

    ---------
    arr: np.Array
        Array in which to count duplicate rows.
    Returns
    ---------
    np.Array
        Array of the number of duplicates per instance of a repeat. 
    """
    row_counts = Counter(tuple(row[:2]) for row in arr)
    duplicates_count = [count for count in row_counts.values() if count >= 1]
    return duplicates_count


# ## Essential Functions ##

# In[18]:


def createPossiblePairs(syn_csv, mcs_csv, pairs_distance):
    """
    Creates array combining the two possible initial conditions for pairs from createPairs1 and createPairs2. Removes duplicates. Sorts. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks to compare
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks to compare
    distance : integer
        distance (km) to which the initial points should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
 
    Returns
    ---------
    numpy.array
        2-dimensional array in form [synoptic_track, mcs_track] of the global index (from CSV file) of each potential pair without duplicates and sorted. 
    """
    pairs_1 = createPairs1(syn_csv, mcs_csv, pairs_distance)
    pairs_2 = createPairs2(syn_csv, mcs_csv, pairs_distance)
    arr = np.concatenate((pairs_1, pairs_2))
    unique_rows, indices = np.unique(arr, axis=0, return_index=True)
    possiblePairs = arr[np.sort(indices)]
    return possiblePairs


# In[25]:


def createPossiblePairs_1(syn_csv, mcs_csv=mcs_tuples, pairs_distance=4000):
    """
    Creates array combining the two possible initial conditions for pairs from createPairs1 and createPairs2. Removes duplicates. Sorts. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks to compare
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks to compare
    distance : integer
        distance (km) to which the initial points should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
 
    Returns
    ---------
    numpy.array
        2-dimensional array in form [synoptic_track, mcs_track] of the global index (from CSV file) of each potential pair without duplicates and sorted. 
    """
    pairs_1 = createPairs1_1(syn_csv)
    pairs_2 = createPairs2_1(syn_csv)
    if((pairs_1.ndim == 2) & (pairs_2.ndim != 2)):
        return pairs_1
    elif((pairs_1.ndim != 2) & (pairs_2.ndim == 2)):
        return pairs_2
    else:
        arr = np.concatenate((pairs_1, pairs_2))
        unique_rows, indices = np.unique(arr, axis=0, return_index=True)
        possiblePairs = arr[np.sort(indices)]
        return possiblePairs


# In[20]:


def createCoupledPairs(syn_csv, mcs_csv, pairs_distance, coupling_distance, time_delta):
    """
    Creates array of data of pairs which follow the conditions: satisfies pairs1 and pairs2. There exists a point between the MCS track and synoptic track that exists within 3 hours (plus or minus 1.5 hours) of eachother and these two points are within 500km of eachother. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks to compare
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks to compare
    pairs_distance : integer
        distance (km) to which the initial points (using createPairs1 and createPairs2) should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
    coupling_distance : integer
        distance (km) for which to test each point as a feasible pair. Should be a relatively small value, i.e. < 500km.
    time_delta : float
        hour difference for which each point is considered coupled, i.e. if 3 is entered any point within +-3 hours (6 hour range) is temporally coupled. Typical value is 1.5
 
    Returns
    ---------
    numpy.array
        2-dimensional str array of 10 columns. Columns:
            0: synoptic global index
            1: MCS global index
            2: number of indices which were coupled 
            3: synoptic time 
            4: MCS time
            5: synoptic latitude
            6: synoptic longitude
            7: MCS latitude
            8: MCS longitude
            9: distance between points
    """
    possiblePairs = createPossiblePairs(syn_csv, mcs_csv, pairs_distance)
    if(possiblePairs.ndim == 1): ## returns if no input pairs
        return
    coupledPairs = np.array([0,0,0,0,0,0,0,0,0,0])
    for tracks in possiblePairs: #iterates thru each synoptic/mcs track pair
        syn_track = synIndexToTrack(tracks[0])
        syn_lat_arr = syn_track.lat.to_numpy()
        syn_lon_arr = syn_track.lon.to_numpy()
        syn_internal_idx = 0
        length_pair = 0
        while syn_internal_idx < syn_lat_arr.size: #iterates thru each point in synoptic track
            mcs_dataset = mcsIndexToTrack(tracks[1])
            synCoords = (syn_lat_arr[syn_internal_idx], syn_lon_arr[syn_internal_idx])
            mcs_lat_arr = (mcs_dataset.meanlat.data) 
            mcs_lon_arr = (mcs_dataset.meanlon.data)
            synTime = synToTimestamp(syn_track, syn_internal_idx)
            syn_time_lower = synTime - pd.Timedelta(hours=time_delta) 
            syn_time_upper = synTime + pd.Timedelta(hours=time_delta)
            for i, mcs_lat in enumerate(mcs_lat_arr): #iterates thru each point in mcs track
                if(~np.isnan(mcs_lat)): # only checks MCS points that are non-NaN due to standardized 200-length tracks filled with NaN after the final timestep
                    mcs_time = pd.to_datetime((mcs_dataset.base_time.data)[i])
                    if(syn_time_lower <= mcs_time <= syn_time_upper): # checks to see if the current MCS point is in the temporal range to be coupled to current synoptic point as described with time_delta. Includes boundaries. 
                        mcs_coords = (mcs_lat, mcs_lon_arr[i])
                        dist = distance.distance(mcs_coords, synCoords).km
                        if(dist <= coupling_distance): # checks to see if current MCS point is in spatial range (within coupling_distance) of current synoptic point (in km)
                            length_pair = length_pair+1
                            coupledPairs = np.vstack([coupledPairs, np.array([tracks[0], tracks[1], length_pair, str(synTime), str(mcs_time), syn_lat_arr[syn_internal_idx], syn_lon_arr[syn_internal_idx], mcs_lat,                                         mcs_lon_arr[i], dist])])
            syn_internal_idx = syn_internal_idx+1
    coupledPairs = coupledPairs[1:]
    return coupledPairs

#format: ([syn_index, mcs_index, lengthOfCoupling, syn_time, mcs_time, syn_lat, syn_lon, mcs_lat, mcs_lon, dist_between_points])


# In[21]:


def createCoupledPairs_1(syn_csv, mcs_csv, pairs_distance, coupling_distance, time_delta):
    """
    Creates array of data of pairs which follow the conditions: satisfies pairs1 and pairs2. There exists a point between the MCS track and synoptic track that exists within 3 hours (plus or minus 1.5 hours) of eachother and these two points are within 500km of eachother. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks to compare
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks to compare
    pairs_distance : integer
        distance (km) to which the initial points (using createPairs1 and createPairs2) should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
    coupling_distance : integer
        distance (km) for which to test each point as a feasible pair. Should be a relatively small value, i.e. < 500km.
    time_delta : float
        hour difference for which each point is considered coupled, i.e. if 3 is entered any point within +-3 hours (6 hour range) is temporally coupled. Typical value is 1.5
 
    Returns
    ---------
    numpy.array
        2-dimensional str array of 10 columns. Columns:
            0: synoptic global index
            1: MCS global index
            2: number of indices which were coupled 
            3: synoptic time 
            4: MCS time
            5: synoptic latitude
            6: synoptic longitude
            7: MCS latitude
            8: MCS longitude
            9: distance between points
    """
    possiblePairs = createPossiblePairs_1(syn_csv, mcs_csv, pairs_distance)
    coupledPairs = np.array([0,0,0,0,0,0,0,0,0,0])
    if(possiblePairs.ndim == 1): ## returns if no input pairs
        return np.array([0,0,0,0,0,0,0,0,0])
    for tracks in possiblePairs: #iterates thru each synoptic/mcs track pair
        syn_track = synIndexToTrack(tracks[0])
        syn_lat_arr = syn_track.lat.to_numpy()
        syn_lon_arr = syn_track.lon.to_numpy()
        syn_internal_idx = 0
        length_pair = 0
        while syn_internal_idx < syn_lat_arr.size: #iterates thru each point in synoptic track
            mcs_dataset = mcsIndexToTrack(tracks[1])
            synCoords = (syn_lat_arr[syn_internal_idx], syn_lon_arr[syn_internal_idx])
            mcs_lat_arr = (mcs_dataset.meanlat.data) 
            mcs_lon_arr = (mcs_dataset.meanlon.data)
            synTime = synToTimestamp(syn_track, syn_internal_idx)
            syn_time_lower = synTime - pd.Timedelta(hours=time_delta) 
            syn_time_upper = synTime + pd.Timedelta(hours=time_delta)
            for i, mcs_lat in enumerate(mcs_lat_arr): #iterates thru each point in mcs track
                if(~np.isnan(mcs_lat)): # only checks MCS points that are non-NaN due to standardized 200-length tracks filled with NaN after the final timestep
                    mcs_time = pd.to_datetime((mcs_dataset.base_time.data)[i])
                    if(syn_time_lower <= mcs_time <= syn_time_upper): # checks to see if the current MCS point is in the temporal range to be coupled to current synoptic point as described with time_delta. Includes boundaries. 
                        mcs_coords = (mcs_lat, mcs_lon_arr[i])
                        dist = distance.distance(mcs_coords, synCoords).km
                        if(dist <= coupling_distance): # checks to see if current MCS point is in spatial range (within coupling_distance) of current synoptic point (in km)
                            length_pair = length_pair+1
                            coupledPairs = np.vstack([coupledPairs, np.array([tracks[0], tracks[1], length_pair, str(synTime), str(mcs_time), syn_lat_arr[syn_internal_idx], syn_lon_arr[syn_internal_idx], mcs_lat,                                         mcs_lon_arr[i], dist])])
            syn_internal_idx = syn_internal_idx+1
    coupledPairs = coupledPairs[1:]
    return coupledPairs

#format: ([syn_index, mcs_index, lengthOfCoupling, syn_time, mcs_time, syn_lat, syn_lon, mcs_lat, mcs_lon, dist_between_points])


# In[22]:


def createCoupledPairs_2(syn_csv, mcs_csv=mcs_tuples, pairs_distance=4000, coupling_distance=250, time_delta=1.5):
    #tuple[syn_csv, mcs_csv, pairs_distance, coupling_distance, time_delta]
    """
    Creates array of data of pairs which follow the conditions: satisfies pairs1 and pairs2. There exists a point between the MCS track and synoptic track that exists within 3 hours (plus or minus 1.5 hours) of eachother and these two points are within 500km of eachother. 
 
    ---------
    syn_csv : pandas CSV DataFrame
        DataFrame of synoptic tracks to compare
    mcs_csv : pandas CSV DataFrame
        DataFrame of MCS tracks to compare
    pairs_distance : integer
        distance (km) to which the initial points (using createPairs1 and createPairs2) should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
    coupling_distance : integer
        distance (km) for which to test each point as a feasible pair. Should be a relatively small value, i.e. < 500km.
    time_delta : float
        hour difference for which each point is considered coupled, i.e. if 3 is entered any point within +-3 hours (6 hour range) is temporally coupled. Typical value is 1.5
 
    Returns
    ---------
    numpy.array
        2-dimensional str array of 10 columns. Columns:
            0: synoptic global index
            1: MCS global index
            2: number of indices which were coupled 
            3: synoptic time 
            4: MCS time
            5: synoptic latitude
            6: synoptic longitude
            7: MCS latitude
            8: MCS longitude
            9: distance between points
    """
    possiblePairs = createPossiblePairs_1(syn_csv, mcs_csv, pairs_distance)
    coupledPairs = np.array([0,0,0,0,0,0,0,0,0,0])
    if(possiblePairs.ndim == 1): ## returns if no input pairs
        return np.array([0,0,0,0,0,0,0,0,0])
    for tracks in possiblePairs: #iterates thru each synoptic/mcs track pair
        syn_track = synIndexToTrack(tracks[0])
        syn_lat_arr = syn_track.lat.to_numpy()
        syn_lon_arr = syn_track.lon.to_numpy()
        syn_internal_idx = 0
        length_pair = 0
        while syn_internal_idx < syn_lat_arr.size: #iterates thru each point in synoptic track
            mcs_dataset = mcsIndexToTrack(tracks[1])
            synCoords = (syn_lat_arr[syn_internal_idx], syn_lon_arr[syn_internal_idx])
            mcs_lat_arr = (mcs_dataset.meanlat.data) 
            mcs_lon_arr = (mcs_dataset.meanlon.data)
            synTime = synToTimestamp(syn_track, syn_internal_idx)
            syn_time_lower = synTime - pd.Timedelta(hours=time_delta) 
            syn_time_upper = synTime + pd.Timedelta(hours=time_delta)
            for i, mcs_lat in enumerate(mcs_lat_arr): #iterates thru each point in mcs track
                if(~np.isnan(mcs_lat)): # only checks MCS points that are non-NaN due to standardized 200-length tracks filled with NaN after the final timestep
                    mcs_time = pd.to_datetime((mcs_dataset.base_time.data)[i])
                    if(syn_time_lower <= mcs_time <= syn_time_upper): # checks to see if the current MCS point is in the temporal range to be coupled to current synoptic point as described with time_delta. Includes boundaries. 
                        mcs_coords = (mcs_lat, mcs_lon_arr[i])
                        dist = distance.distance(mcs_coords, synCoords).km
                        if(dist <= coupling_distance): # checks to see if current MCS point is in spatial range (within coupling_distance) of current synoptic point (in km)
                            length_pair = length_pair+1
                            coupledPairs = np.vstack([coupledPairs, np.array([tracks[0], tracks[1], length_pair, str(synTime), str(mcs_time), syn_lat_arr[syn_internal_idx], syn_lon_arr[syn_internal_idx], mcs_lat,                                         mcs_lon_arr[i], dist])])
            syn_internal_idx = syn_internal_idx+1
    coupledPairs = coupledPairs[1:]
    return coupledPairs

#format: ([syn_index, mcs_index, lengthOfCoupling, syn_time, mcs_time, syn_lat, syn_lon, mcs_lat, mcs_lon, dist_between_points])


# In[23]:


# In[ ]:


csv_path = '/home/glach/projects/def-rfajber/shared/tracks-summer-2024/coupledtracks_nh'
# In[23]:
def createCoupledAndSave(syn_csv, mcs_csv=mcs_tuples, pairs_distance=4000, coupling_distance=250, time_delta=1.5, base_path=csv_path):
    #tuple[syn_csv, mcs_csv, pairs_distance, coupling_distance, time_delta]
    """
    Creates array of data of pairs which follow the conditions: satisfies pairs1 and pairs2. There exists a point between the MCS track and synoptic track that exists within 3 hours (plus or minus 1.5 hours) of eachother and these two points are within 500km of eachother. Saves these pairs as CSV files.
 
    ---------
    syn_csv : 1d array
        1d array of a synoptic track (same format as csv)
    mcs_csv : 2d array
        2d array of all MCS tracks to test against synoptic track
    pairs_distance : integer
        distance (km) to which the initial points (using createPairs1 and createPairs2) should be compared and filtered as such (should be a large value i.e. > 1000km, meant to make the output smaller by filtering out impossible pairs)
    coupling_distance : integer
        distance (km) for which to test each point as a feasible pair. Should be a relatively small value, i.e. < 500km.
    time_delta : float
        hour difference for which each point is considered coupled, i.e. if 3 is entered any point within +-3 hours (6 hour range) is temporally coupled. Typical value is 1.5
 
    Returns
    ---------
    void
    """
    possiblePairs = createPossiblePairs_1(syn_csv, mcs_csv, pairs_distance)
    if(possiblePairs.ndim == 1): ## returns if no input pairs
        return
    ##
    for tracks in possiblePairs: #iterates thru each synoptic/mcs track pair

        coupledPairs = np.array([0,0,0,0,0,0,0,0,0,0])
        syn_track = synIndexToTrack(tracks[0])
        syn_lat_arr = syn_track.lat.to_numpy()
        syn_lon_arr = syn_track.lon.to_numpy()
        syn_internal_idx = 0
        length_pair = 0
        while syn_internal_idx < syn_lat_arr.size: #iterates thru each point in synoptic track
            mcs_dataset = mcsIndexToTrack(tracks[1])
            synCoords = (syn_lat_arr[syn_internal_idx], syn_lon_arr[syn_internal_idx])
            mcs_lat_arr = (mcs_dataset.meanlat.data) 
            mcs_lon_arr = (mcs_dataset.meanlon.data)
            synTime = synToTimestamp(syn_track, syn_internal_idx)
            syn_time_lower = synTime - pd.Timedelta(hours=time_delta) 
            syn_time_upper = synTime + pd.Timedelta(hours=time_delta)
            for i, mcs_lat in enumerate(mcs_lat_arr): #iterates thru each point in mcs track
                if(~np.isnan(mcs_lat)): # only checks MCS points that are non-NaN due to standardized 200-length tracks filled with NaN after the final timestep
                    mcs_time = pd.to_datetime((mcs_dataset.base_time.data)[i])
                    if(syn_time_lower <= mcs_time <= syn_time_upper): # checks to see if the current MCS point is in the temporal range to be coupled to current synoptic point as described with time_delta. Includes boundaries. 
                        mcs_coords = (mcs_lat, mcs_lon_arr[i])
                        dist = distance.distance(mcs_coords, synCoords).km
                        if(dist <= coupling_distance): # checks to see if current MCS point is in spatial range (within coupling_distance) of current synoptic point (in km)
                            length_pair = length_pair+1
                            coupledPairs = np.vstack([coupledPairs, np.array([tracks[0], tracks[1], length_pair, str(synTime), str(mcs_time), syn_lat_arr[syn_internal_idx], syn_lon_arr[syn_internal_idx], mcs_lat,                                         mcs_lon_arr[i], dist])])
            syn_internal_idx = syn_internal_idx+1

        pairArr = np.array(['Syn_Global_Index','MCS_Global_Index', 'Coupled_Points', 'Syn_Time', 'MCS_Time', 'Syn_Lat', 'Syn_Lon', 'MCS_Lat', 'MCS_Lon', 'Distance'], dtype=str)
        coupledPairs = coupledPairs[1:]
        currentSyn = tracks[0]
        currentMCS = tracks[1]
        if(coupledPairs.ndim != 1):
            i_2=1
            j_2=0
            rows = np.array([], dtype=int)
            while i_2<len(coupledPairs): # filters out edge cases by only keeping the point which has a nearer distance
                if(coupledPairs[i_2][4] == coupledPairs[i_2-1][4]): # 4 is the syn lat
                    if(coupledPairs[i_2][9] > coupledPairs[i_2-1][9]): # 9 is the distance between the coupled points
                        rows = np.append(rows, i_2)
                    else:
                        rows = np.append(rows, i_2-1)
                    j_2 = j_2+1
                i_2=i_2+1
            if 0 in rows: # makes sure that the global indices aren't filtered out since they are contained in the 0th row which may be an edge case
                coupledPairs[1] = coupledPairs[0]
            new_Arr_2 = coupledPairs[~np.isin(np.arange(coupledPairs.shape[0]), rows)] 
            couplings = int(new_Arr_2[0][2]) #sets the correct number of coupled points in the data (3rd column)
            new_Arr_2[:,0] = ''
            new_Arr_2[:,1] = ''
            new_Arr_2[:,2] = ''

            new_Arr_2[0][0] = currentSyn #
            new_Arr_2[0][1] = currentMCS #
            new_Arr_2[0][2] = str(len(new_Arr_2)) #
            
            #
            final_arr = np.vstack((pairArr, new_Arr_2)) # creates full assembled array of data 
            
            name = f'{currentSyn}-{currentMCS}.csv' # names the file 
            path = f'{base_path}/{name}' # saves to this path
            with open(path, 'w', newline='') as csvfile: # creates and saves csv file
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(final_arr)
            ##

#format: ([syn_index, mcs_index, lengthOfCoupling, syn_time, mcs_time, syn_lat,


# In[23]:


def save_data(pairs, BASE_PATH):
    """
    Saves data of potential pairs to individual CSV files. 
 
    ---------
    pairs : np.array
        Array of pairs exactly as outputted from createCoupledPairs. 
    BASE_PATH : str
        "/home/glach/projects/def-rfajber/shared/tracks-gabe/pairs"
 
    void return 
    ---------
    Saves data as individual csv files. Removes duplicate MCS points (edge cases). 
        Format: 
            0: synoptic global index
            1: MCS global index
            2: number of indices which were coupled 
            3: synoptic time 
            4: MCS time
            5: synoptic latitude
            6: synoptic longitude
            7: MCS latitude
            8: MCS longitude
            9: distance between points
    """
    if(pairs.ndim == 1):
        return
    duplicateMembers = np.array(count_duplicate_rows(pairs)) # array of how many duplicates there are of each pair based on global indices (how many matching points)
    cur = 0
    pairArr = np.array(['Syn_Global_Index','MCS_Global_Index', 'Coupled_Points', 'Syn_Time', 'MCS_Time', 'Syn_Lat', 'Syn_Lon', 'MCS_Lat', 'MCS_Lon', 'Distance'], dtype=str)
    for count in duplicateMembers:
        count = int(count)
        currentSyn = str(pairs[cur][0])
        currentMCS = str(pairs[cur][1])
        data_arr = pairs[cur:cur+count]
        #
        i_2=1
        j_2=0
        rows = np.array([], dtype=int)
        while i_2<len(data_arr): # filters out edge cases by only keeping the point which has a nearer distance
            if(data_arr[i_2][4] == data_arr[i_2-1][4]): # 4 is the syn lat
                if(data_arr[i_2][9] > data_arr[i_2-1][9]): # 9 is the distance between the coupled points
                    rows = np.append(rows, i_2)
                else:
                    rows = np.append(rows, i_2-1)
                j_2 = j_2+1
            i_2=i_2+1
        if 0 in rows: # makes sure that the global indices aren't filtered out since they are contained in the 0th row which may be an edge case
            data_arr[1] = data_arr[0]
        new_Arr_2 = data_arr[~np.isin(np.arange(data_arr.shape[0]), rows)] 
        couplings = int(new_Arr_2[0][2]) #sets the correct number of coupled points in the data (3rd column)
        new_Arr_2[:,0] = ''
        new_Arr_2[:,1] = ''
        new_Arr_2[:,2] = ''
        
        new_Arr_2[0][0] = currentSyn #
        new_Arr_2[0][1] = currentMCS #
        new_Arr_2[0][2] = str(len(new_Arr_2)) #
        #
        final_arr = np.vstack((pairArr, new_Arr_2)) # creates full assembled array of data 

        name = f'{currentSyn}-{currentMCS}.csv' # names the file 
        path = f'{BASE_PATH}/{name}' # saves to this path
        with open(path, 'w', newline='') as csvfile: # creates and saves csv file
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(final_arr)
        cur = cur + count


# In[24]:


def getPairsType(pairs_file_paths):
    """
    Returns array of useful info about individual pairs. 
    format: [synindex, mcsindex, #%coupled, #%before, #%after]
 
    ---------
    pairs_file_paths : string array
        array of paths to pairs csv files 
    ---------
    Returns array of useful info about individual pairs. 
        Format: 
            0: synoptic global index
            1: MCS global index
            2: percent of MCS indices which were coupled (compared to full MCS track)  
            3: percent of MCS indices which were coupled BEFORE the midpoint of the track
            4: percent of MCS indices which were coupled AFTER the midpoint of the track
    """
    coolInfo = np.array([0,0,0,0,0], dtype = str)

    for track in pairs_file_paths:
        pair = pd.read_csv(track)
        synIndex = pair.Syn_Global_Index[0]
        mcsIndex = pair.MCS_Global_Index[0]
        mcs = sc.mcsIndexToTrack(mcsIndex)
        coupledMCStimes = pd.to_datetime(pair.MCS_Time.tolist())
        ind = (~np.isnan(mcs.base_time.data)).cumsum().argmax()-1
        times = mcs.base_time.data[0:ind]
        midIndex = int(ind/2)
        coupledBefore = np.where(coupledMCStimes < times[midIndex])[0]
        coupledAfter = np.where(coupledMCStimes >= times[midIndex])[0]
        numBefore = len(coupledBefore)
        numAfter = len(coupledAfter)
        percentBefore = str(float(numBefore / (ind+1)))
        percentAfter = str(float(numAfter / (ind+1)))
        percentCoupled = str(float((len(coupledMCStimes))/(ind+1)))
        arr = np.array([synIndex, mcsIndex, percentCoupled, percentBefore, percentAfter], dtype = str)
        coolInfo = np.vstack([coolInfo, arr])
    coolInfo = coolInfo[1:]

