# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:29:56 2023

@author: douglas.nyolei
"""

#required packages
import requests
from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import fiona
import rasterio
import rasterstats
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import scipy.sparse as sparse
# import patoolib
from osgeo import gdal

# Create an empty pandas dataframe called 'table'
table = pd.DataFrame(index = np.arange(0,1))

#First create a shapefile of points stations 
#Read the points shapefile using GeoPandas 
stations = gpd.read_file(r'C:\Douglas_Offline\Lower_Kafue\GIS\Weather\Kafue_Dammy_Stations.shp')
stations['long'] = stations['geometry'].x
stations['lat'] = stations['geometry'].y

Matrix = pd.DataFrame()

# Iterate through the rasters and save the data as individual arrays to a Matrix 
for files in os.listdir(r'C:\Douglas_Offline\Lower_Kafue\GIS\Weather\Lower_Kafue_CHIRPS'):
    if files[-4: ] == '.tif':
        dataset = rasterio.open(r'C:\Douglas_Offline\Lower_Kafue\GIS\Weather\Lower_Kafue_CHIRPS'+'\\'+files)
        data_array = dataset.read(1)
        data_array_sparse = sparse.coo_matrix(data_array, shape = (41,68))
        data = files[ :-4]
        Matrix[data] = data_array_sparse.toarray().tolist()
        print('Processing is done for the raster: '+ files[:-4])

# Iterate through the stations and get the corresponding row and column for the related x, y coordinates    
for index, row in stations.iterrows():
        station_name = str(row['Id'])#Name refers to the station name in the stations shapefile
        lon = float(row['Long'])
        lat = float(row['Lat'])
        x,y = (lon, lat)
        row, col = dataset.index(x, y)
        print('Processing: '+ station_name)
       
        # Pick the rainfall value from each stored raster array and record it into the previously created 'table'
        for records_date in Matrix.columns.tolist():
            a = Matrix[records_date]
            rf_value = a.loc[int(row)][int(col)]
            table[records_date] = rf_value
            transpose_mat = table.T
            transpose_mat.rename(columns = {0: 'Rainfall(mm)'}, inplace = True)
        
            transpose_mat.to_csv(r'C:\Douglas_Offline\Lower_Kafue\GIS\Weather\Extracted_Station_Data'+'\\'+station_name+'.txt') #provide path to station data