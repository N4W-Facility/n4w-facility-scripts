# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:14:30 2023

@author: douglas.nyolei
"""

#required packages
import requests
from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import rasterio
import rasterstats
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import scipy.sparse as sparse
import patoolib
from osgeo import gdal

#define paths to zipped files and the output path where unzipped files will be saved
input_path=r'C:\Douglas_offline\Weather\CHIRPS\Unzipped_prec' #change paths to folders accordingly
output_path=r'C:\Douglas_offline\Yala\GIS\Weather\Precipitation2'
        
#define path to shapefile
shapefile=r"C:\Douglas_offline\Yala\GIS\Yala_Frame2.shp"

raster_list = [raster for raster in os.listdir(input_path) if raster[-4:]=='.tif']

#clip all the selected raster files with the Warp option from GDAL
for raster in raster_list:
    print(output_path)
    options = gdal.WarpOptions(cutlineDSName=shapefile,cropToCutline=True)
    outBand = gdal.Warp(srcDSOrSrcDSTab=os.path.join(input_path, raster),
                        destNameOrDestDS=os.path.join(output_path, raster[:-4]+raster[-4:]),
                        options=options)
    outraster= None