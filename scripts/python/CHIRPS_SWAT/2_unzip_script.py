# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:43:23 2023

@author: douglas.nyolei
"""
import gzip
import glob
import os.path
source_dir = r"C:\Douglas_offline\Weather\CHIRPS\Precipitation"
dest_dir = r"C:\Douglas_offline\Weather\CHIRPS\Unzipped_prec"

# Unzipped the gz zipped files
for src_name in glob.glob(os.path.join(source_dir, '*.gz')):
    base = os.path.basename(src_name)
    dest_name = os.path.join(dest_dir, base[:-3])
    with gzip.open(src_name, 'rb') as infile:
        with open(dest_name, 'wb') as outfile:
            for line in infile:
                outfile.write(line)
    
# Changing the unzipped file formats to .tif                
for unzipped_file in os.listdir(r'C:\Douglas_offline\Weather\CHIRPS\Unzipped_prec'):
    os.chdir(r'C:\Douglas_offline\Weather\CHIRPS\Unzipped_prec')
    os.rename(unzipped_file, unzipped_file +'.tif')