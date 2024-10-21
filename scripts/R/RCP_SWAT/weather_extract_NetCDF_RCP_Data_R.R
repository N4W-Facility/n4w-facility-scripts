#---------------------------------------------------------------
# Project   : N4W BLACK VOLTA
# Purpose   : EXTRACT NetCDF CLIMATE FORECAST AT COORDINATES
# Author    : Corjan Nolet, c.nolet@futurewater.nl
# Date      : JUNE 2024
#----------------------------------------------------------------

# list packages
packages <- c("terra") 

# Install packages if needed 
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))
}
# Load packages
lapply(packages , require, character.only = TRUE)

#------------------------------------------------------------------
# STATION COORDINATES
#------------------------------------------------------------------
base_path <- 'D:/Pegasys Strategy and Development/Nature 4 Water Facility - 2. SE15_Black Volta_Ghana/03. Resources/09. SWAT/Climate Data/Climate_Forecast/'

# station coordinates to dataframe & as point shapefile
df <- read.table(paste0(base_path,'/Station_coordinates.txt'), sep=',', header = T, stringsAsFactors = F)
stat_loc <- vect(df[,c(3,4)], geom=c("LAT", "LONG"))

# BV catchment
bv <- vect(paste0(base_path, 'Black_Volta_1.shp'))

#------------------------------------------------------------------
# PRECIPITATION
#------------------------------------------------------------------
# list nc files & load into single SpatRaster
nc_paths <- list.files(paste0(base_path, '/RCP45/Precipitation'), pattern='.nc', full.names = T, recursive = T)
nc <- rast(nc_paths)

# check data
plot(nc[[1]])
plot(bv, add=T)
plot(stat_loc, add=T)

# extract based on station location shapefile
pr <-  extract(nc, stat_loc)
pr_tp <- as.data.frame(t(pr[,c(2:ncol(pr))])) # transpose
pr_mm <- round(pr_tp * 86400, 2) # kg/m2/s to mm
colnames(pr_mm) <- df$NAME
row.names(pr_mm) <- NULL

# get date header
t <- time(nc)
t <- format(as.Date(t),'%Y%m%d')

for(i in seq_along(pr_mm)){
  pr_stat <- data.frame(pr_mm[,i])
  colnames(pr_stat) <- t[1]
  outfile <- paste0('pcp_',names(pr_mm)[i],'_',t[1],'_',t[length(t)],'.txt')
  write.table(pr_stat, paste0(base_path,'EXTRACT/',outfile), sep=',', row.names = F)
}

#------------------------------------------------------------------
# TEMPERATURE
#------------------------------------------------------------------
# list nc files & load into single SpatRaster
max_paths <- list.files(paste0(base_path, 'RCP45/Tmax'), pattern='.nc', full.names = T, recursive = T)
nc_max <- rast(max_paths)

min_paths <- list.files(paste0(base_path, 'RCP45/Tmin'), pattern='.nc', full.names = T, recursive = T)
nc_min <- rast(min_paths)

# get date header
t <- time(nc_max)
t <- format(as.Date(t),'%Y%m%d')

# check data
plot(nc_max[[1]])
plot(bv, add=T)
plot(stat_loc, add=T)

# extract TMAX based on station location shapefile
tmax  <-  extract(nc_max, stat_loc)
tmax_tp <- as.data.frame(t(tmax[,c(2:ncol(tmax))])) # transpose
tmax_c <- round(tmax_tp - 273.15, 2) # Kelvin to Celcius
colnames(tmax_c) <- df$NAME
row.names(tmax_c) <- NULL

# extract TMIN based on station location shapefile
tmin  <-  extract(nc_min, stat_loc)
tmin_tp <- as.data.frame(t(tmin[,c(2:ncol(tmin))])) # transpose
tmin_c <- round(tmin_tp - 273.15, 2) # Kelvin to Celcius
colnames(tmin_c) <- df$NAME
row.names(tmin_c) <- NULL

for(i in seq_along(tmax_c)){
  tmax_stat <- tmax_c[,i]
  tmin_stat <- tmin_c[,i]
  t_stat <- data.frame(tmax_stat, tmin_stat)
  colnames(t_stat) <- rep(t[1],2)
  outfile <- paste0('tmp_',names(tmax_c)[i],'_',t[1],'_',t[length(t)],'.txt')
  write.table(t_stat, paste0(base_path,'EXTRACT/',outfile), sep=',', row.names = F)
}

