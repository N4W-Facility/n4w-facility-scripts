#-------------------------------------------------------------------------------

# Project: N4W FS23-TCCC

# Purpose: Sankey plot - Analysis of land cover trasition from 2000-2021 by
# TCCC priority watersheds: France (Artois-Picardie and Upper Seine) and Italy 
#(Sicily)

# Database: Copernicus Land Monitoring Service Product - CORINE Land Cover, 100m

# Author: Gislaine Costa, gislaine.costa@tnc.org

# Date: Octobre 2024

#-------------------------------------------------------------------------------

#Packages

library(tidyverse)
library(viridis)
library(patchwork)
library(hrbrthemes)
library(circlize)
library(networkD3)

#-------------------------------------------------------------------------------

files_ <- list.files('data-raw/',full.names = T)
files_[3]

data <- read.csv(files_[3])[,-c(1,5)] ### changing the file using the index from 1 to 3

data_long <- data |> 
  dplyr::mutate(
    `From.Class`=dplyr::case_when(
      `From.Class` =='FOREST AND SEMI[1]NATURAL AREAS' ~ 'FOREST AND SEMINATURAL AREAS',
      .default = `From.Class`
    ) ,
    `To.Class`=dplyr::case_when(
      `To.Class` =='FOREST AND SEMI[1]NATURAL AREAS' ~ 'FOREST AND SEMINATURAL AREAS',
      .default = `To.Class`
    )
  ) |> 
  janitor::clean_names() |>
  dplyr::select('from_class','to_class','area_sq_m') |> 
  dplyr::mutate(
    'source'=from_class,
    'target'=to_class,
    'value'=area_sq_m
  ) |> 
  dplyr::select(source,target,value)

data_long <- data_long |> 
  dplyr::group_by(source,target) |> 
  dplyr::summarise(
    value=sum(value)
  ) |>
  dplyr::ungroup() |> 
  as.data.frame()

data_long$target <- paste(data_long$target, " ", sep="")


nodes <- data.frame(name=c(as.character(data_long$source), 
                           as.character(data_long$target)) %>% unique())

# With networkD3, connection must be provided using id, not using real name like in the links dataframe.. So we need to reformat it.
data_long$IDsource=match(data_long$source, nodes$name)-1 
data_long$IDtarget=match(data_long$target, nodes$name)-1



# prepare colour scale
ColourScal ='d3.scaleOrdinal() .range(["#FDE725FF","#B4DE2CFF","#6DCD59FF","#35B779FF","#1F9E89FF"])'

# Make the Network
p <- sankeyNetwork(Links = data_long, Nodes = nodes,
                   Source = "IDsource", Target = "IDtarget",
                   Value = "value", NodeID = "name", 
                   sinksRight=F, colourScale=ColourScal,
                   nodeWidth=80, fontSize=20, nodePadding=20,
                   width = 1000, 
                   height = 485
)

p



data_long |> write.csv('transitions_sicily.csv')

webshot::webshot('sankey.html',"sn.png", vwidth = 1000, vheight = 900)

