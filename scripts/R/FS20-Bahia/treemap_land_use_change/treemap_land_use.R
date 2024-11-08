#-------------------------------------------------------------------------------

# Project: N4W FS20-Bahia

# Purpose: Treemap - Analysis of temporal land use drivers (2012,2017,2022) by
# areas: Great Paraguaçu River Basin; Upper Paraguaçu Basin; Marimbus-Iraquara APA

# Database: Land Cover and Use Map, MApBiomas(Collection 9)

# Author: Gislaine Costa, gislaine.costa@tnc.org

# Date: JUNE 2024

#-------------------------------------------------------------------------------

#Packages

library(ggplot2)
library(tidyverse)
library(treemapify)

#-------------------------------------------------------------------------------

#Marimbus

df <- readxl::read_excel('data-raw/LUC.xlsx',sheet = 'marimbus_luc_2022') ## change the year according with the desired one

df |>
  arrange(Class) |>
  ungroup()  |>
  mutate(area_p = Area_m2/sum(Area_m2)*100)  |>
  ggplot(aes(area = area_p, fill = Class))+
  geom_treemap() +
  geom_treemap_text(
    aes(label = paste(Class,
                      paste0(round(area_p, 2), "%"), sep = "\n")),
    colour = "white") +
  theme(legend.position = "none")

ggplot2::ggsave('img_tnc/marimbus_luc_2022.png',
                units="in", width=8, height=6,
                dpi=300)

#-------------------------------------------------------------------------------

# Paraqguaçu

df <- readxl::read_excel('data-raw/LUC.xlsx',sheet = 'paraguacu')


df |>
  ggplot(aes(x=Year,y=Area_m2/1e6,col=Class))+
  geom_line()+
  labs(x='',
       y=expression('Area (km'^2~')'))+
  scale_x_continuous(breaks = seq(2012,2022,5))

ggplot2::ggsave('img_tnc/paraguacu.png',units="in", width=8, height=6,
                dpi=300)


#-------------------------------------------------------------------------------

# Upper Paraguaçu
df <- readxl::read_excel('data-raw/LUC.xlsx',sheet = 'upper_paraguacu_luc_2022')


df |>
  arrange(Class) |>
  ungroup()  |>
  mutate(area_p = Area_m2/sum(Area_m2)*100)  |>
  ggplot(aes(area = area_p, fill = Class))+
  geom_treemap() +
  geom_treemap_text(
    aes(label = paste(Class,
                      paste0(round(area_p, 2), "%"), sep = "\n")),
    colour = "white") +
  theme(legend.position = "none")

ggplot2::ggsave('img_tnc/marimbus_luc_2022.png',
                units="in", width=8, height=6,
                dpi=300)
