import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime



path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization/'
name_input = '04_Data_Completa.xlsx'

path_out= r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization/'

xls_writer = pd.ExcelWriter(path_out+'01_Completeness_analysis.xlsx')

file = pd. ExcelFile(path+name_input)

variables = file.sheet_names

year_i = 1970#1980
year_f = 2024#


for var in variables:

    if var == 'PT_4':
        N_year = 20
        porc_faltantes = 30

    if var == 'TS_1' or var == 'TS_2' or var == 'TS_3':

        N_year = 20
        porc_faltantes = 30

    else:
        N_year = 20
        porc_faltantes = 35


    print(var)

    data = file.parse(var,index_col=0)

    estaciones= data.columns

    lres = []
    for sta in estaciones:
        print(sta)

        serie = data[sta]


        start =serie.dropna().index.min()
        end = serie.dropna().index.max()

        record = np.round((end - start).days/365,1)
        lon_obs = serie.dropna().index.size

        lon_esp = (year_f - year_i) * 365

        faltantes = 100.0 * float((lon_esp - lon_obs)) / lon_esp

        porcen_completos = 100.0 - faltantes

        Cumple = record >= N_year and faltantes <= porc_faltantes

        res = [sta, start, end, record, lon_esp, lon_obs, faltantes, porcen_completos,Cumple]

        lres.append(res)
    result = pd.DataFrame(lres, columns=['Codigo Estacion', 'Fecha Inicio', 'Fecha Fin', 'Años', 'Longitud Esperada',
                                         'Longitud Observada', '% Faltantes', '% Completitud', 'Cumple'])
    result=result.set_index('Codigo Estacion')
    result.to_excel(xls_writer, sheet_name=var, merge_cells=False)


xls_writer.close()