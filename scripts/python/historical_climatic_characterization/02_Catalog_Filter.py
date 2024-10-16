import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization/'


name_input = '04_Data_Completa.xlsx' # entrada datos series
name_ac = '01_Completeness_analysis.xlsx'# entrada completitud
catalogo = 'Catalogo_General.xlsx'
path_out= r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

Fecha_Inicio = '1970-01-01' # entrada


xls_writer = pd.ExcelWriter(path_out+'02_Selected_data.xlsx') # salida
xls_wc = pd.ExcelWriter(path_out+'03_Catalog_Check.xlsx')# salida'03_Catalogo_Cumplen.xlsx'

file = pd. ExcelFile(path+name_input)
file_select = pd. ExcelFile(path_out+name_ac)
variables = file.sheet_names

catalogo = pd.read_excel(path+catalogo,'Catalogo',index_col=0) # entrada catalogo general

for var in variables:

    print(var)

    data = file.parse(var,index_col=0)

    estaciones = file_select.parse(var,index_col=0)

    select_sta = estaciones['Cumple'][estaciones['Cumple']==True].index

    datos_selected = pd.DataFrame( columns=select_sta , index=data.index)
    new_catalogo = pd.DataFrame( columns=select_sta)
    for sta in select_sta:
        print(sta)
        try:
            datos_selected[sta] = data[sta]  # datos[es]

            new_catalogo[sta]=catalogo.loc[sta]
        except:
            pass

    datos_selected = datos_selected.loc[Fecha_Inicio:]
    new_catalogo = new_catalogo.T
    datos_selected.to_excel(xls_writer, sheet_name=var, merge_cells=False)
    new_catalogo.to_excel(xls_wc, sheet_name=var, merge_cells=False)

xls_writer.close()
xls_wc.close()