# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd



path =r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly\03_Fit_PDF/'

path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly\03_Fit_PDF/'

variables = ['PT_4','QL_1','TS_2','TS_3','RS','Vwind','Uwind','HR_1','QL_1']##'BS_4','EV_4'
for v in variables[0:]:

    xls_in = path + 'Fit_' + v + '.xlsx'
    excel_salida =  path_out + 'PDF_'+v+'.xlsx'
    libro_salida = pd.ExcelWriter(excel_salida)

    Datos = pd.ExcelFile(xls_in)
    pestanas = Datos.sheet_names[:]

    meses =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    elegidos_todos = []
    for p in pestanas:
        print (p)
        datos = pd.read_excel(xls_in,  p)
        elegidos_pestana = []
        for m in meses[0:]:
            print (m)
            datos_mes = datos.loc[datos['Columna']==m]
            datos_aprov = datos_mes.loc[datos_mes['Kolmogorov']==1]
            err_min = datos_aprov['Error Medio'].min()
            el_elegido = datos_aprov.loc[datos_aprov['Error Medio']==err_min]['Distribucion']
            if len(el_elegido) == 0:

                datos_aprov_2 = datos_mes.loc[datos_mes['Kolmogorov'] == 0]
                err_min_2 = datos_aprov_2['Error Medio'].min()
                el_elegido_2 = datos_aprov_2.loc[datos_aprov_2['Error Medio'] == err_min_2]['Distribucion']

                el_elegido_2 = el_elegido_2.iloc[0]
                el_elegido = str(np.asarray(el_elegido_2))

                # el_elegido = str()

            else:
                el_elegido = el_elegido.iloc[0]
                el_elegido = str(np.asarray(el_elegido))
            elegidos_pestana.append(el_elegido)
        elegidos_todos.append(elegidos_pestana)

    elegidos_todos = np.asarray(elegidos_todos)

    distri_elegida = pd.DataFrame(elegidos_todos, index=pestanas, columns=meses)

    distri_elegida.to_excel(libro_salida, sheet_name='PDF')
    libro_salida.close()