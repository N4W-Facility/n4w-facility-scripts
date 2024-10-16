# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

variables = ['PT_4','TS_2','TS_3','RS','Vwind','Uwind','HR_1','QL_1']# ENTRADA

path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly\01_Groups/'

path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/'

if not os.path.exists(path_out + '02_Randomness_M/'):
    os.makedirs(path_out + '02_Randomness_M/')

for v in variables[0:]:

    excel_entrada =path +v+'_groups_M.xlsx'
    excel_salida = path_out + '02_Randomness_M/'+ 'Randomness_M_'+v+'.xlsx'


    libro = pd.ExcelFile(excel_entrada)
    pestanas = libro.sheet_names[0:]
    print (pestanas)

    Rachas_Test = pd.DataFrame()
    Resumen_Rachas = pd.DataFrame()
    libro = pd.ExcelWriter(excel_salida)


    for p in pestanas:
        print ('Trabajando pestaña:', p)
        datos = pd.read_excel(excel_entrada, p, index_col=0)
        mindex = datos.index.values
        promedios = datos.mean()
        print (datos.to_string())
        signos = datos.where(datos >= promedios, other='-')
        signos = signos.where(datos < promedios, other='+')
        print (signos.to_string())

        signos_ini = signos[0:-1]
        signos_fin = signos[1:]
        cols = signos_ini.columns
        Re = pd.DataFrame(np.where(signos_fin.values != signos_ini.values,1, 0), columns=cols).sum() + 1

        n = datos.count()
        print ('**********')
        print ('Tamaño del grupo:', n, np.size(n))
        Rt = (n + 1)/2.0
        Std_Rt = np.sqrt(n - 1)/2
        talfa = 1.96
        res_Test = np.logical_and(Re >= Rt - talfa*Std_Rt, Re <= Rt + talfa*Std_Rt)
        res_Test = np.where(res_Test, 'ACEPTADA', 'RECHAZADA')
        Rachas_Test['n'] = n
        Rachas_Test['Re'] = Re
        Rachas_Test['Rt'] = Rt
        Rachas_Test['Std_Rt'] = Std_Rt
        Rachas_Test['Limite Inferior'] = Rt - talfa*Std_Rt
        Rachas_Test['Limite Superior'] = Rt + talfa*Std_Rt
        Rachas_Test['Ho'] = res_Test
        Rachas_Test.to_excel(libro, sheet_name=p, merge_cells=False)
        print (Rachas_Test.to_string())
        Resumen_Rachas[p] = res_Test
    Resumen_Rachas.reindex(index=mindex)
    Resumen_Rachas.to_excel(libro, sheet_name='Resumen Aleatoriedad', merge_cells=False)
    # libro.save()
    libro.close()