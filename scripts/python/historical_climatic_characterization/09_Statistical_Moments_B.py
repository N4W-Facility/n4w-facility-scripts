# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import scipy.stats as ss

path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/' # general donde esta toda la informacion

path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/'

type_m = 'm1'

excel_salida = path_out + '04_Seasonality/'+'/Summary_'+type_m+'.xlsx'

xls_writer = pd.ExcelWriter(excel_salida)

variables = ['PT_4','QL_1','TS_2','TS_3','RS','Vwind','Uwind','HR_1','QL_1']##'BS_4','EV_4'

for v in variables[0:]:
    print(v)

    excel_entrada = path_out + '04_Seasonality/'+'01_Moments/'+'/Moments_'+v+'.xlsx'

    libro = pd.ExcelFile(excel_entrada)

    pestanas = libro.sheet_names

    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'jul', 'Ago', 'Sep', 'Oct',
             'Nov', 'Dic']

    for p in pestanas[0:]:
        print(p)
        datos = pd.read_excel(excel_entrada, p, index_col=0)
        datos.index = meses

        df = pd.DataFrame(index=pestanas, columns=meses)
        df.loc[p] = datos['m1'].T

        df.to_excel(xls_writer, sheet_name=v, merge_cells=False)
xls_writer.close()