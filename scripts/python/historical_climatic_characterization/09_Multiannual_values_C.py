import numpy as np
import pandas as pd
import os


path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization/'

path_out= r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/'

variables = ['QL_1','TS_2','TS_3','RS','Vwind','Uwind','HR_1','PT_4']



meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'jul', 'Ago', 'Sep', 'Oct',
             'Nov', 'Dic']


if not os.path.exists(path_out + '04_Seasonality/'):
    os.makedirs(path_out +'04_Seasonality/')

xls_data_salida = path_out + '04_Seasonality/'+'01_monthly_multiannuals.xlsx'
xls_salida = pd.ExcelWriter(xls_data_salida )
for v in variables[0:]:
    print(v)

    xls_data = path +'02_Monthly/01_Groups/'+ v+'_groups_M.xlsx'

    temp = pd.ExcelFile(xls_data)

    hojas = temp.sheet_names

    df = pd.DataFrame(index=hojas, columns=meses)

    dfmax = pd.DataFrame(index=hojas, columns=meses)

    for sta in hojas[0:]:
        data = temp.parse(sta, index_col=0)
        data.columns = meses

        df_month = data.mean(axis=0)

        df.loc[sta] = df_month.T

        df_month_max = data.std(axis=0)

        dfmax.loc[sta] = df_month_max.T
    df.to_excel(xls_salida, sheet_name=v, merge_cells=False)
xls_salida.close()