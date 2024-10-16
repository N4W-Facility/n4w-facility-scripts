import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



def fn_sr2mg(sr_ts):
    """
    This function transforms a time series into a dataframe monthly grouped.
    :param sr_ts: pandas time series to be transformed.
    :return: pandas dataframe monthly grouped.
    """
    df_data = pd.DataFrame(sr_ts)
    df_data['Year'] = df_data.index.year
    df_data['month'] = df_data.index.month
    df_mg = df_data.pivot(index='Year', columns='month', values=sr_ts.name)
    return df_mg


path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/'

path_out= r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/'

if not os.path.exists(path_out +  '01_Groups/'):
    os.makedirs(path_out +  '01_Groups/')

xls_data = path+'05_Monthly_Series.xlsx'

file = pd.ExcelFile(xls_data)

variables =file.sheet_names # ('QL','PT_4','TS_1','TS_2','TS_8','EV_4','QL_1','QL_2','QL_3')

for v in variables[0:]:
    print (v)
    xls_salida = path_out +  '01_Groups/'+v+'_groups_M.xlsx'
    xls_writer = pd.ExcelWriter(xls_salida)
    data = file.parse(v, index_col =0)#.interpolate().fillna(method='bfill')

    for c in data.columns[0:]:
        print (c)
        datos = fn_sr2mg(data[c])

        datos = datos.dropna(how='all')

        datos.to_excel(xls_writer, sheet_name = str(c), merge_cells=False)

    xls_writer.close()