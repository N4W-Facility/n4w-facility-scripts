import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
import warnings
warnings.filterwarnings('ignore')



def fn_dg2sr(dg_input,fecha_inicial,fecha_final, name='Series'):
    """
    This function transforms a dataframe monthly grouped into a time series.
    :param dg_input: pandas dataframe monthly grouped to be transformed.
    :param name: output time series name.
    :return: pandas time series.
    """
    ix_input = dg_input.index
    start_date =fecha_inicial #pd.datetime(dg_input.index.min(), 1, 1)
    end_date = fecha_final#pd.datetime(dg_input.index.max(), 12, 31)
    sr_try = dg_input.unstack()
    # sr_try.index.levels[0].name = 'jday'
    df_try = sr_try.reset_index()
    df_try.columns = ['jday', 'year', 0]
    leap_years = list({year for year in ix_input if calendar.isleap(year)})
    df_try['jday'][(df_try['year'].isin(leap_years)) & (df_try['jday'] > 59)] += 1
    df_try['Date'] = pd.to_datetime(df_try['year'].astype('str') + df_try['jday'].astype('str'), format="%Y%j")
    df_try.set_index(df_try['Date'], inplace=True)
    df_try.drop(['year', 'jday', 'Date'], axis=1, inplace=True)
    df_try.sort_index(inplace=True)
    # index_output = pd.DatetimeIndex(freq='D', start=start_date, end=end_date, name='Date')
    index_output = pd.date_range(start=start_date, end=end_date,freq='D', name='Date')
    index_output = index_output[~((index_output.month == 2) & (index_output.day == 29))]
    new = pd.DataFrame(index=index_output)
    # sr_output = df_try.loc[index_output, 0]
    new = new.join(df_try,how='outer')
    sr_output = new
    sr_output.columns = [name]
    sr_output = pd.DataFrame(sr_output)
    return sr_output

def dias_series(filename_in,fecha_inicial,fecha_final):
    xls = pd.ExcelFile(filename_in)
    cols = xls.sheet_names
    data = xls.parse(cols[0],index_col=0)
    df_out = fn_dg2sr(data,fecha_inicial,fecha_final,name=cols[0])
    for h in cols[1:]:
        print ('leyendo hoja de Excel:', h)
        try:
            datos = xls.parse(h,index_col=0)
            dft=fn_dg2sr(datos,fecha_inicial,fecha_final,name=h)
            df_out = df_out.join(dft, how='outer')
        except:
            print ('No se Puede hacer la hoja:')

    return df_out


path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

path_out= r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'



xls_writer = pd.ExcelWriter(path_out+'04_Datos_Sel_sin_outliers.xlsx') # SALIDA

variables = ['PT_4','TS_2','TS_3','RS','Vwind','Uwind','HR_1','QL_1']# ENTRADA

fecha_min = '1/01/1970'
fecha_max = '31/12/2024'
for var in variables[0:]:
    print(var)

    name = path + '01_Outliers/'+'02_Outliers/' + var + '_G_limpios.xlsx'#'_Outliers_fill.xlsx'

    df_series = dias_series(name,fecha_min,fecha_max)
    df_series = df_series.loc[fecha_min:fecha_max]#.fillna(-99.0)

    df_series.to_excel(xls_writer, sheet_name=var, merge_cells=False)

xls_writer.close()