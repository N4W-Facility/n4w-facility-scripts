import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
import os

def rem0229(ix_input):
    """
    This fucntion removes february 29th from a daily index.
    :param ix_input:
    :return:
    """
    ix_output = ix_input.drop(ix_input[(ix_input.month == 2) & (ix_input.day == 29)])
    return ix_output
def fn_sr2dg(sr_input):
    """
    This function transforms a time series into a dataframe daily grouped.
    :param sr_input:
    :return:
    """
    name = sr_input.name
    ix_input = sr_input.index
    ix_rem0229 = rem0229(ix_input)
    # sr_rem0229 = sr_input.loc[ix_rem0229]
    df_dg = pd.DataFrame(index=ix_rem0229, columns=['year', 'day', name])
    df_dg['year'] = ix_rem0229.year
    df_dg['day'] = pd.Series(ix_rem0229.strftime('%j').astype(int), index=ix_rem0229, name='jday')
    df_dg[name] = sr_input.loc[ix_rem0229]
    leap_years = list({year for year in ix_rem0229.year if calendar.isleap(year)})

    ix_correct = ix_rem0229[df_dg['year'].isin(leap_years) & (ix_rem0229.month > 2)]
    df_dg.loc[ix_correct, 'day'] = df_dg['day'] - 1
    # df_dg.to_clipboard()
    dg_output = df_dg.pivot(index='year', columns='day', values=name)

    return dg_output
def daily_pivot(series):
    df_serie = pd.DataFrame(series)
    df_serie.columns = ['Data']
    df_serie['Year'] = pd.to_datetime(df_serie.index).year
    nyears = ((df_serie['Year'].max() - df_serie['Year'].min()) + 1)
    month_start = df_serie.index.month[0]
    if month_start > 2:
        start = df_serie.index.dayofyear[0] - 2
        end = (df_serie.index.dayofyear[0] - 2) + len(df_serie)
    else:
        start = df_serie.index.dayofyear[0] - 1
        end = (df_serie.index.dayofyear[0] - 1) + len(df_serie)
    df_serie['DayJulian'] = (nyears * range(1, 366))[start:end]
    df_daily = df_serie.pivot(index='Year', columns='DayJulian', values='Data')
    return df_daily

"""
Este codigo transforma las series a grupo y hace parte del algoritmo para la determinacion de outliers por la metodologia
de grubbs

"""


path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

path_out= r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

name = '02_Selected_data.xlsx'

file = pd. ExcelFile(path+name)

variables =file.sheet_names

fecha_in = '1/01/1970'
fecha_fin = '31/12/2024'

for var in variables:

    print(var)

    if not os.path.exists(path_out + '01_Outliers/'+'01_Grupos/'):
        os.makedirs(path_out + '01_Outliers/'+'01_Grupos/' )

    xls_salida = path_out + '01_Outliers/'+'01_Grupos/'+var+'_select.xlsx'
    xls_writer = pd.ExcelWriter(xls_salida)

    data = file.parse(var, index_col=0)

    stations = data.columns

    for sta in stations :
        print(sta)


        datos = fn_sr2dg(data[sta].loc[fecha_in:fecha_fin])

        datos = datos.dropna(how='all')
        datos.to_excel(xls_writer, sheet_name =str(sta), merge_cells=False)
    xls_writer.close()


