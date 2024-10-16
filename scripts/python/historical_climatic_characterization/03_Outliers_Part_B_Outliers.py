import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, zscore
import scipy.stats as ss
import os

def fn_grubbs(df_input, alpha=0.01, two_tail=True):
    """
    This function applies the Grubbs' Test for outliers in a dataframe and returns two dataframes, the first one
    without outliers and the second one just with the outliers
    :param df_input: Pandas dataframe with series to test.
    :param alpha: Significance level [1% as default].
    :param two_tail: Two tailed distribution [True as default].
    :return: tuple with two dataframes, the first one without outliers and the second one just for outliers.
    """
    df_try = df_input.copy()
    df_output = pd.DataFrame(index=df_input.index, columns=df_input.columns)
    df_outliers = pd.DataFrame(data=0, index=df_input.index, columns=df_input.columns)
    if two_tail:
        alpha /= 2

    while not df_outliers.isnull().values.all():
        mean = df_try.mean()
        std = df_try.std()
        n = len(df_try)
        tcrit = ss.t.ppf(1 - (alpha / (2 * n)), n - 2)
        #print tcrit
        zcrit = (n - 1) * tcrit / (n * (n - 2 + tcrit ** 2)) ** .5
        df_outliers = df_try.where(((df_try - mean) / std) > zcrit)
        df_output.update(df_input[df_outliers.isnull() == False])
        df_try = df_try[df_outliers.isnull()]

    return df_try, df_output

def Groobs(data):
    promedios = data.mean()
    desviaciones = data.std()
    talfa = 1.96
    deltas = np.abs(data - promedios)
    data_sin = data.where(deltas <= talfa*desviaciones, other=np.nan)

    anomal_location = []
    for c in data.columns:
        dat = data_sin[c]

        anomal_loc = list(set(dat.index.values) - set(dat.dropna().index.values))

        try:
            anomal_location.append((len(anomal_loc) / data_sin[c].dropna().size))
        except:
            anomal_location.append(0.0)


    return data_sin, anomal_location


"""
Este codigo  determinar los valores outliers mediante la metodologia grubbs, como entrada tiene los grupos diarios por variable
y salida genera:
1. grupo de outliers
2. grupo de limpios
3. porcentaje de anomalos por estacion 
"""

path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

path_out= r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

variables = ['PT_4','TS_2','TS_3','RS','Vwind','Uwind','HR_1','QL_1']# ENTRADA

if not os.path.exists(path_out + '01_Outliers/'+'02_Outliers/'):
    os.makedirs(path_out + '01_Outliers/'+'02_Outliers/')

excel_salida_1 = pd.ExcelWriter(path_out + '01_Outliers/'+'02_Outliers/' + '01_Percentage_of_anomalous.xlsx')
for var in variables[0:]:
    print(var)



    xls_g = path  + '01_Outliers/'+'01_Grupos/' +var+'_select.xlsx' # ENTRADA

    excel_salida = pd.ExcelWriter(path_out + '01_Outliers/'+'02_Outliers/' + var + '_G_limpios.xlsx')
    excel_salida_out = pd.ExcelWriter(path_out + '01_Outliers/'+'02_Outliers/'+ var + '_G_outliers.xlsx')

    file = pd.ExcelFile(xls_g )

    stations = file.sheet_names

    df_result = pd.DataFrame(columns=stations)
    for sta in stations:

        try:
            print('Leyendo Estacion: ', sta)
            datos = file.parse(sta,index_col=0)

            datos_sin, outliers = fn_grubbs(datos) #Groobs(datos)

            anomal_loc = outliers.count()/datos.count()

            df_result[sta] = anomal_loc #.T[0]
            datos_sin.to_excel(excel_salida, sheet_name=sta, merge_cells=False)
            outliers.to_excel(excel_salida_out, sheet_name=sta, merge_cells=False)

        except:
            print('No se puede hacer la estacion: ',sta )
    df_result.to_excel(excel_salida_1, sheet_name=var, merge_cells=False)

    excel_salida.close()
    excel_salida_out.close()
excel_salida_1.close()