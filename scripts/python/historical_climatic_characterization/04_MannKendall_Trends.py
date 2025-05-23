# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:16:06 2015

@author_Funcion MK: Michael Schramm

"""


import numpy as np
from scipy.stats import norm
import pandas as pd
import os
from pandas import Series



def mk_t(x, alpha=0.05):
    """
    This function is derived from code originally posted by Sat Kumar Tomer
    (satkumartomer@gmail.com)
    See also: http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm

    The purpose of the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert
    1987) is to statistically assess if there is a monotonic upward or downward
    trend of the variable of interest over time. A monotonic upward (downward)
    trend means that the variable consistently increases (decreases) through
    time, but the trend may or may not be linear. The MK test can be used in
    place of a parametric linear regression analysis, which can be used to test
    if the slope of the estimated linear regression line is different from
    zero. The regression analysis requires that the residuals from the fitted
    regression line be normally distributed; an assumption not required by the
    MK test, that is, the MK test is a non-parametric (distribution-free) test.
    Hirsch, Slack and Smith (1982, page 107) indicate that the MK test is best
    viewed as an exploratory analysis and is most appropriately used to
    identify stations where changes are significant or of large magnitude and
    to quantify these findings.

    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics

    Examples
    --------


    """
    n = len(x)

    # calculate S
    s = 0
    x = np.append(x,x[n-1:n])
    for j in range(0,n):
        s += np.sign(x[j] - x[j+1])
    # for k in range(n-1):
    #     for j in range(k+1, n):
    #         s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(x == unique_x[i])
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    else: # s == 0:
        z = 0

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z






path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'


xls_file = path + '04_Datos_Sel_sin_outliers.xlsx'

file_data = pd.ExcelFile(xls_file)

variables = file_data.sheet_names


os.makedirs(path_out + '02_Trends/',exist_ok=True )

excel_salida = path_out + '02_Trends/' + 'Tendencias_Mk.xlsx'
#'D:\Dropbox\TESIS\Proyecto_Universidad_Javeriana_Cenigaa_2018\Trabajo_2019\Datos\Datos_Ideam\TSA\Trends/Tendencias_Mk.xlsx'

xls_writer = pd.ExcelWriter(excel_salida)

for v in variables[0:]:
    print (v)

    data = file_data.parse(v,index_col=0)

    pestanas = data.columns[0:]

    result_list = []
    for p in pestanas[0:]:

        try:

            print('Trabajando pestaña:', p)
            x = data[p] #data[data.index.year == 1981][p]#
            x = x[np.logical_not(np.isnan(x))]
            trend,h,p_v,z = mk_t(x)
        except:
            trend, h, p_v, z = np.nan,  np.nan,  np.nan,  np.nan
        result_list.append([p,trend,h,p_v,z])

    col_names = ['Estacion', 'Tendencia', 'Hipotesis', 'p value', 'normalized test statistics']
    result = pd.DataFrame(result_list, columns=col_names)

    result.to_excel(xls_writer, sheet_name=str(v), merge_cells=False)

xls_writer.close()
