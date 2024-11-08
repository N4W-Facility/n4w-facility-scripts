# -*- encoding: utf-8 -*-
# !/usr/bin/python
from pylab import *
import pandas as pd
import scipy.stats as ss
import calendar
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
import os
from datetime import datetime
import statistics
from numpy.polynomial import polynomial as p
from statsmodels.graphics import utils
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import scipy.linalg
from statsmodels.api import OLS
import statsmodels.api as sm

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

def fn_dg2sr2(dg_input, name='Series',tipo='day'):
    """
    This function transforms a dataframe monthly grouped into a time series.
    :param dg_input: pandas dataframe monthly grouped to be transformed.
    :param name: output time series name.
    :return: pandas time series.
    """
    ix_input = dg_input.index
    start_date =datetime(dg_input.index.min(), 1, 1)
    end_date = datetime(dg_input.index.max(), 12, 31)
    sr_try = dg_input.unstack()
    # sr_try.index.levels[0].name = 'jday'
    if tipo =='day':
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
        new = new.join(df_try, how='outer')
        sr_output = new
        sr_output.columns = [name]
    else:
        data = dg_input
        vals = data.values
        years =data.index.values
        nrows = np.size(vals[:, 0])
        ncols = np.size(vals[0, :])
        n_datos = nrows*ncols #np.size(vals[:, 1:])
        rango_fechas = pd.date_range(start_date, end_date, freq='M', name='Fecha')
        df = pd.DataFrame(index=rango_fechas)
        fechas = []
        k = 0
        datos = np.zeros(n_datos)
        for i in range(nrows):
            for j in range(0, ncols):
                datos[k] = vals[i, j]
                fecha_str = '01' + '/' + '{0:02d}'.format(j+1) + '/' + str(int(years[i]))
                # print fecha_str
                fecha = datetime.strptime(fecha_str, '%d/%m/%Y')
                fechas.append(fecha)
                k += 1
        dft = pd.DataFrame(datos, index=fechas, columns=[name])
        df = df.join(dft, how='right')
        sr_output = df

    sr_output = pd.DataFrame(sr_output)
    return sr_output


def transform_to_daily_series(df_input):
    sr_unstacked = df_input.unstack()
    df_reset = sr_unstacked.reset_index()
    df_reset.columns = ['jday', 'year', 'value']

    # Adjust for leap years
    leap_years = {year for year in df_input.index if calendar.isleap(year)}
    mask = (df_reset['year'].isin(leap_years)) & (df_reset['jday'] > 59)
    df_reset.loc[mask, 'jday'] += 1

    df_reset['Date'] = pd.to_datetime(df_reset['year'].astype('str') + df_reset['jday'].astype('str'), format="%Y%j")
    df_reset.set_index('Date', inplace=True)
    df_reset.drop(['year', 'jday'], axis=1, inplace=True)

    start_date = datetime(df_input.index.min(), 1, 1)
    end_date = datetime(df_input.index.max(), 12, 31)
    index_output = pd.date_range(start=start_date, end=end_date, freq='D', name='Date')
    index_output = index_output[~((index_output.month == 2) & (index_output.day == 29))]

    return df_reset.reindex(index_output)
def transform_to_monthly_series(df_input):
    start_date = datetime(df_input.index.min(), 1, 1)
    end_date = datetime(df_input.index.max(), 12, 31)
    rango_fechas = pd.date_range(start_date, end_date, freq='MS', name='Fecha')

    data_array = df_input.values
    years = df_input.index.values
    dates = [datetime(year, month + 1, 1) for year in years for month in range(data_array.shape[1])]
    values = data_array.ravel()

    return pd.DataFrame(values, index=dates, columns=['value']).reindex(rango_fechas)

def fn_dg2sr(df_input, name='Series', tipo='day'):
    if tipo == 'day':
        result = transform_to_daily_series(df_input)
    else:
        result = transform_to_monthly_series(df_input)

    result.columns = [name]
    return result



def Grouptodeseasonalizing(serie,tipo):

    if tipo=='day':
        df_g = fn_sr2dg(serie)
        N = 366
    else:
        df_g = fn_sr2mg(serie)
        N = 13
    data = []
    for i in range(1,N):
        x =  df_g[i]
        mean = np.mean(x)
        s = np.std(x)
        data.append((x - mean) / s)
    df_temp = pd.DataFrame(data).T

    return df_temp

def order_matrix_corr(df_corr, tipo):
    if tipo == 'day':
        n_end = 366
        n_start = -365
    else:
        n_end = 13
        n_start = -12

    # Create a base DataFrame with the desired index
    df_ready = pd.DataFrame(index=pd.Index(range(n_start, n_end)))

    # Generate shifted columns
    shifted_cols = [df_corr[col].shift(periods=-(col - 1)) for col in df_corr.columns]

    # Concatenate all columns
    df_ready = pd.concat([df_ready] + shifted_cols, axis=1)

    # Handle the reversed indices
    for j in df_corr.columns:
        values = df_corr[j][:(j - 1)]
        reversed_idx = np.asarray(sorted(values.index.values, reverse=True)) * -1
        df_ready[j][reversed_idx] = values.values

    df_ready = df_ready.drop(0)
    return df_ready
def filter_lag(df_line, df_error,tipo):
    if tipo == 'day':
        n_end = 366
        n_start = -365
    else:
        n_end = 13
        n_start = -12

    filter_po = df_line.loc[range(1, n_end)].where(df_line > df_error, other=np.nan)
    day_po = filter_po.isnull()[filter_po.isnull()].index.min()
    filter_No = df_line.loc[range(n_start, 0)].where(df_line > df_error, other=np.nan)
    day_ne = filter_No.isnull()[filter_No.isnull()].index.max()
    return day_po, day_ne


def ACF_Groups(df,var,Deseasonalizing = True,type_day = 'day'):
    stations = df.columns

    # Determine the Excel writer based on conditions
    prefix = 'ACF_dst_' if Deseasonalizing else 'ACF_Normal_'
    suffix = 'D' if type_day == 'day' else 'M'
    xls_writer_Acf = pd.ExcelWriter('02_Monthly/06_Inertia/' + prefix + var + '_' + suffix + 'G.xlsx')
    xls_writer_Acf_sig = pd.ExcelWriter('02_Monthly/06_Inertia/ACF_sig_' + prefix + var + '_' + suffix + 'G.xlsx')

    path_graficas = '02_Monthly/06_Inertia/01_Graphics/' + var
    result_rezago = pd.DataFrame(index=stations, columns=['Positivo_Median', 'Negativo_Median', 'Positivo_Mean_std',
                                                          'Negativo_Mean_std'])


    for sta in stations[0:]:
        sta = str(sta)
        print(sta)
        if Deseasonalizing:
            start = df[sta].dropna().index.min()
            df_g = Grouptodeseasonalizing(df[sta].loc[start:], type_day)
        else:
            start = df[sta].dropna().index.min()
            if type_day == 'day':
                df_g  = fn_sr2dg(df[sta].loc[start:])
            else:
                df_g  = fn_sr2mg(df[sta].loc[start:])

        df_g = df_g.interpolate(axis=0)

        mcorr = df_g.corr()

        n_corr = pd.DataFrame(np.where(mcorr.isnull(), 0, mcorr), index=np.arange(1, N),
                              columns=np.arange(1, N))
        cols_0 = df_g[df_g.columns[df_g[df_g.columns].mean() == 0]].columns
        if np.size(cols_0) > 0:
            f_corr = n_corr
            for cz in cols_0:
                f_corr[cz].loc[cz] = 1
        else:
            f_corr = n_corr

        new_df = order_matrix_corr(f_corr, type_day)
        n = df_g.count(axis=0)

        error_M = (1 - f_corr ** 2) / (np.sqrt(n - 1)) * 1.96

        Y = new_df.median(axis=1)
        y_3 = new_df.std(axis=1) + Y

        Line_M = Y  # Mediana
        Line_M_std = y_3  # mediana + std

        error = (1 - Y ** 2) / (np.sqrt(n.max() - 1)) * 1.96

        result_rezago.loc[sta]['Positivo_Median'] = filter_lag(Line_M, error, type_day)[0]  # day_p
        result_rezago.loc[sta]['Negativo_Median'] = filter_lag(Line_M, error, type_day)[1]

        result_rezago.loc[sta]['Positivo_Mean_std'] = filter_lag(Line_M_std, error, type_day)[0]  # day_p
        result_rezago.loc[sta]['Negativo_Mean_std'] = filter_lag(Line_M_std, error, type_day)[1]

        error_pt = (1 - new_df.median(axis=1) ** 2) / (np.sqrt(40 - 1)) * 1.96

        fig = plt.figure(figsize=(9, 7))

        ax1 = fig.add_subplot(111)
        s_n = 12

        dict_tipo_analisis = {'day': [-365, 366, 45, 0.1, 0.1, 'darkseagreen', '-'],
                              'm': [-12, 13, 2, 1.0, 0.5, 'royalblue', '--.']}

        ax1.plot(new_df.index, new_df, dict_tipo_analisis[type_day][6], color=dict_tipo_analisis[type_day][5],
                 linewidth=dict_tipo_analisis[type_day][3], alpha=dict_tipo_analisis[type_day][4])
        ax1.plot(new_df.index, new_df.median(axis=1), color='k', linewidth=1, alpha=0.5, label='Median')
        ax1.fill_between(error_pt.index, error_pt, -1 * error_pt, alpha=0.5, color='dimgrey',
                         label=r"$\varepsilon_{r} = \frac{1-r^2}{\sqrt{n-1} } t_\alpha$")  # 'cyan'
        ax1.legend(loc='upper left', prop={'size': 10})

        ax1.set_xlabel('$Rezagos$', fontsize=s_n)
        ax1.set_ylabel('$ACF$', fontsize=s_n)
        ax1.xaxis.set_ticks(np.arange(dict_tipo_analisis[type_day][0], dict_tipo_analisis[type_day][1],
                                      dict_tipo_analisis[type_day][2]))
        ax1.set_title('$ACF$ ' + dict_names_var[var], fontdict={'fontsize': 13, })
        plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)
        X = new_df.index
        y1 = new_df.median(axis=1)
        y2 = error_pt
        idx = np.argwhere(np.diff(np.sign(y2 - y1))).flatten()
        x = filter_lag(Line_M, error, type_day)[0]
        try:
            ax1.annotate('Rezago Significativo',
                         xytext=(80, 20), xy=(x, y2[idx[1]]), xycoords='data', textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"))
        except:
            continue
        plt.savefig(path_graficas + '/' + type_day + '_Incercia_' + str(sta) + '_' + var + '.png')

        plt.close()

        df_sig = f_corr.where((f_corr >= error_M) & (f_corr > -1 * error_M))

        f_corr.to_excel(xls_writer_Acf, sheet_name=sta, merge_cells=False)
        df_sig.to_excel(xls_writer_Acf_sig, sheet_name=sta, merge_cells=False)
    xls_writer_Acf.close()
    xls_writer_Acf_sig.close()
    return result_rezago

def fn_acorr_plot(x,name, ax=None, lags=None, alfa=.05, use_vlines=True, unbiased=False, fft=False,path_g='', **kwargs):
    """Esta funcion plotea la funcion de autocorrelacion de una serie de tiempo para un numero de lags determinados por el usuario.
    :param x: Arreglo con la serie de tiempo
    :param ax: plot alternativo para superponer la figura generada con esta figura creada previamente
    :param lags: numero maximo de lags evaluados
    :param alfa: Nivel de significancia estadistica de la prueba (0.05 default)
    :param use_vlines: Opcional, boolean. "True" las lineas verticales y los marcadores se plotean, "False" solo los marcadores son ploteados
    :param unbiased: Opciona, boolean. "True" los denominadores para autocovarianza son n-k, "False" seran n
    :param fft: Opciona, boolean. "True" se calcula la funcion ACF a traves de la transformada rapida de Fourier
    **kwargs: Opcional, kwargs. Palabras claves y argumentos que pueden ser pasados de las herramientas de Matplotlib "plot" y "axhline".
    Returns: Una figura de Matplotlib, si ax=None entonces la figura es creada, sino el grafico se conecta a la figura ax incluida

    Notes: Adaptado de codigo de statsmodels, que a su vez se adapto de Matplotlib "xcorr"

    """
    plt.style.use('bmh')
    fig, ax = utils.create_mpl_ax(ax)

    if lags is None:
        lags = np.arange(len(x.dropna()))
        nlags = len(lags) - 1
    else:
        nlags = lags
        lags = np.arange(lags + 1)  # Se adiciona en uno por el rezago cero

    acf_x, confint = acf(x.dropna(), nlags=nlags, alpha=alfa, fft=fft)

    if use_vlines:
        ax.vlines(lags, [0], acf_x,color='black', alpha=0.9)
        ax.axhline(**kwargs)

    # center the confidence interval
    confint = confint - confint.mean(1)[:, None]
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 2)
    kwargs.setdefault('linestyle', 'None')
    kwargs.setdefault('color', 'Black')
    # ax.margins(.05)
    ax.plot(lags, acf_x, **kwargs)
    ax.fill_between(lags, confint[:, 0], confint[:, 1], alpha=0.25, label=u'Intervalo de confianza',color='dimgrey')
    plt.plot([0, lags.max()], [0, 0], color='grey', linewidth='0.5')
    plt.plot([0, 0], [-0.5, 1], color='grey', linewidth='0.5')

    plt.title('Autocorr - {}'.format(name))



    plt.xlabel('$Lags$')
    plt.ylabel('$ACF$')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(path_g+'Autocorr_{}'.format(name), dpi=150)
    plt.close()

    acf_signif = np.where(np.abs(acf_x) > np.abs(confint[:, 0]), acf_x, np.nan)



    return pd.Series(data=acf_x, index=lags, name=name), pd.Series(data=acf_signif, index=lags, name=name)

def ACF_series(df_data, var, type_day, Deseasonalizing, nlags):
    """
    Genera archivos Excel con ACF y ACF significativos.

    Parámetros:
    - df_data: DataFrame con los datos.
    - var: Variable para el nombre del archivo.
    - type_day: Tipo de día ('day' o otro valor).
    - Deseasonalizing: Booleano para determinar si se desestacionaliza.
    - nlags: Número de rezagos.
    - Grouptodeseasonalizing: Función para desestacionalizar.
    - fn_dg2sr: Función para transformar datos.
    - fn_acorr_plot: Función para plotear ACF.

    Retorna:
    None
    """

    prefix = 'ACF_dst_' if Deseasonalizing else 'ACF_Normal_'
    suffix = 'D' if type_day == 'day' else 'M'
    xls_writer_Acf = pd.ExcelWriter('02_Monthly/06_Inertia/' + prefix + var + '_' + suffix + '_S.xlsx')
    xls_writer_Acf_sig = pd.ExcelWriter('02_Monthly/06_Inertia/Sig_' + prefix + var + '_' + suffix + '_S.xlsx')

    path_graficas = '02_Monthly/06_Inertia/01_Graphics/' + var + '/'+type_day+'_'

    stations = df_data.columns
    result_rezago = pd.DataFrame(index=stations, )

    df_acorr = pd.DataFrame(index=range(0, nlags))
    df_acorr_signf = pd.DataFrame(index=range(0, nlags))
    for sta in stations[0:]:
        print(sta)

        if Deseasonalizing:
            start = df_data[sta].dropna().index.min()
            df = Grouptodeseasonalizing(df_data[sta].loc[start:], type_day)
            serie = fn_dg2sr(df, name=sta, tipo=type_day)
        else:
            start = df_data[sta].dropna().index.min()
            serie = df_data[sta].loc[start]

        alpha = 0.05
        acorr, acorr_signf = fn_acorr_plot(serie, str(sta), lags=nlags, alpha=alpha, path_g=path_graficas)

        # rezago = filter_lag(acorr,acorr_signf, type_day)[0]
        df_acorr[sta] = acorr
        df_acorr_signf[sta] = acorr_signf
        # result_rezago.loc[sta] =[df_acorr_signf [df_acorr_signf [sta].notna()].index.tolist()]
        list = df_acorr_signf[df_acorr_signf[sta].notna()].index.tolist()
        result_rezago.loc[sta, sta] = str(list)
    df_acorr.to_excel(xls_writer_Acf, sheet_name='ACF', merge_cells=True)
    df_acorr_signf.to_excel(xls_writer_Acf_sig, sheet_name='ACF_signif', merge_cells=True)
    xls_writer_Acf.close()
    xls_writer_Acf_sig.close()

    return result_rezago

if __name__ == "__main__":


    os.chdir(r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization')

    type_day = 'm'#''m #'day'#
    Deseasonalizing = True #True #
    tipo = 'Grupo' #'Serie' #

    if type_day=='day':

        in_file_variable = '04_Datos_Sel_sin_outliers.xlsx'
        N = 366
        nlags = 366*2  # 72

    else:
        in_file_variable = '02_Monthly/05_Monthly_Series.xlsx'
        N = 13
        nlags = 72

    dict_names_var = {'PT_4':'Precipitacion Total','TS_1':'Temperatura Media',
                      'TS_2':'Temperatura Max', 'TS_3':'Temperatura Min',
                      'EV_4': 'Evaporacion Total','BS_4': 'Brillo Solar',
                      'QL_1':'Caudal Medio'
                      }
    excel_salida = '02_Monthly/06_Inertia/01_Graphics/'

    file_data = pd.ExcelFile(in_file_variable)

    os.makedirs('02_Monthly/06_Inertia/',exist_ok= True )

    xls_writer = pd.ExcelWriter('02_Monthly/06_Inertia/' + 'Significant_lags.xlsx')

    variables = ['PT_4', 'TS_2', 'TS_3',  'QL_1']# 'EV_4','BS_4',
    for var in variables[0:]:
        if not os.path.exists('02_Monthly/06_Inertia/01_Graphics/'+var):
            os.makedirs('02_Monthly/06_Inertia/01_Graphics/'+var)

        print(var)

        df_data = file_data.parse(var, index_col=0)

        if tipo=='Grupo':

            rezago = ACF_Groups(df_data, var, Deseasonalizing=Deseasonalizing, type_day=type_day)

        else:
            rezago = ACF_series(df_data, var, type_day, Deseasonalizing, nlags)

        rezago.to_excel(xls_writer, sheet_name=var, merge_cells=False)
    xls_writer.close()