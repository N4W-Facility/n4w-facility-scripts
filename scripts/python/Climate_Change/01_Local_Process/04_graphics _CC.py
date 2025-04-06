# -*- coding: utf-8 -*-

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

def read_all_xlsfiles_m(modelos,escenarios,path,var):


    list_file = {}
    esc = escenarios
    for mn in modelos[0:]:
        print(mn)
        file = pd.ExcelFile(path + 'Monthly_Series_' + mn + '_' + var + '.xlsx')
        list_file[mn] = {
            esc[0]: file.parse(esc[0], index_col=0).loc[:'31-12-2100'],
            esc[1]: file.parse(esc[1], index_col=0).loc[:'31-12-2100'],
            esc[2]: file.parse(esc[2], index_col=0).loc[:'31-12-2100'],
            esc[3]: file.parse(esc[3], index_col=0).loc[:'31-12-2100'],
            esc[4]: file.parse(esc[4], index_col=0).loc[:'31-12-2100'],
            esc[5]: file.parse(esc[5], index_col=0).loc[:'31-12-2100']}
    return list_file

def read_all_xlsfiles_day(modelos,escenarios,path,var):


    list_file = {}
    esc = escenarios
    for mn in modelos[0:]:
        new_path = path+mn+'/'+var+'/'
        list_file[mn] = {esc[0]: pd.read_csv(path + '01_Series_historical-obs_' +  var + '.csv', index_col=0).loc[:'31-12-2100'],
            esc[1]: pd.read_csv(new_path  + '02_Ds_Series_historical-gcm_' +  var + '.csv', index_col=0).loc[:'31-12-2100'],
            esc[2]: pd.read_csv(new_path  + '03_Ds_Series_ssp126_' +  var + '.csv', index_col=0).loc[:'31-12-2100'],
            esc[3]: pd.read_csv(new_path  + '04_Ds_Series_ssp245_' +  var + '.csv', index_col=0).loc[:'31-12-2100'],
            esc[4]: pd.read_csv(new_path  + '05_Ds_Series_ssp370_' +  var + '.csv', index_col=0).loc[:'31-12-2100'],
            esc[5]: pd.read_csv(new_path  + '06_Ds_Series_ssp585_' +  var + '.csv', index_col=0).loc[:'31-12-2100']}
    return list_file

def add_leap(df,tipo='presente'):
    if tipo=='presente':
        start = '2000-01-01'
        end = '2020-12-31'
    else:
        start = '2015-01-01'
        end = '2100-12-31'
    rango_fechas = pd.date_range(start=start, end=end,freq='D')
    df_fechas = pd.DataFrame(index=rango_fechas)
    df.index = pd.to_datetime(df.index).date
    df_result = pd.merge(df_fechas, df, left_index=True, right_index=True, how="left")
    df_result.interpolate(method='linear', inplace=True)
    return df_result

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

def series_grafica_m(sta,variable, all_result_daily, modelos, escenarios,path_out):
    new_models = [str(ee) + '_' + str(nn) for ee in escenarios[2:] for nn in modelos]
    df_result = pd.DataFrame(columns=new_models, index=list(range(1,13)))
    df_result_26 = pd.DataFrame(columns=models)
    df_result_37 = pd.DataFrame(columns=models)
    df_result_45 = pd.DataFrame(columns=models)
    df_result_85 = pd.DataFrame(columns=models)
    df_result_histgcm = pd.DataFrame(columns=models)

    def agregacion(df,variable):

        if variable == 'PT_4':
            df_m = df.resample('M').sum()
            grupos = fn_sr2mg(df_m)


        elif variable == 'TS_1':
            df_m = df.resample('M').mean()
            grupos = fn_sr2mg(df_m)


        return grupos.mean()

    for m, file in all_result_daily.items():
        # print(m)

        df_his_gcm = add_leap(file['historical-gcm'])[sta]
        df_his = add_leap(file['historical-obs'])[sta]
        df_ssp126 = add_leap(file['ssp126'],'futuro')[sta]
        df_ssp245 = add_leap(file['ssp245'], 'futuro')[sta]
        df_ssp370 = add_leap(file['ssp370'], 'futuro')[sta]
        df_ssp585 =  add_leap(file['ssp585'],'futuro')[sta]


        his_gcm = agregacion(df_his_gcm,var)
        his = agregacion(df_his,var)
        ssp126 = agregacion(df_ssp126,var)
        ssp245 =agregacion(df_ssp245,var)
        ssp370 = agregacion(df_ssp370,var)
        ssp585 = agregacion(df_ssp585,var)


        df_result[str('ssp126') + '_' + str(m)] = ssp126
        df_result[str('ssp245') + '_' + str(m)] = ssp245
        df_result[str('ssp370') + '_' + str(m)] = ssp370
        df_result[str('ssp585') + '_' + str(m)] = ssp585

        df_result_26[m] = ssp126
        df_result_37[m] = ssp370
        df_result_45[m] = ssp245
        df_result_85[m] = ssp585
        df_result_histgcm[m] = his_gcm

    df_result.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    df_result_26.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    df_result_37.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    df_result_45.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    df_result_85.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    figsize = (13, 9)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    dict_momento = {'m1': 'Esperanza Matematica (Estacionalidad)'}
    size_line = 1.5
    type_line = 'None'  # '-'#
    # plt.bar(hist_gcm.index, hist_gcm, color='black', alpha=.7, label='Historico')


    max_pt = df_result.max(axis=1)
    min_pt = df_result.min(axis=1)
    meansp26 = df_result_26.mean(axis=1)
    meansp37 = df_result_37.mean(axis=1)
    meansp45 = df_result_45.mean(axis=1)
    meansp85 = df_result_85.mean(axis=1)
    histor = his_gcm
    histor.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    if var == 'PT_4':
        plt.bar(histor.index, histor, color='black', alpha=.7, label='Historico')
    else:
        ax.plot(histor.index, histor,
                     c='black', linewidth=2.5, linestyle='--', alpha=0.9, label='Historico')


    ax.plot(meansp26.index, meansp26, c='green', linewidth=size_line,
                 alpha=0.9, linestyle=type_line, marker='o', markersize=5, label='ssp126')
    ax.plot(meansp45.index, meansp45, c='blue', linewidth=size_line,
                 alpha=0.9, linestyle=type_line, marker="v", markersize=5, label='ssp245')
    ax.plot(meansp37.index, meansp37, c='yellow', linewidth=size_line,
                 alpha=0.9, linestyle=type_line, marker="^", markersize=5, label='ss370')
    ax.plot(meansp85.index, meansp85, c='red', linewidth=size_line,
                 alpha=0.9, linestyle=type_line, marker="p", markersize=5, label='ssp585')

    ax.fill_between(max_pt.index, max_pt.values,
                         min_pt.values, color='gray', alpha=.5, label='GCMs ' + 'm1')

    s_n = 12
    dict_var_g = {'TS_1': ['Temperatura Media', 'TS', 'C'], 'PT_4': ['Precipitacion total', 'PT', 'mm']}
    ax.set_title(dict_momento['m1'], fontdict={'fontsize': 10, 'fontweight': "bold"})
    ax.set_ylabel('{} ({})'.format(dict_var_g[var][1], dict_var_g[var][2]), fontsize=s_n, fontweight="bold")
    ax.set_xlabel('Meses', fontsize=s_n, fontweight="bold")  # '$Meses$'
    plt.grid()
    plt.legend(loc=1,prop={'size': 12})
    fig.subplots_adjust(hspace=.001, wspace=0.15)

    plt.savefig(path_out + '/Grafica_Estacional_' + str(sta) + '_' + var, dpi=300)
    plt.close()

def series_grafica(sta,variable, all_result_daily, modelos, escenarios,path_out):
    new_models = [str(ee) + '_' + str(nn) for ee in escenarios[2:] for nn in modelos]
    new_index = pd.Series(pd.date_range('1985-12-31', '2100-12-31', freq="Y"))
    df_result = pd.DataFrame(columns=new_models, index=new_index)
    df_result_26 = pd.DataFrame(columns=models)
    df_result_37 = pd.DataFrame(columns=models)
    df_result_45 = pd.DataFrame(columns=models)
    df_result_85 = pd.DataFrame(columns=models)
    df_result_histgcm = pd.DataFrame(columns=models)

    for m, file in all_result_daily.items():
        # print(m)

        df_his_gcm = add_leap(file['historical-gcm'])[sta]
        df_his = add_leap(file['historical-obs'])[sta]
        df_ssp126 = add_leap(file['ssp126'],'futuro')[sta]
        df_ssp245 = add_leap(file['ssp245'], 'futuro')[sta]
        df_ssp370 = add_leap(file['ssp370'], 'futuro')[sta]
        df_ssp585 =  add_leap(file['ssp585'],'futuro')[sta]

        if variable == 'PT_4':
            his_gcm = df_his_gcm.resample('Y').sum()
            his = df_his.resample('Y').sum()
            ssp126 = df_ssp126.resample('Y').sum()
            ssp245 = df_ssp245.resample('Y').sum()
            ssp370 = df_ssp370.resample('Y').sum()
            ssp585 = df_ssp585.resample('Y').sum()
        elif variable == 'TS_1':
            his_gcm = df_his_gcm.resample('Y').mean()
            his = df_his.resample('Y').mean()
            ssp126 = df_ssp126.resample('Y').mean()
            ssp245 = df_ssp245.resample('Y').mean()
            ssp370 = df_ssp370.resample('Y').mean()
            ssp585 = df_ssp585.resample('Y').mean()

        df_result[str('ssp126') + '_' + str(m)] = ssp126
        df_result[str('ssp245') + '_' + str(m)] = ssp245
        df_result[str('ssp370') + '_' + str(m)] = ssp370
        df_result[str('ssp585') + '_' + str(m)] = ssp585
        df_result_26[m] = ssp126
        df_result_37[m] = ssp370
        df_result_45[m] = ssp245
        df_result_85[m] = ssp585
        df_result_histgcm[m] = his_gcm

    figsize = (13, 9)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    max_pt = df_result.max(axis=1)
    min_pt = df_result.min(axis=1)
    meansp26 = df_result_26.mean(axis=1)
    meansp37 = df_result_37.mean(axis=1)
    meansp45 = df_result_45.mean(axis=1)
    meansp85 = df_result_85.mean(axis=1)

    data_hist_p = his_gcm[0:]   #his[0:]  # control de la serie
        # data_hist_p = df_result_histgcm.mean(axis=1)
    min_index = data_hist_p.index.min()
    temp_126 = pd.concat([data_hist_p.loc[min_index:'31-12-2014'], meansp26])  # data_hist_p.loc[min_index:].append(meansp26.loc['31/12/2014':])
    temp_37 = pd.concat([data_hist_p.loc[min_index:'31-12-2014'], meansp37])
    temp_45 = pd.concat([data_hist_p.loc[min_index:'31-12-2014'], meansp45])# data_hist_p.loc[min_index:].append(meansp45.loc['31/12/2014':])
    temp_85 = pd.concat([data_hist_p.loc[min_index:'31-12-2014'], meansp85])


    ax.plot(temp_126.index, temp_126, c='green', linewidth=0.8,
                alpha=0.9, label='ssp126')
    ax.plot(temp_37.index, temp_37, c='yellow', linewidth=0.8,
                alpha=0.9, label='ssp370')
    ax.plot(temp_45.index, temp_45, c='blue', linewidth=0.8,
                alpha=0.9, label='ssp245')
    ax.plot(temp_85.index, temp_85, c='red', linewidth=0.8,
                alpha=0.9, label='ssp585')


    ax.plot(data_hist_p.loc[min_index:'31-12-2014'].index, data_hist_p.loc[min_index:'31-12-2014'],
                c='black', linewidth=1.5, alpha=0.9, label='Historico')

    ax.fill_between(max_pt.loc['31-12-2014':].index, max_pt.loc['31-12-2014':].values,
                        min_pt.loc['31-12-2014':].values, color='gray', alpha=.5)
    s_n = 14
    dict_var_g = {'TS_1': ['Temperatura Media', 'TS', 'C'], 'PT_4': ['Precipitacion total', 'PT', 'mm']}
    ax.set_title('GCM- Variable:' + dict_var_g[var][0], fontdict={'fontsize': 18, })
    ax.set_ylabel('{} ({})'.format(dict_var_g[var][1], dict_var_g[var][2]), fontsize=s_n)
    ax.set_xlabel('$Años$', fontsize=s_n)

    plt.grid()
    plt.legend()
    fig.subplots_adjust(hspace=.001, wspace=0.15)
    fig.tight_layout()
    plt.savefig(path_out + '/Grafica_Anual_' + str(sta) + '_' + var, dpi=300)
    plt.close()

    # plt.show()


    return


BASE_PATH = r'D:\Cambio_Climatico_Mendoza/'# path input project


models = ['ACCESS-ESM1-5','CanESM5','CESM2','EC-Earth3','MIROC6','MPI-ESM1-2-LR','MRI-ESM2-0']

escenarios = ['historical-obs','historical-gcm','ssp126', 'ssp245', 'ssp370', 'ssp585']

variables = ['PT_4','TS_1']

tipo= 'd'#'Monthly'#

if __name__ == "__main__":

    os.chdir(BASE_PATH)

    for var in variables[0:]:
        print(var)
        if tipo == 'Monthly':
            path_in = '02_Climate_Change_Scenarios/03_monthly_Series/' + var + '/'

            output_path = os.path.join('02_Climate_Change_Scenarios', '04_graphics', var )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            dic_cc = read_all_xlsfiles_m(models, escenarios, path_in, var)
        else:
            path_in = '02_Climate_Change_Scenarios/02_Series_Downscaling/'
            output_path = os.path.join('02_Climate_Change_Scenarios', '04_graphics', var)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            dic_cc = read_all_xlsfiles_day(models, escenarios, path_in, var)

        obs = add_leap(dic_cc[models[0]]['historical-obs']).loc['1980-01-01':'2014-12-31']
        stations = obs.columns

        for sta in stations[0:]:
            print(sta)

            series_grafica_m(sta, var, dic_cc, models, escenarios, output_path)
            series_grafica(sta, var, dic_cc, models, escenarios, output_path)