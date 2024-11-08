import functools as ft
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import scipy.stats as stats


def quality_data(df_station, station):
    df = df_station.dropna(subset=[station])
    df.set_index(keys=['Day', 'Month', 'Year'], inplace=True)
    if df.empty:
        # print station
        # print df.to_string()
        start = 'Empty'
        end = 'Empty'
        real = 'Empty'
        completeness = 'Empty'
    else:
        # start = pd.to_datetime(str(df.index[0][0]) + '/' + str(df.index[0][1]) + '/' + str(df.index[0][2]), infer_datetime_format=True)
        # end = pd.to_datetime(str(df.index[-1][0]) + '/' + str(df.index[-1][1]) + '/' + str(df.index[-1][2]), infer_datetime_format=True)
        start = pd.to_datetime(str(df.index[0][0]) + '/' + str(df.index[0][1]) + '/' + str(df.index[0][2]),
                               format='%d/%m/%Y')

        end = pd.to_datetime(str(df.index[-1][0]) + '/' + str(df.index[-1][1]) + '/' + str(df.index[-1][2]),
                             format='%d/%m/%Y')
        potential = end - start
        try:
            real = len(df[station]) / 365
            completeness = real / (float(potential.days) / 365)
        except ZeroDivisionError:
            completeness = 'ZeroDivisionError'

    return station, start, end, real, completeness

def var_mean(df_base, k):
    print (k)
    sr_daily = df_base[k].reset_index()
    df_monthly = sr_daily.groupby(['Year', 'Month']).count()
    df_monthly['Check'] = df_monthly[k] / df_monthly['Day']
    df_monthly['Value'] = sr_daily.groupby(['Year', 'Month']).mean()[k]
    mask = (df_monthly['Check'] < 0.7)
    df_monthly.loc[mask, 'Value'] = np.nan
    df_monthly.columns = ['Potential', 'Real', 'Check', k]

    quality = quality_data(df_station=sr_daily, station=k)

    df_temp = df_monthly[k].reset_index()
    df_yearly = df_temp.groupby(['Year']).count()
    df_yearly['Check'] = df_yearly[k] / df_yearly['Month']
    df_yearly['Value'] = df_temp.groupby(['Year']).mean()[k]
    mask_y = (df_yearly['Check'] < 0.7)
    df_yearly.loc[mask_y, 'Value'] = np.nan
    df_yearly.columns = ['Potential', 'Real', 'Check', k]
    return df_monthly[k], df_yearly[k], quality

def var_sum(df_base, k):
    print (k)
    sr_daily = df_base[k].reset_index()
    df_monthly = sr_daily.groupby(['Year', 'Month']).count()
    df_monthly['Check'] = df_monthly[k] / df_monthly['Day']
    df_monthly['Total Value'] = sr_daily.groupby(['Year', 'Month']).sum()[k]
    df_monthly['Mean Value'] = sr_daily.groupby(['Year', 'Month']).mean()[k]
    mask2 = (df_monthly['Check'] <= 0.99)
    if (df_monthly.loc[mask2, 'Total Value']).empty is False:
        df_monthly.loc[mask2, 'Total Value'] = df_monthly['Mean Value'] * df_monthly['Day']
        mask1 = (df_monthly['Check'] < 0.7)
        df_monthly.loc[mask1, 'Total Value'] = np.nan
    df_monthly.columns = ['Potential', 'Real', 'Check', k, 'Mean Value']

    quality = quality_data(df_station=sr_daily, station=k)

    df_temp = df_monthly[k].reset_index()
    df_yearly = df_temp.groupby(['Year']).count()
    df_yearly['Check'] = df_yearly[k] / df_yearly['Month']
    df_yearly['Total Value'] = df_temp.groupby(['Year']).sum()[k]
    df_yearly['Mean Value'] = df_temp.groupby(['Year']).mean()[k]
    mask2_y = (df_yearly['Check'] <= 0.99)
    if (df_yearly.loc[mask2_y, 'Total Value']).empty is False:
        df_yearly.loc[mask2_y, 'Total Value'] = df_yearly['Mean Value'] * df_yearly['Month']
        mask1_y = (df_yearly['Check'] < 0.7)
        df_yearly.loc[mask1_y, 'Total Value'] = np.nan
    df_yearly.columns = ['Potential', 'Real', 'Check', k, 'Mean Value']
    return df_monthly[k], df_yearly[k], quality

def var_freq(df_base, k):
    print (k)
    sr_daily = df_base[k].reset_index()
    df_monthly = sr_daily.groupby(['Year', 'Month']).count()
    df_monthly['Check'] = df_monthly[k] / df_monthly['Day']
    df_rem_na = sr_daily.dropna(axis=0, how='any')
    df_monthly['Value'] = df_rem_na.groupby(['Year', 'Month']).agg(lambda x: stats.mode(x)[0][0])[k]
    mask = (df_monthly['Check'] < 0.5)
    df_monthly.loc[mask, 'Value'] = np.nan
    df_monthly.columns = ['Potential', 'Real', 'Check', k]

    quality = quality_data(df_station=sr_daily, station=k)

    df_temp = df_monthly[k].reset_index()
    df_yearly = df_temp.groupby(['Year']).count()
    df_yearly['Check'] = df_yearly[k] / df_yearly['Month']
    dfy_rem_na = df_temp.dropna(axis=0, how='any')
    df_yearly['Value'] = dfy_rem_na.groupby(['Year']).agg(lambda x: stats.mode(x)[0][0])[k]
    mask_y = (df_yearly['Check'] < 0.5)
    df_yearly.loc[mask_y, 'Value'] = np.nan
    df_yearly.columns = ['Potential', 'Real', 'Check', k]
    return df_monthly[k], df_yearly[k], quality

def var_caudal(df_base, k):
    print(k)
    sr_daily = df_base[k].reset_index()

    df_monthly = sr_daily.groupby(['Year', 'Month']).count()
    df_monthly['Check'] = df_monthly[k] / df_monthly['Day']
    df_monthly['Value'] = sr_daily.groupby(['Year', 'Month']).mean()[k]
    mask = (df_monthly['Check'] < 0.7)
    df_monthly.loc[mask, 'Value'] = np.nan
    df_monthly.columns = ['Potential', 'Real', 'Check', k]

    df_max = sr_daily.groupby(['Year', 'Month']).count()
    df_max['Check'] = df_max[k] / df_max['Day']
    df_max['Value'] = sr_daily.groupby(['Year', 'Month']).max()[k]
    mask = (df_max['Check'] < 0.7)
    df_max.loc[mask, 'Value'] = np.nan
    df_max.columns = ['Potential', 'Real', 'Check', k]

    quality = quality_data(df_station=sr_daily, station=k)

    df_min = sr_daily.groupby(['Year', 'Month']).count()
    df_min ['Check'] = df_min [k] / df_min ['Day']
    df_min ['Value'] = sr_daily.groupby(['Year', 'Month']).min()[k]
    mask = (df_min ['Check'] < 0.7)
    df_min .loc[mask, 'Value'] = np.nan
    df_min .columns = ['Potential', 'Real', 'Check', k]

    df_temp = df_monthly[k].reset_index()
    df_yearly = df_temp.groupby(['Year']).count()
    df_yearly['Check'] = df_yearly[k] / df_yearly['Month']
    df_yearly['Value'] = df_temp.groupby(['Year']).mean()[k]
    mask_y = (df_yearly['Check'] < 0.7)
    df_yearly.loc[mask_y, 'Value'] = np.nan
    df_yearly.columns = ['Potential', 'Real', 'Check', k]


    return df_monthly[k],  df_yearly[k],quality,df_max [k], df_min [k]




def var_sum_PT(df_base, k):
    print (k)
    df_base[k] = pd.to_numeric(df_base[k], errors='coerce')
    sr_daily = df_base[k].reset_index()
    df_monthly = sr_daily.groupby(['Year', 'Month']).count()
    df_monthly['Check'] = df_monthly[k] / df_monthly['Day']
    df_monthly['Total Value'] = sr_daily.groupby(['Year', 'Month']).sum()[k]
    df_monthly['Mean Value'] = sr_daily.groupby(['Year', 'Month']).mean()[k]
    mask2 = (df_monthly['Check'] <= 0.99)
    if (df_monthly.loc[mask2, 'Total Value']).empty is False:
        df_monthly.loc[mask2, 'Total Value'] = df_monthly['Mean Value'] * df_monthly['Day']
        mask1 = (df_monthly['Check'] < 0.7)
        df_monthly.loc[mask1, 'Total Value'] = np.nan
    df_monthly.columns = ['Potential', 'Real', 'Check', k, 'Mean Value']


    df_max = sr_daily.groupby(['Year', 'Month']).count()
    df_max['Check'] = df_max[k] / df_max['Day']
    df_max['Value'] = sr_daily.groupby(['Year', 'Month']).max()[k]
    mask = (df_max['Check'] < 0.7)
    df_max.loc[mask, 'Value'] = np.nan
    df_max.columns = ['Potential', 'Real', 'Check', k]

    quality = quality_data(df_station=sr_daily, station=k)

    df_temp = df_monthly[k].reset_index()
    df_yearly = df_temp.groupby(['Year']).count()
    df_yearly['Check'] = df_yearly[k] / df_yearly['Month']
    df_yearly['Total Value'] = df_temp.groupby(['Year']).sum()[k]
    df_yearly['Mean Value'] = df_temp.groupby(['Year']).mean()[k]
    mask2_y = (df_yearly['Check'] <= 0.99)
    if (df_yearly.loc[mask2_y, 'Total Value']).empty is False:
        df_yearly.loc[mask2_y, 'Total Value'] = df_yearly['Mean Value'] * df_yearly['Month']
        mask1_y = (df_yearly['Check'] < 0.7)
        df_yearly.loc[mask1_y, 'Total Value'] = np.nan
    df_yearly.columns = ['Potential', 'Real', 'Check', k, 'Mean Value']

    return df_monthly[k], df_yearly[k],quality,df_max [k]



variable_Dict = {'PT_4': 4,'TS_1': 1,'TS_2': 1,'TS_3': 1, 'TS_8': 1,
                'BS_4': 0,'EV_4': 0,'HR_1': 1, 'QL_1': 3,
                 'ETP': 0,'ETR': 0 ,'RS':2,'Vwind':2,'Uwind':2}

path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

path_out= r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization/'


name = r'04_Datos_Sel_sin_outliers.xlsx' #
name_out = 'Data'

file = pd. ExcelFile(path+name)

variables =file.sheet_names #['QL_1',]#

multiprocessing = False

if not os.path.exists(path_out+'02_Monthly/'):
    os.makedirs(path_out+'02_Monthly/',exist_ok=True)
if not os.path.exists(path_out+'03_Yearly/'):
    os.makedirs(path_out+'03_Yearly/',exist_ok=True)


xls_monthly = pd.ExcelWriter(path_out+'02_Monthly/' + '05_Monthly_Series.xlsx')

xls_yearly = pd.ExcelWriter(path_out+'03_Yearly/' +  '06_Yearly_Series.xlsx')



for var in variables[0:]:
    print(var)

    df_base = file.parse(sheet_name=var, index_col=0)

    tipo = variable_Dict[var]
    stations = df_base.columns
    df_base.index = pd.to_datetime(df_base.index)
    df_base['Day'] = df_base.index.day
    df_base['Month'] = df_base.index.month
    df_base['Year'] = df_base.index.year
    df_base.set_index(keys=['Day', 'Month', 'Year'], drop=True, inplace=True)

    if tipo == 0:
        partial_info = ft.partial(var_sum, df_base)
    if tipo == 1:
        partial_info = ft.partial(var_mean, df_base)
    if tipo == 2:
        partial_info = ft.partial(var_freq, df_base)
    if tipo == 3:
        partial_info = ft.partial(var_caudal, df_base)
    if tipo == 4:
        partial_info = ft.partial(var_sum_PT, df_base)

    if multiprocessing:
        pool = mp.Pool()
        ls_month = pool.map(partial_info, stations)
        pool.close()
        pool.join()
    else:
        ls_month = list(map(partial_info, stations))

    if var == 'QL_1' or var == 'sed' or var == 'OrgN' or var == 'OrgP':
        dfb_monthly = pd.DataFrame(index=ls_month[0][0].index)
        dfb_max = pd.DataFrame(index=ls_month[0][0].index)
        dfb_min = pd.DataFrame(index=ls_month[0][0].index)
        dfb_yearly = pd.DataFrame(index=ls_month[0][1].index)
        df_quality = pd.DataFrame(columns=['Stations', 'Start', 'End', 'Years on Record', 'Completeness'])
    elif var == 'PT_4':
        dfb_monthly = pd.DataFrame(index=ls_month[0][0].index)
        dfb_monthly_max24 = pd.DataFrame(index=ls_month[0][0].index)
        dfb_yearly = pd.DataFrame(index=ls_month[0][1].index)
        df_quality = pd.DataFrame(columns=['Stations', 'Start', 'End', 'Years on Record', 'Completeness'])

    else:
        dfb_monthly = pd.DataFrame(index=ls_month[0][0].index)
        dfb_yearly = pd.DataFrame(index=ls_month[0][1].index)
        df_quality = pd.DataFrame(columns=['Stations', 'Start', 'End', 'Years on Record', 'Completeness'])

    if var == 'QL_1'or var == 'sed' or var == 'OrgN' or var == 'OrgP':
        for sta in range(len(stations)):
            dfb_monthly = pd.concat([dfb_monthly, ls_month[sta][0]], axis=1)
            dfb_yearly = pd.concat([dfb_yearly, ls_month[sta][1]], axis=1)
            dfb_max = pd.concat([dfb_max, ls_month[sta][3]], axis=1)
            dfb_min = pd.concat([dfb_min, ls_month[sta][4]], axis=1)

        indexado_ql =  dfb_monthly.index.get_level_values('Year').astype(str) + dfb_monthly.index.get_level_values('Month').astype(
                str) + str(1)
        dfb_monthly.index = pd.to_datetime(indexado_ql, format='%Y%m%d', infer_datetime_format=True)
        dfb_min.index = pd.to_datetime(indexado_ql, format='%Y%m%d', infer_datetime_format=True)
        dfb_max.index = pd.to_datetime(indexado_ql, format='%Y%m%d', infer_datetime_format=True)


        dfb_monthly.to_excel(xls_monthly, sheet_name=var, merge_cells=False)
        dfb_max.to_excel(xls_monthly, sheet_name='QL_2', merge_cells=False)
        dfb_min.to_excel(xls_monthly, sheet_name='QL_3', merge_cells=False)
        dfb_yearly.to_excel(xls_yearly, sheet_name=var, merge_cells=False)
    elif var == 'PT_4':
        for sta in range(len(stations)):
            dfb_monthly = pd.concat([dfb_monthly, ls_month[sta][0]], axis=1)
            dfb_yearly = pd.concat([dfb_yearly, ls_month[sta][1]], axis=1)
            new_row = pd.Series(ls_month[sta][2], index=['Stations', 'Start', 'End', 'Years on Record', 'Completeness'])
            # df_quality = df_quality.append(pd.Series(ls_month[sta][2], index=['Stations', 'Start', 'End', 'Years on Record', 'Completeness']),             ignore_index=True)
            dfb_monthly_max24 = pd.concat([dfb_monthly_max24, ls_month[sta][3]], axis=1)


        indexado_pt = dfb_monthly.index.get_level_values('Year').astype(str) + dfb_monthly.index.get_level_values('Month').astype(
                str) + str(1)
        dfb_monthly.index = pd.to_datetime(indexado_pt, format='%Y%m%d', infer_datetime_format=True)
        dfb_monthly_max24.index = pd.to_datetime(indexado_pt, format='%Y%m%d', infer_datetime_format=True)


        dfb_monthly.to_excel(xls_monthly, sheet_name=var, merge_cells=False)
        dfb_monthly_max24.to_excel(xls_monthly, sheet_name='PT_9', merge_cells=False)
        dfb_yearly.to_excel(xls_yearly, sheet_name=var, merge_cells=False)

    else:
        for sta in range(len(stations)):
            dfb_monthly = pd.concat([dfb_monthly, ls_month[sta][0]], axis=1)
            dfb_yearly = pd.concat([dfb_yearly, ls_month[sta][1]], axis=1)
            new_row = pd.Series(ls_month[sta][2], index=['Stations', 'Start', 'End', 'Years on Record', 'Completeness'])
            df_quality = pd.concat([df_quality, new_row.to_frame().T], ignore_index=True)



        indexado_all = dfb_monthly.index.get_level_values('Year').astype(str) + dfb_monthly.index.get_level_values('Month').astype(
                str) + str(1)
        dfb_monthly.index = pd.to_datetime(indexado_all, format='%Y%m%d')

        dfb_monthly.to_excel(xls_monthly, sheet_name=var, merge_cells=False)
        dfb_yearly.to_excel(xls_yearly, sheet_name=var, merge_cells=False)
xls_monthly.close()
xls_yearly.close()