import functools as ft
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import scipy.stats as stats


"""this code allows you to add the PT and TS series at daily, monthly and yearly resolution."""


def var_mean(df_base, k):
    # print (k)
    sr_daily = df_base[k].reset_index()
    df_monthly = sr_daily.groupby(['Year', 'Month']).count()
    df_monthly['Check'] = df_monthly[k] / df_monthly['Day']
    df_monthly['Value'] = sr_daily.groupby(['Year', 'Month']).mean()[k]
    mask = (df_monthly['Check'] < 0.7)
    df_monthly.loc[mask, 'Value'] = np.nan
    df_monthly.columns = ['Potential', 'Real', 'Check', k]



    df_temp = df_monthly[k].reset_index()
    df_yearly = df_temp.groupby(['Year']).count()
    df_yearly['Check'] = df_yearly[k] / df_yearly['Month']
    df_yearly['Value'] = df_temp.groupby(['Year']).mean()[k]
    mask_y = (df_yearly['Check'] < 0.7)
    df_yearly.loc[mask_y, 'Value'] = np.nan
    df_yearly.columns = ['Potential', 'Real', 'Check', k]
    return df_monthly[k], df_yearly[k]

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
    return df_monthly[k], df_yearly[k]

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



    df_temp = df_monthly[k].reset_index()
    df_yearly = df_temp.groupby(['Year']).count()
    df_yearly['Check'] = df_yearly[k] / df_yearly['Month']
    dfy_rem_na = df_temp.dropna(axis=0, how='any')
    df_yearly['Value'] = dfy_rem_na.groupby(['Year']).agg(lambda x: stats.mode(x)[0][0])[k]
    mask_y = (df_yearly['Check'] < 0.5)
    df_yearly.loc[mask_y, 'Value'] = np.nan
    df_yearly.columns = ['Potential', 'Real', 'Check', k]
    return df_monthly[k], df_yearly[k]

def var_sum_PT(df_base, k):
    # print (k)
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

    return df_monthly[k], df_yearly[k],df_max [k]


dict_var = {'Tas':'tas', 'PT':'pr'}
dict_cat = {'Tas':'TS_1', 'PT':'PT_4'}
escenarios = ['historical-obs','historical-gcm','ssp126', 'ssp245', 'ssp370', 'ssp585']
variables = ['PT', 'Tas']
variable_Dict = {'PT_4': 4,'TS_1': 1,'TS_2': 1, 'TS_8': 1,
                'BS_4': 0,'EV_4': 0,'HR_1': 1, 'QL_1': 3,
                 'ETP': 0,'ETR': 0 }


BASE_PATH = r'D:\Cambio_Climatico_Mendoza/'# path input project

path_series = os.path.join(BASE_PATH, '02_Climate_Change_Scenarios', '02_Series_Downscaling')# do not change

multiprocessing = True
if __name__ == "__main__":


    for var in variables[0:]:
        print(var)

        subdir = path_series.replace("\\", "/")

        models = next(os.walk(subdir))[1]


        output_path = os.path.join(BASE_PATH, '02_Climate_Change_Scenarios','03_monthly_Series', dict_cat[var])# do not change
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for m in models[0:]:
            print(m)

            new_path = os.path.join(subdir, m, dict_cat[var])
            tipo = variable_Dict[dict_cat[var]]
            if tipo == 1:
                name_out_M = pd.ExcelWriter(os.path.join(output_path, f'Monthly_Series_{m}_{dict_cat[var]}.xlsx'))
                name_out_Y = pd.ExcelWriter(os.path.join(output_path, f'Yearly_Series_{m}_{dict_cat[var]}.xlsx'))
            if tipo == 4:
                name_out_M = pd.ExcelWriter(os.path.join(output_path, f'Monthly_Series_{m}_{dict_cat[var]}.xlsx'))
                name_out_M24 = pd.ExcelWriter(os.path.join(output_path, f'Monthly_Series_{m}_PT_9.xlsx'))
                name_out_Y = pd.ExcelWriter(os.path.join(output_path, f'Yearly_Series_{m}_{dict_cat[var]}.xlsx'))


            i = 1
            for es in escenarios:
                print(es)
                if es == 'historical-obs':
                    new_path_obs =  subdir
                    df_base = pd.read_csv(os.path.join(new_path_obs, f'01_Series_historical-obs_{dict_cat[var]}.csv'), index_col=0)
                else:
                    df_base= pd.read_csv(os.path.join(new_path, f'0{i}_Ds_Series_{es}_{dict_cat[var]}.csv'), index_col=0)


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
                if tipo == 4:
                    partial_info = ft.partial(var_sum_PT, df_base)

                if multiprocessing:
                    pool = mp.Pool()
                    ls_month = pool.map(partial_info, stations)
                    pool.close()
                    pool.join()
                else:
                    ls_month = list(map(partial_info, stations))

                if dict_cat[var] == 'PT_4':
                    dfb_monthly = pd.DataFrame(index=ls_month[0][0].index)
                    dfb_monthly_max24 = pd.DataFrame(index=ls_month[0][0].index)
                    dfb_yearly = pd.DataFrame(index=ls_month[0][1].index)

                else:
                    dfb_monthly = pd.DataFrame(index=ls_month[0][0].index)
                    dfb_yearly = pd.DataFrame(index=ls_month[0][1].index)

                if dict_cat[var] == 'PT_4':
                    for sta in range(len(stations)):
                        dfb_monthly = pd.concat([dfb_monthly, ls_month[sta][0]], axis=1)
                        dfb_yearly = pd.concat([dfb_yearly, ls_month[sta][1]], axis=1)
                        dfb_monthly_max24 = pd.concat([dfb_monthly_max24, ls_month[sta][2]], axis=1)
                    indexado_pt = dfb_monthly.index.get_level_values('Year').astype(str) + dfb_monthly.index.get_level_values('Month').astype(str)+str(1)
                    dfb_monthly.index = pd.to_datetime(indexado_pt, format='%Y%m%d')
                    dfb_monthly_max24.index = pd.to_datetime(indexado_pt, format='%Y%m%d')

                    dfb_monthly.to_excel(name_out_M , sheet_name=es, merge_cells=False)
                    dfb_monthly_max24.to_excel(name_out_M24, sheet_name=es, merge_cells=False)
                    dfb_yearly.to_excel(name_out_Y, sheet_name=es, merge_cells=False)
                else:
                    for sta in range(len(stations)):
                        dfb_monthly = pd.concat([dfb_monthly, ls_month[sta][0]], axis=1)
                        dfb_yearly = pd.concat([dfb_yearly, ls_month[sta][1]], axis=1)

                    indexado_all = dfb_monthly.index.get_level_values('Year').astype(str) + dfb_monthly.index.get_level_values('Month').astype(str) + str(1)
                    dfb_monthly.index = pd.to_datetime(indexado_all, format='%Y%m%d')

                    dfb_monthly.to_excel(name_out_M, sheet_name=es, merge_cells=False)
                    dfb_yearly.to_excel(name_out_Y, sheet_name=es, merge_cells=False)


                i += 1
            name_out_M.close()
            name_out_Y.close()
            name_out_M24.close()