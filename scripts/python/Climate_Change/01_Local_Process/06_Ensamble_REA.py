
import pandas as pd
from pylab import *
matplotlib.style.use('ggplot')
import os



def read_all_xlsfiles_m(modelos,escenarios,path,var):


    list_file = {}
    esc = escenarios
    for mn in modelos[0:]:
        print(mn)
        file = pd.ExcelFile(path + 'Series_Mensuales_' + mn + '_' + var + '.xlsx')
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

def convergence_factor(model, other_models):

   # Calcula la diferencia media entre un modelo y todos los demás

   diffs = [np.abs(model - m) for m in other_models]

   diff = np.mean(diffs, axis=0)

   return pd.DataFrame(diff).mean()

def mse(predictions, observations):
    mse =np.mean((predictions - observations)**2,axis=0)
    return mse

def performance_factor(predictions, observations):

   mse_value = mse(predictions, observations)

   return 1 / mse_value

def calculate_weights(hist_models, observations):
    weight = []
    for mm in hist_models:
        c = convergence_factor(mm, [m for m in hist_models if not np.array_equal(m, mm)])
        D = performance_factor(mm, observations)
        weight.append(pd.DataFrame(c.values * D.values))
    total_weight = np.sum(weight, axis=0)
    normalized_weights = [w / total_weight for w in weight]

    return normalized_weights

def ok_to_csv_m(df):
    df.index = pd.DatetimeIndex(df.index)

    df['Year'] = df.index.year
    df['Month'] = df.index.month #32200010
    df = df.reset_index()
    df = df.rename(columns={"index": "Date"})
    df = df.drop(columns=['Date'])
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df_result = df[cols]
    return df_result

def add_leap(df,tipo='present'):
    if tipo=='present':
        start = '1970-01-01'
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

models = ['ACCESS-ESM1-5','CanESM5','CESM2','EC-Earth3','MIROC6','MPI-ESM1-2-LR','MRI-ESM2-0']

escenarios = ['historical-obs','historical-gcm','ssp126', 'ssp245', 'ssp370', 'ssp585']

variables = ['PT_4','TS_1']

tipo= 'd'#'Monthly'#

if __name__ == "__main__":

    os.chdir(r'D:\Cambio_Climatico_Mendoza/')

    for var in variables:
        print(var)
        if tipo == 'Monthly':
            path_in = '02_Climate_Change_Scenarios/03_monthly_Series/' + var + '/'

            output_path = os.path.join('02_Climate_Change_Scenarios', '06_Ensamble_Series', 'Monthly', var)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            dic_cc = read_all_xlsfiles_m(models,escenarios,path_in,var)
            xls_writer = pd.ExcelWriter(os.path.join(output_path, f'Series_GCM_{tipo}_{var}.xlsx'))
        else:
            path_in = '02_Climate_Change_Scenarios/02_Series_Downscaling/'
            output_path = os.path.join( '02_Climate_Change_Scenarios', '06_Ensamble_Series', 'Daily', var)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            dic_cc = read_all_xlsfiles_day(models,escenarios,path_in,var)
            xls_writer = pd.ExcelWriter(os.path.join(output_path,f'Series_GCM_{tipo}_{var}.xlsx'))


        obs = add_leap(dic_cc[models[0]]['historical-obs']).loc['1980-01-01':'2014-12-31']

        hist_models  = [add_leap(dic_cc[model]['historical-gcm']).loc['1980-01-01':'2014-12-31'] for model in models]

        #convergencia

        weights = calculate_weights(hist_models, obs)

        tem_ensamble =[]
        for mm, w in zip(hist_models,weights):

            tem_ensamble.append(w.T.values*mm)
        ensamble_Historico = pd.DataFrame(np.sum(tem_ensamble, axis=0),index=obs.index, columns=obs.columns)
        ensamble_Historico.to_excel(xls_writer, sheet_name='historical-gcm', merge_cells=False)

        for es in escenarios[2:]:
            print(es)

            Ssp_models = [add_leap(dic_cc[model][es],tipo='Future').loc['2015-01-01':'2100-12-31'] for model in models]

            tem_ensamble_ssp = []
            for mm, w in zip(Ssp_models, weights):
                tem_ensamble_ssp.append(w.T.values * mm)
            ensamble_ssp= pd.DataFrame(np.sum(tem_ensamble_ssp, axis=0), index=Ssp_models[0].index, columns=Ssp_models[0].columns)

            ensamble_ssp.to_excel(xls_writer, sheet_name=es, merge_cells=False)
        xls_writer.close()