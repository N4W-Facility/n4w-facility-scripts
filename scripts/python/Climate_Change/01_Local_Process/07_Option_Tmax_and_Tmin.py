import pandas as pd

# Función para estimar temperaturas mínima y máxima para cada escenario

path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects\SE18-Yaque_del_Norte\05-Science_Workstreams\00-Working_Folder\SA\02_Climate_Caracterization\04_Climate_Change/'

path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects\SE18-Yaque_del_Norte\05-Science_Workstreams\00-Working_Folder\SA\02_Climate_Caracterization\04_Climate_Change\02_Escenarios_Cambio_Climatico\06_Ensamble_Series\Daily\TS_1/'
# Cargar los archivos de datos
file_path_obs = '02_Selected_data_check.xlsx'
file_path_scenarios = 'Series_GCM_d_TS_1.xlsx'

# Cargar datos observados
obs_data = pd.read_excel(path +file_path_obs, sheet_name=None)
scenarios_data = pd.read_excel(path +r'02_Escenarios_Cambio_Climatico\06_Ensamble_Series\Daily\TS_1/'+file_path_scenarios, sheet_name=None)

# Extraer temperaturas media, máxima y mínima observadas
temp_mean_obs = obs_data['TS_1'].iloc[:, 1:]  # Saltar la columna de fecha
temp_max_obs = obs_data['TS_2'].iloc[:, 1:]
temp_min_obs = obs_data['TS_3'].iloc[:, 1:]

# Identificar columnas comunes entre los datos observados y los escenarios
common_columns = temp_mean_obs.columns.intersection(scenarios_data['historical-gcm'].columns[1:])

# Filtrar los datos observados a las columnas comunes
temp_mean_obs = temp_mean_obs[common_columns]
temp_max_obs = temp_max_obs[common_columns]
temp_min_obs = temp_min_obs[common_columns]

# Calcular las diferencias promedio entre media-máxima y media-mínima
diff_mean_max = temp_max_obs - temp_mean_obs
diff_mean_min = temp_mean_obs - temp_min_obs
avg_diff_mean_max = diff_mean_max.mean()
avg_diff_mean_min = diff_mean_min.mean()




# Función para estimar temperaturas mínima y máxima en escenarios
def estimate_min_max_temps(scenario_df, avg_diff_max, avg_diff_min, common_cols):
    temp_mean = scenario_df[common_cols]
    temp_max_estimated = temp_mean + avg_diff_max
    temp_min_estimated = temp_mean - avg_diff_min

    # Insertar la columna de fecha e índice para mantener el formato
    temp_max_estimated.insert(0, 'Date', scenario_df.iloc[:, 0])
    temp_min_estimated.insert(0, 'Date', scenario_df.iloc[:, 0])

    # Establecer 'Date' como índice
    temp_max_estimated.set_index('Date', inplace=True)
    temp_min_estimated.set_index('Date', inplace=True)

    return temp_max_estimated, temp_min_estimated

# Crear los archivos de Excel con hojas por cada escenario
with pd.ExcelWriter(path_out +'Tmax_estimations.xlsx') as writer_max, pd.ExcelWriter(path_out +'Tmin_estimations.xlsx') as writer_min:
    for scenario_name, scenario_df in scenarios_data.items():
        temp_max_est, temp_min_est = estimate_min_max_temps(scenario_df, avg_diff_mean_max, avg_diff_mean_min,
                                                            common_columns)

        # Guardar cada escenario en una hoja diferente
        temp_max_est.to_excel(writer_max, sheet_name=scenario_name)
        temp_min_est.to_excel(writer_min, sheet_name=scenario_name)

print("Estimaciones completadas y exportadas a archivos Tmax_estimations.xlsx y Tmin_estimations.xlsx con una hoja por cada escenario.")



