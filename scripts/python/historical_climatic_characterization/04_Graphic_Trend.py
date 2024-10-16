import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm





dir_name = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'


path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'





name_file = '04_Datos_Sel_sin_outliers.xlsx'

file = pd.ExcelFile(dir_name+name_file)

variables = file.sheet_names

dic_var = {'PT_4': 'Precipitacion (mm)',
           'TS_1': 'Temperatura (°C)',
           'TS_2': 'Temperatura (°C)',
            'TS_3': 'Temperatura (°C)',
            'EV_4': 'Evaporacion (mm)',
            'BS_4': 'Brillo Solar (horas)',
            'HR_1': 'Humedad Relativa (%)',
            'QL_1': 'Caudal (mcs)',
             'RS':  'Radiacion Solar',
           'Uwind': 'Uwind',
           'Vwind': 'Vwind',

           }
for var in variables[0:]:
    print(var)
    df = file.parse(var,index_col=0)

    stations = df.columns

    for sta in stations[0:]:
        print(sta)

        try:
            # Encuentra el índice del primer punto no-NaN
            first_valid_index = df[sta].first_valid_index()

            # Filtra tu DataFrame y el rango numérico basado en ese punto
            df_filtered = df.loc[first_valid_index:]

            # Crea una máscara para los datos válidos en el DataFrame filtrado
            mask_valid = ~np.isnan(df_filtered[sta])

            Y = df_filtered[sta][mask_valid]

            x_numeric = np.arange(len(Y.index))



            # Ajusta la línea de tendencia solo usando datos válidos
            z = np.polyfit(x_numeric, Y, 1)
            p = np.poly1d(z)


            X = sm.add_constant(x_numeric, prepend=False)
            YvsX = sm.OLS(Y, X).fit()
            slope = YvsX.params[0]

            # slope = z[0]

            # Decide el color de la línea de tendencia basado en la pendiente
            trend_color = "green" if slope < 0 else "red"

            # Grafica
            figsize=(10, 5)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            ax.plot(Y.index, Y, label="Datos originales", alpha=0.6)
            ax.plot(Y.index, p(x_numeric), "--", color=trend_color,
                     label=f"Línea de tendencia (pendiente: {slope:.8f})")

            ax.set_title('Tendencia Variable '+var+' Estacion: '+str(sta))
            ax.set_xlabel('Años')
            ax.set_ylabel(dic_var[var])
            ax.legend()
            ax.grid(True)
            plt.tight_layout()


            os.makedirs(path_out + '02_Trends/' + '01_Graphic_Trend/'+var+'/', exist_ok=True)

            # plt.show()

            plt.savefig(path_out + '02_Trends/' + '01_Graphic_Trend/'+var+'/'+ 'Trends_'+str(sta))
            plt.close()
        except:
            pass
