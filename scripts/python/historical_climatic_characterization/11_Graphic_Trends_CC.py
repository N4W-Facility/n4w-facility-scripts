import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
import os


# Función para calcular la tendencia
def calcular_tendencia(x, y):
    coef = np.polyfit(x, y, 1)
    polinomio = np.poly1d(coef)
    return polinomio(x)


def calcular_tendencia2(x, y):
    mejor_ajuste = None
    mejor_r2 = -np.inf
    mejor_orden = 0

    for orden in range(1, 5):
        coef = np.polyfit(x, y, orden)
        polinomio = np.poly1d(coef)
        y_ajustado = polinomio(x)
        r2 = r2_score(y, y_ajustado)

        if r2 > mejor_r2:
            mejor_ajuste = polinomio
            mejor_r2 = r2
            mejor_orden = orden

    print(f"Mejor ajuste: polinomio de orden {mejor_orden} con R^2 = {mejor_r2}")
    return mejor_ajuste(x)

path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\01_Completitud_Consistencia/'

path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization/'

name_file = '04_Datos_Sel_sin_outliers.xlsx'
file = pd.ExcelFile(path+name_file)

variables = ['QL_1','TS_2','TS_3','RS','HR_1','PT_4']

dict_var = {'PT_4':['Precipitacion Total',"Precipitacion (mm)"],
            'TS_1':['Temperatura Media',"Temperatura (°C)"],
            'TS_2':['Temperatura Maxima',"Temperatura (°C)"],
            'TS_8':['Temperatura Minima',"Temperatura (°C)"],
            'TS_3':['Temperatura Minima',"Temperatura (°C)"],
            'EV_4':['Evaporacion Total',"Evaporacion (mm)"],
            'BS_4':['Brillo solar',"Brillo solar (horas)"],
            'QL_1':['Caudal Medio',"Caudal (m3/s)"],
            'HR_1':['Humedad Relativa',"HR (%)"],
            'RS':['Radiacion Solar',""]}

for var in variables[3:]:
    print(var)
    data = file.parse(var,index_col=0)
    stations = data.columns
    output_path = path_out + '04_Climate_Change/01_Trends_Graphics_CC'

    if not os.path.exists(output_path + '/' + var + '/'):
        os.makedirs(output_path + '/' + var+ '/', exist_ok=True)

    for sta in stations[0:]:
        print(sta)

        start = data[sta].dropna().index.min()
        df  = data[sta].loc[start:].interpolate()


        fig, axes = plt.subplots(1, 12, figsize=(18, 10), sharey=True)
        meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre',
                 'Noviembre',                 'Diciembre']
        fechas = df.index
        s_n = 14
        for i, ax in enumerate(axes):
            # print(i + 1)
            mask = np.where(df.index.month == i + 1)
            x = fechas[mask]
            y = df.loc[df.index[mask]]

            años_unicos = np.unique(x.year)
            promedios_mensuales = np.array([np.mean(y[x.year == año]) for año in años_unicos])

            ax.plot(años_unicos, promedios_mensuales, linewidth=0.8, color='red')
            ax.plot(años_unicos, calcular_tendencia(años_unicos.astype(int), promedios_mensuales), color='blue', alpha=.9)
            ax.set_title(meses[i], fontsize=s_n, fontweight="bold")
            ax.grid()

            for tick in ax.get_xticklabels():
                tick.set_rotation(45)  # Rotar los ticks para mejorar legibilidad, alpha=.9

            if i == 0:
                ax.set_ylabel(dict_var[var][1],fontsize=13)

            fig.suptitle(f"Variable: {dict_var[var][0]} - Estacion {sta}", fontsize=18, fontweight="bold")


        plt.tight_layout()
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()

        plt.savefig(output_path + '/' + var + '/'+'Trend_graph_CC' + '_' + str(sta.replace('.', '_')) + '_' + var, dpi=300)
        plt.close()