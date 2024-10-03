# Import necessary libraries for data processing, geospatial manipulation, and plotting
import functools as ft
import pandas as pd
from osgeo import gdal, osr
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa
from matplotlib import gridspec
from rasterio.io import MemoryFile
from affine import Affine
matplotlib.style.use('ggplot')
import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from sklearn.metrics import r2_score
import warnings

# Function to read monthly Excel files for different models and scenarios
def read_all_xlsfiles_m(models, scenarios, path, var):
    """
    Reads monthly Excel files for a specific variable, model, and scenario.

    Parameters:
    - models: List of climate models.
    - scenarios: List of climate scenarios.
    - path: Path to the folder containing Excel files.
    - var: Variable name (e.g., 'PT_4', 'TS_1').

    Returns:
    - Dictionary where keys are model names and values are data for each scenario.
    """
    list_file = {}
    for mn in models:
        file = pd.ExcelFile(path + 'Monthly_Series_' + mn + '_' + var + '.xlsx')
        list_file[mn] = {esc: file.parse(esc, index_col=0).loc[:'31-12-2100'] for esc in scenarios}
    return list_file

# Function to read yearly Excel files for different models and scenarios
def read_all_xlsfiles_yearly(models, scenarios, path, var):
    """
    Reads yearly Excel files for a specific variable, model, and scenario.

    Parameters:
    - models: List of climate models.
    - scenarios: List of climate scenarios.
    - path: Path to the folder containing Excel files.
    - var: Variable name (e.g., 'PT_4', 'TS_1').

    Returns:
    - Dictionary where keys are model names and values are data for each scenario.
    """
    list_file = {}
    for mn in models:
        file = pd.ExcelFile(path + 'Yearly_Series_' + mn + '_' + var + '.xlsx')
        list_file[mn] = {esc: file.parse(esc, index_col=0).loc[:'31-12-2100'] for esc in scenarios}
    return list_file

# Function to read daily CSV files for different models and scenarios
def read_all_xlsfiles_day(models, scenarios, path, var):
    """
    Reads daily CSV files for a specific variable, model, and scenario.

    Parameters:
    - models: List of climate models.
    - scenarios: List of climate scenarios.
    - path: Path to the folder containing CSV files.
    - var: Variable name (e.g., 'PT_4', 'TS_1').

    Returns:
    - Dictionary where keys are model names and values are data for each scenario.
    """
    list_file = {}
    for mn in models:
        new_path = path + mn + '/' + var + '/'
        list_file[mn] = {esc: pd.read_csv(new_path + f'Ds_Series_{esc}_{var}.csv', index_col=0).loc[:'31-12-2100'] for esc in scenarios}
    return list_file

# Function to calculate the convergence factor of a model compared to other models
def convergence_factor(model, other_models):
    """
    Calculates the mean absolute difference between a model and all other models.

    Parameters:
    - model: Data for the model being compared.
    - other_models: List of data for other models.

    Returns:
    - Mean absolute difference between the model and other models.
    """
    diffs = [np.abs(model - m) for m in other_models]
    diff = np.mean(diffs, axis=0)
    return pd.DataFrame(diff).mean()

# Function to calculate Mean Squared Error (MSE)
def mse(predictions, observations):
    """
    Computes the Mean Squared Error (MSE) between model predictions and observations.

    Parameters:
    - predictions: Model predictions.
    - observations: Observed data.

    Returns:
    - Mean Squared Error between predictions and observations.
    """
    return np.mean((predictions - observations)**2, axis=0)

# Function to calculate the performance factor based on MSE
def performance_factor(predictions, observations):
    """
    Computes the performance factor, which is the inverse of the Mean Squared Error (MSE).

    Parameters:
    - predictions: Model predictions.
    - observations: Observed data.

    Returns:
    - Performance factor calculated as 1 / MSE.
    """
    mse_value = mse(predictions, observations)
    return 1 / mse_value

# Function to calculate model weights based on performance and convergence
def calculate_weights(hist_models, observations):
    """
    Calculates weights for each model based on its performance and convergence with other models.

    Parameters:
    - hist_models: List of historical model data.
    - observations: Observed data.

    Returns:
    - Normalized weights for each model.
    """
    weights = []
    for mm in hist_models:
        c = convergence_factor(mm, [m for m in hist_models if not np.array_equal(m, mm)])
        D = performance_factor(mm, observations)
        weights.append(pd.DataFrame(c.values * D.values))
    total_weight = np.sum(weights, axis=0)
    normalized_weights = [w / total_weight for w in weights]
    return normalized_weights

# Function to format DataFrame for CSV output
def ok_to_csv_m(df):
    """
    Formats a DataFrame by adding Year and Month columns and rearranging columns.

    Returns:
    - Formatted DataFrame ready for CSV output.
    """
    df.index = pd.DatetimeIndex(df.index)

    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df = df.reset_index()
    df = df.rename(columns={"index": "Date"})
    df = df.drop(columns=['Date'])
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df_result = df[cols]
    return df_result

# Function to add leap years and interpolate missing data
def add_leap(df, start='1980-01-01', end='2014-12-31'):
    """
    Adds leap years to the dataset and interpolates any missing values.

    Parameters:
    - df: DataFrame to be processed.
    - start: Start date of the date range.
    - end: End date of the date range.

    Returns:
    - DataFrame with leap years added and missing values interpolated.
    """
    rango_fechas = pd.date_range(start=start, end=end)
    df_fechas = pd.DataFrame(index=rango_fechas)
    df.index = pd.to_datetime(df.index).date
    df_result = pd.merge(df_fechas, df, left_index=True, right_index=True, how="left")
    df_result.interpolate(method='linear', inplace=True)
    return df_result

# Function to calculate the Taylor Skill Score for a model
def taylor_skill_score(obser_df, model):
    """
    Calculates the Taylor Skill Score for a model based on its performance relative to observed data.

    Parameters:
    - obser_df: Observed data.
    - model: Model data.

    Returns:
    - Taylor Skill Score for the model.
    """
    obser_df = obser_df.loc['1980-01-01':'2014-12-31']
    model = model.loc['1980-01-01':'2014-12-31']

    df_model = add_leap(model)
    observations = add_leap(obser_df)

    obs_std = np.std(observations,axis=0)
    model_std = np.std(df_model,axis=0)
    mse_model = np.mean((df_model - observations)**2,axis=0)

    correlation = df_model.apply(lambda col: np.corrcoef(col, observations[col.name].interpolate(method='linear', limit_direction='both'))[0, 1])

    numerator = obs_std**2
    denominator = obs_std**2 + (obs_std**2 / model_std**2) * (mse_model - 2 * correlation * model_std * obs_std)

    tss = numerator / denominator
    return tss.median()

class TaylorDiagram(object):
    def __init__(self, STD, fig=None, rect=111, label='_'):
        self.STD = STD
        tr = PolarAxes.PolarTransform()
        # Correlation labels
        rlocs = np.concatenate(((np.arange(11.0) / 10.0), [0.95, 0.99]))
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs)  # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        # Standard deviation axis extent
        self.smin = 0
        self.smax = 1.6 * self.STD
        gh = fa.GridHelperCurveLinear(tr, extremes=(0, (np.pi / 2), self.smin, self.smax), grid_locator1=gl1,
                                      tick_formatter1=tf1, )
        if fig is None:
            fig = plt.figure()
        ax = fa.FloatingSubplot(fig, rect, grid_helper=gh)
        fig.add_subplot(ax)
        # Angle axis
        ax.axis['top'].set_axis_direction('bottom')
        ax.axis['top'].label.set_fontsize(10)  # revisar
        ax.axis['top'].label.set_text("Correlation coefficient")
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].label.set_axis_direction('top')
        # X axis
        ax.axis['left'].set_axis_direction('bottom')
        ax.axis['left'].label.set_fontsize(10) # revisar
        ax.axis['left'].label.set_text("Standard deviation")
        ax.axis['left'].toggle(ticklabels=True, label=True)
        ax.axis['left'].major_ticklabels.set_axis_direction('bottom')
        ax.axis['left'].label.set_axis_direction('bottom')
        # Y axis
        ax.axis['right'].set_axis_direction('top')
        ax.axis['right'].label.set_fontsize(10)  # revisar
        ax.axis['right'].label.set_text("Standard deviation")
        ax.axis['right'].toggle(ticklabels=True, label=True)
        ax.axis['right'].major_ticklabels.set_axis_direction('left')
        ax.axis['right'].label.set_axis_direction('top')
        # Useless
        ax.axis['bottom'].set_visible(False)
        # Contours along standard deviations
        ax.grid()
        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates
        # Add reference point and STD contour
        l, = self.ax.plot([0], self.STD, 'k*', ls='', ms=12, label=label)
        l1, = self.ax.plot([0], self.STD, 'k*', ls='', ms=12, label=label)
        t = np.linspace(0, (np.pi / 2.0))
        t1 = np.linspace(0, (np.pi / 2.0))
        r = np.zeros_like(t) + self.STD
        r1 = np.zeros_like(t) + self.STD
        self.ax.plot(t, r, 'k--', label='_')
        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]
        self.samplePoints = [l1]

    def add_sample(self, STD, r, *args, **kwargs):
        l, = self.ax.plot(np.arccos(r), STD, *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)
        return l

    def add_sample(self, STD, r1, *args, **kwargs):
        l1, = self.ax.plot(np.arccos(r1), STD, *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l1)
        return l1

    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax), np.linspace(0, (np.pi / 2.0)))
        RMSE = np.sqrt(np.power(self.STD, 2) + np.power(rs, 2) - (2.0 * self.STD * rs * np.cos(ts)))
        contours = self.ax.contour(ts, rs, RMSE, levels, **kwargs)
        return contours

def moving_average_rmse(observed, *models, window_size=5):
    """
    Calcula el RMSE de la media móvil de una ventana de tamaño window_size.

    Args:
    - observed: serie de tiempo observada
    - *models: series de tiempo de los modelos (pueden ser varios)
    - window_size: tamaño de la ventana de la media móvil

    Returns:
    - rmse_values_list: Lista de listas con los valores RMSE para cada ventana por modelo
    """
    rmse_values_list = []

    for model in models:
        rmse_values = []
        for i in range(len(observed) - window_size + 1):
            observed_window = observed[i:i + window_size]
            model_window = model[i:i + window_size]
            n = observed_window.columns[0]
            rmse_values.append(np.sqrt(np.mean((observed_window[n] - model_window) ** 2)))

        rmse_values_list.append(rmse_values)

    return rmse_values_list
# Function to plot and save the Taylor Skill Score for various models
def graficar_taylor_skill_score(models, tss_scores, path_out):
    """
    Plots the Taylor Skill Score for multiple models and saves the plot as a PNG file.

    Parameters:
    - models: List of model names.
    - tss_scores: List of Taylor Skill Scores.
    - path_out: Output path to save the plot.
    """
    sorted_indices = sorted(range(len(tss_scores)), key=lambda k: tss_scores[k], reverse=True)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_tss_scores = [tss_scores[i] for i in sorted_indices]


    fig, ax = plt.subplots(figsize=(10, 6))
    bars = plt.bar(sorted_models, sorted_tss_scores, color='gray',)

    plt.ylabel('Taylor Skill Score')
    plt.xlabel('Modelos')
    plt.title('GCMs Taylor Skill Score')
    plt.ylim(0, 2)  # El Taylor Skill Score varía de 0 a 1

    # Rotar las etiquetas del eje x para evitar que se sobrepongan
    plt.xticks(fontsize=9)

    # Añadir etiquetas de valor en cada barra
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

    # Mostrar el gráfico
    plt.savefig(path_out , dpi=300)
    plt.close()

# Function to generate a Taylor diagram for model comparison
def graphics_taylor_evaluation(data_obs, data_obs_year, input_hist_models_year, input_hist_models, output_path):
    """
    Creates a Taylor diagram comparing multiple models based on observed data.

    Parameters:
    - data_obs: Observed daily data.
    - data_obs_year: Observed yearly data.
    - input_hist_models_year: Historical model data (yearly).
    - input_hist_models: Historical model data (daily).
    - output_path: Path to save the Taylor diagram.

    Returns:
    - Saves the generated Taylor diagram.
    """
    hist_models_Day = input_hist_models
    hist_models_year = input_hist_models_year
    select_models_day = [add_leap(his_model).loc['1980-01-01':'2014-12-31'] for his_model in hist_models_Day]
    select_models_year = [his_model.loc['1980':'2014'] for his_model in hist_models_year]
    for sta in data_obs.columns:
        print(sta)
        obs_data = add_leap(data_obs[sta]).loc['1980-01-01':'2014-12-31'].interpolate(method='linear',limit_direction='both')
        obs_year = data_obs_year[sta].loc['1980':'2014'].interpolate(method='linear', limit_direction='both')
        ref_stddev = np.std(obs_data).values[0]
        model_stddevs = [np.std(modelo[sta].interpolate(method='linear', limit_direction='both')) for modelo in select_models_day]
        correlations = [np.corrcoef(modelo[sta].interpolate(method='linear', limit_direction='both'), obs_data[sta])[0, 1] for modelo in select_models_day]
        markers = ['o', '^', 's', '*', '+', 'x', 'D', '|', '_']
        fig = plt.figure(figsize=(10, 5))
        dict_var_g = {'TS_1': ['Mean Temperature', 'TS', 'C'], 'PT_4': ['Total Precipitations', 'PT', 'mm']}
        fig.suptitle(dict_var_g[var][0] + ' ' + str(sta), size='x-large')
        dia = TaylorDiagram(ref_stddev, fig=fig, rect=122, label='Obs')
        for i in range(7):
            dia.add_sample(model_stddevs[i], correlations[i], marker=markers[i], label=models[i])
        contours = dia.add_contours(levels=5, colors='0.5')
        dia.ax.clabel(contours, inline=1, fontsize=8, fmt='%.1f')
        plt.legend(dia.samplePoints, [p.get_label() for p in dia.samplePoints], numpoints=1, prop=dict(size='small'), loc='lower center', bbox_to_anchor=(0.5, 0.1), fontsize=9, ncol=2)
        ax1 = fig.add_subplot(221)
        window_size = 365
        modelos_sta = [modelo[sta].interpolate(method='linear', limit_direction='both') for modelo in select_models_day]
        rmse_lists = moving_average_rmse(obs_data, *modelos_sta, window_size=window_size)
        dates = obs_data.index
        average_rmse = [np.mean(rmse_values) for rmse_values in rmse_lists]
        best_model_index = np.argmin(average_rmse)
        for index, rmse_values in enumerate(rmse_lists):
            if index == best_model_index:
                label = f"{models[index]} (Best RMSE)"
            else:
                label = None
            ax1.plot(dates[window_size - 1:], rmse_values, marker=markers[index % len(markers)], markersize=1.0, linewidth=0.2, label=label)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel(f"{window_size}-day moving RMSE", fontsize=10)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        ax2 = fig.add_subplot(223)
        modelos_sta_year = [modelo[sta].interpolate(method='linear', limit_direction='both') for modelo in select_models_year]
        ax2.plot(obs_year.index, obs_year, color="black", linewidth=0.5)
        for index, modelo in enumerate(modelos_sta_year):
            ax2.scatter(modelo.index, modelo, marker=markers[index % len(markers)], s=7.5, label=models[index])
        ax2.set_xlabel("Date", fontsize=10)
        ax2.set_ylabel('{} ({})'.format(dict_var_g[var][1], dict_var_g[var][2]), fontsize=10)
        plt.tight_layout()
        name_out = sta
        plt.savefig(output_path + '/taylor_diagram_' + str(name_out) + '_' + var, dpi=300)
        plt.close()

# Main logic that runs the evaluation, generates graphs, and saves the results
BASE_PATH = r'TNC-N4WF\00_Example/' # path input project

models = ['ACCESS-ESM1-5','CanESM5','CESM2','EC-Earth3','MIROC6','MPI-ESM1-2-LR','MRI-ESM2-0']
escenarios = ['historical-obs','historical-gcm','ssp126', 'ssp245', 'ssp370', 'ssp585']
variables = ['PT_4','TS_1']
tipo='d'#'Monthly'# Daily or monthly

if __name__ == "__main__":
    os.chdir(BASE_PATH)
    for var in variables:
        print(var)
        if tipo == 'Monthly':
            path_in = '02_Climate_Change_Scenarios/03_monthly_Series/' + var + '/'
            output_path = os.path.join('02_Climate_Change_Scenarios', '05_Historical_Evaluation', var )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            dic_cc = read_all_xlsfiles_m(models, escenarios, path_in, var)
            dic_cc_year = read_all_xlsfiles_yearly(models, escenarios, path_in, var)
        else:
            path_in = '02_Climate_Change_Scenarios/02_Series_Downscaling/'
            path_in_y = '02_Climate_Change_Scenarios/03_monthly_Series/' + var + '/'
            output_path = os.path.join('02_Climate_Change_Scenarios', '05_Historical_Evaluation', var)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            dic_cc = read_all_xlsfiles_day(models, escenarios, path_in, var)
            dic_cc_year = read_all_xlsfiles_yearly(models, escenarios, path_in_y, var)

        obs_data = dic_cc[models[0]]['historical-obs']
        obs_yearly = dic_cc_year[models[0]]['historical-obs']
        hist_models_Day = [dic_cc[model]['historical-gcm'] for model in models]
        hist_models_year = [dic_cc_year[model]['historical-gcm'] for model in models]

        tss_scores = [taylor_skill_score(obs_data, his_model) for his_model in hist_models_Day]
        graficar_taylor_skill_score(models, tss_scores,output_path+'/TSS_'+var)
        graphics_taylor_evaluation(obs_data, obs_yearly, hist_models_year, hist_models_Day, output_path)