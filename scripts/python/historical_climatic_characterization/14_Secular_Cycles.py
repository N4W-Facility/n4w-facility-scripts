# -*- encoding: utf-8 -*-
# !/usr/bin/python
from pylab import *
import pandas as pd
import scipy.stats as ss
import calendar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *
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
import numpy as np
from scipy.signal import periodogram

def cdi(df_input):
    """
    This functions calculates the curve of integrated differences.
    :param df_input:
    :return:
    """
    mean = df_input.mean()
    std = df_input.std()
    cv = mean / std
    k = ((df_input / mean) - 1) / cv
    zita = k.cumsum()
    return zita

def cid_plot(cdi, name, basicdata_freq, savefig=False, namefig=None):
    """
    This function plots the Curve of Integrated Differences.
    :param cdi: Curve of Integrated Differences data.
    :param name: Name for the plot.
    :param basicdata_freq: Frequency of the basic data.
    :param dict_times: Dictionary mapping frequencies to time units.
    :param savefig: save figure.
    :param namefig: figure's name.
    :return:
    """
    dict_times = {'D': r'$Dias$', 'MS': r'$Meses$', 'H': r'$Horas$'}
    fig, ax = plt.subplots()
    cdi.plot(style='-k', linewidth=2., ax=ax)
    diff = cdi.diff()
    diff.plot(style='k', linewidth=.6, ax=ax)
    ax.fill_between(cdi.index, diff, where=diff >= 0, color='blue', alpha=.6)
    ax.fill_between(cdi.index, diff, where=diff <= 0, color='red', alpha=.6)
    ax.axhline(color='k')
    ax.set_title('Curva de Diferencias Integradas ({})'.format(name))
    ax.set_ylabel(r'$\xi$')
    ax.set_xlabel('Tiempo [{}]'.format(dict_times[basicdata_freq]))
    plt.tight_layout()
    if savefig:
        if namefig is None:
            namefig = str(name) + '_cdi'
        plt.savefig(namefig, dpi=600)
        plt.close()
    else:
        plt.show()



if __name__ == "__main__":


    os.chdir(r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization')




    file_day = pd.ExcelFile('04_Datos_Sel_sin_outliers.xlsx')

    file_mon = pd.ExcelFile('02_Monthly/05_Monthly_Series.xlsx')

    file_year = pd.ExcelFile('03_Yearly/06_Yearly_Series.xlsx')


    variables = file_day.sheet_names

    for var in variables[3:]:
        print(var)
        if not os.path.exists('02_Monthly/07_Cycles/'+var):
            os.makedirs('02_Monthly/07_Cycles/'+var)

        path_graficas = '02_Monthly/07_Cycles/'+var+'/'

        df_day = file_day.parse(var,index_col=0)
        df_m = file_mon.parse(var, index_col=0)
        df_year= file_year.parse(var, index_col=0)

        stations = df_day .columns

        for sta in stations[0:]:
            print(sta)
            cdi_d = cdi(df_day[sta].dropna())
            cdi_m = cdi(df_m[sta].dropna())
            cdi_y = cdi(df_year[sta].dropna())



            fig = plt.figure(figsize=(11, 9))
            # fig.suptitle(r'Curvas de diferencias integrales', fontsize=16)
            grid = gridspec.GridSpec(nrows=6, ncols=4, hspace=1.4, wspace=0.8, figure=fig)


            # Dias

            ax1 = fig.add_subplot(grid[0:2, 0:], )  # mensuales multianuales
            ax1.set_title('Curvas de diferencias integrales', fontdict={'fontsize': 16, 'weight': 'bold' })


            ax_1 = ax1.twinx()

            # ax_1.invert_yaxis()
            diff_day = cdi_d.diff()

            ax_1.fill_between(cdi_d.index, diff_day, where=diff_day >= 0, color='blue', alpha=.1)
            ax_1.fill_between(cdi_d.index, diff_day, where=diff_day <= 0, color='red', alpha=.1)
            cdi_d.plot(style='-k', linewidth=1.2, ax=ax1)


            ax1.set_ylabel(r'$CDI$')
            ax_1.set_ylabel(r'$\xi CDI$')

            ax1.set_xlabel('Tiempo [{}]'.format(r'$Dias$'))
            plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)

            # Meses

            ax2 = fig.add_subplot(grid[2:4, 0:], )

            ax_2 = ax2.twinx()

            diff_m = cdi_m.diff()
            cdi_m.index =diff_m.index

            cdi_m.plot(style='-k', linewidth=1.2, ax=ax2)


            ax_2.fill_between(diff_m.index, diff_m, where=diff_m >= 0, color='blue', alpha=.3)
            ax_2.fill_between(diff_m.index, diff_m, where=diff_m <= 0, color='red', alpha=.3)



            ax2.set_ylabel(r'$CDI$')
            # ax_2.set_ylabel(r'$\xi CDI$')
            ax2.set_xlabel('Tiempo [{}]'.format('$Meses$'))
            plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)
            # años

            ax3 = fig.add_subplot(grid[4:6, 0:], )

            cdi_y.plot(style='-k', linewidth=1.2, ax=ax3)
            ax_3 = ax3.twinx()
            diff_y = cdi_y.diff()
            ax_3.fill_between(diff_y.index, diff_y, where=diff_y >= 0, color='blue', alpha=.3)
            ax_3.fill_between(diff_y.index, diff_y, where=diff_y <= 0, color='red', alpha=.3)
            ax3.set_ylabel(r'$CDI$')
            ax_3.set_ylabel(r'$\xi CDI$')
            ax3.set_xlabel('Tiempo [{}]'.format( r'$Años$'))
            plt.grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)
            # # plt.tight_layout()
            plt.savefig(path_graficas+ 'CDI_' + str(sta) + '_' + var + '.png')
            plt.close()
            # plt.show()











