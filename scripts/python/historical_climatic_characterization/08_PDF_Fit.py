# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import math as mt
import os
import matplotlib as mpl

label_size = 11
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

def ckolmo(obs):
    # implementa p(lamda)
    # Obs es el valor de lamda empirico
    # Definicion de los niveles de significancia
    # vns = [0.4000, 0.3000, 0.2000, 0.1000, 0.0500, 0.0250, 0.0100, 0.0050, 0.0010, 0.0005]
    vns = [obs]
    lvns = len(vns)
    ckd = []
    for i in range(lvns):
        ckd.append([0])
    for i in range (lvns):
        ckd[i] = (-np.log(vns[i]/2.0)/np.log(mt.e)/2.0)**(1/2.0)
    return ckd

def Kolmogorov(pe, pt, alfa):

    D=np.max(np.abs(pe-pt))
    lamda=D*np.sqrt(np.size(pe,axis=0))
    # lamda es el lamda empirico
    lamdat=ckolmo(alfa)
    if lamda<=lamdat:
        res=1
    else:
        res=0
    return res


def ajustarpdf(df, pestana, variable,path=None,mostrar=True, guardar=True):

    cols = df.columns.values
    alfa=0.05
    object=st
    datos=np.asarray(df)
    dist_continu=['norm', 'lognorm', 'gamma', 'loggamma', 'genextreme', 'weibull_max',
             'weibull_min', 'expon', 'powerlaw', 'pearson3', 'gumbel_l', 'gumbel_r']
    res_list = []
    for j in range(cols.size):
        print ('Pestana=', pestana, 'Columna=', cols[j])
        i = 1
        plt.figure(figsize=(15.0, 10.0))
        for fdist in dist_continu:
            cur_dist=getattr(object, fdist)
            numargs=cur_dist.numargs
            a = ([0.9,]*numargs)
            #OJO A LOS NAN, BIEN PUEDE SER ESO O PUEDEN SER LOS DATOS ANOMALOS
            data = np.asarray(pd.Series(datos[:,j]).dropna())
            # Verificar si todos los elementos son cero
            if np.all(data == 0):

                # Elegir una posición aleatoria
                random_index = np.random.randint(0, len(data))
                # Colocar un "1" en esa posición
                data[random_index] = 1
            pars=cur_dist.fit(data)
            sort_data=np.sort(data, axis=0)


            pe=np.arange(1.0, np.size(sort_data)+1)/(np.size(sort_data)+1)
            pt=cur_dist.cdf(sort_data,*pars)
            error_medio_abs=np.mean(np.abs((pe-pt)/pe))
            error_max_abs=np.max(np.abs((pe-pt)/pe))
            res_Kolmogorov = Kolmogorov(pe, pt, alfa)
            #print cur_dist.name, 'Errores=> ', 100*error_medio_abs, 100*error_max_abs, res_Kolmogorov, pars
            res_list.append([pestana, cols[j], cur_dist.name, res_Kolmogorov, 100*error_medio_abs, 100*error_max_abs])
            plt.subplot(3,4,i, aspect='auto')
            i+=1
            titulo = str(cur_dist.name) + ' K=' + str(res_Kolmogorov) + ' Em='+str(round(error_medio_abs*100,2)) + ' Emax='+str(round(error_max_abs*100,2))
            plt.title(titulo, fontsize=10)
            plt.plot(sort_data, 1-pe, '.k', sort_data, 1-pt, '-k')
            plt.tight_layout()
            plt.xlabel('x')
            plt.ylabel('F(x)')
        plt.legend(['p_empirica', 'p_teorica'], loc=4, fontsize=10)
        if guardar:
            if not os.path.exists(path +'/Graphs/' + variable):
                os.makedirs(path +'/Graphs/' + variable)
            plt.savefig(path+'/Graphs/'+variable+'/fig'+'_'+str(pestana)+"_"+variable+"_"+"{:0>2d}".format(j+1))
        else:
            pass
        if mostrar:
            plt.show()
        else:
            pass
        plt.close()
    return res_list

path =r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly\01_Groups/'

path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/'


if not os.path.exists(path_out + '03_Fit_PDF/'):
    os.makedirs(path_out + '03_Fit_PDF/')


# if not os.path.exists(path_out+ '02_Ajuste/' ):
#     os.makedirs(path_out+ '02_Ajuste/' )

variables = ['QL_1','TS_2','TS_3','RS','Vwind','Uwind','HR_1','PT_4']# ENTRADA#'PT_4','PT_9','TS_1','TS_2','TS_8','BS_4', 'EV_4', 'HR_1','QL_1', 'PT_4','PT_9','TS_1','TS_2','TS_8','BS_4',

for v in variables[0:]:

    print(v)

    excel_entrada = path + v+'_groups_M.xlsx'

    excel_salida = path_out + '03_Fit_PDF/' + 'Fit_' + v + '.xlsx' #'02_Ajuste/' +


    excel_writer = pd.ExcelWriter(excel_salida)

    Libro = pd.ExcelFile(excel_entrada)
    estaciones = Libro.sheet_names


    col_names = ['Estacion', 'Columna', 'Distribucion', 'Kolmogorov', 'Error Medio', 'Error Maximo']

    ii = 1

    for e in estaciones[0:]:

        print (e)


        data = pd.read_excel(excel_entrada, str(e), index_col = 0)
        if ii  >= 10:
            pdflist = ajustarpdf(data, e,v,path=path_out + '03_Fit_PDF/', mostrar=False, guardar=False)
        else:
            pdflist = ajustarpdf(data, e, v, path=path_out + '03_Fit_PDF/', mostrar=False, guardar=True)
        a = np.array(pdflist)

        if len(str(e)) > 31:
            name = str(e)[:-5]
            if len(str(e)) > 31:
                name = str(e)[:-10]
        else:
            name = e

        res = pd.DataFrame(a, columns=col_names)
        res.to_excel(excel_writer, sheet_name=str(name), merge_cells=True, index=None)

        ii+=1
    excel_writer.close()
