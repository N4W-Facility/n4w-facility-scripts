# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import scipy.stats as ss


path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/' # general donde esta toda la informacion

path_out = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly/'

variables = ['PT_4','QL_1','TS_2','TS_3','RS','Vwind','Uwind','HR_1','QL_1']##'BS_4','EV_4'

if not os.path.exists(path_out + '04_Seasonality/'+'01_Moments/'):
    os.makedirs(path_out +'04_Seasonality/'+'01_Moments/')

for v in variables[0:]:
    print(v)

    excel_entrada = path +r'01_Groups/'+ v+'_groups_M.xlsx'

    excel_pdf =path +'03_Fit_PDF/PDF_'+v+'.xlsx'

    excel_salida = path_out + '04_Seasonality/'+'01_Moments/'+'/Moments_'+v+'.xlsx'

    libro = pd.ExcelFile(excel_entrada)

    object = ss

    pdf =pd.read_excel(excel_pdf,index_col=0) #pd.read_excel(excel_pdf, sheetname = v, index_col=0)

    pestanas = pdf.index#libro.sheet_names
    #print pdf.to_string()

    # Se prepara un libro de Excel NUEVO en el que se guardaran los momentos estadísticos de cada pestaña por cada mes
    libro_excel = pd.ExcelWriter(excel_salida)

    # Segundo paso Leer datos, por pestañas
    for p in pestanas[0:]:
        #datos = pd.read_excel(excel_entrada, sheetname=str(p), index_col=0)

        datos = pd.read_excel(excel_entrada,str(p),index_col = 0)

        columnas = datos.columns.values
        print (p)
        # Tercer paso calcular parametros de la distribución teórica $p(x)$
        lm = []
        for c in columnas[0:]:

            # print ('Distribución:', pdf[c])#[int(p)], '**', p, c

            distr =pdf[c][p]# pdf[c]#[int(p)]
            cur_dist = getattr(object, distr)
            # print(distr)

            data = datos[c].dropna()
            if np.all(data == 0):

                # Elegir una posición aleatoria
                random_index = np.random.randint(0, len(data))
                # Colocar un "1" en esa posición
                data[random_index] = 1

            pars=cur_dist.fit(data)
            xmin = data.min()
            xmax = data.max()

            m1 = cur_dist.moment(1, *pars)
            m2 = cur_dist.moment(2, *pars)
            m3 = cur_dist.moment(3, *pars)
            x0 = data - m1
            try:
                pars_c = cur_dist.fit(x0)
                pdc_mu = cur_dist.pdf(x0, *pars_c)
                mu1 = cur_dist.moment(1, *pars_c)
                mu2 = cur_dist.moment(2, *pars_c)
                mu3 = cur_dist.moment(3, *pars_c)
            except:
                mu1 = np.nan
                mu2 = np.nan
                mu3 = np.nan
            sigma = mu2**0.5
            cv = sigma / m1
            cs = mu3 / sigma**3
            m1, v2, s = cur_dist.stats(*pars, moments='mvs')
            m, v = np.mean(datos[c]), np.std(datos[c])/np.mean(datos[c])

            # print (c, '=====>', m1, m2, m3, mu1, mu2, mu3, sigma, cv, cs, m, v**0.5, s)
            res = [c, m1, m2, m3, mu1, mu2, mu3, sigma, cv, cs, m*1.0, v**0.5, s*1.0]
            lm.append(res)
        lm_df = pd.DataFrame(lm,columns=['mes', 'm1', 'm2', 'm3', 'mu1', 'mu2', 'mu3', 'sigma', 'cv', 'cs', 'm', 'sigma_0', 's'])
        lm_df.to_excel(libro_excel, sheet_name=str(p))

    #Quinto guardar datos complementados en el libro de Excel que se guardará al final
    libro_excel.close()
