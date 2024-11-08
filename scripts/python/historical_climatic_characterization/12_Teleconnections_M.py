import pandas as pd
import os
from pylab import *
import seaborn as sns
import matplotlib.pyplot as plt


plugin_dir = os.path.dirname(__file__)
in_file_indicadores = plugin_dir+'/Series_indices_2021.xlsx'

if __name__ == "__main__":


    os.chdir(r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization')



    in_file_variable = '02_Monthly/05_Monthly_Series.xlsx'


    sns.set(font_scale=1.3)

    # dataframe de indicadores

    file_var = pd.ExcelFile(in_file_variable)
    variables = file_var.sheet_names

    ind = pd.read_excel(in_file_indicadores,sheet_name='ENSO_indices', index_col=0)
    for vs in variables[0:]:
        print(vs)
        if not os.path.exists('02_Monthly/05_Teleconnections/01_Graphics/' + str(vs) + '/'):
            os.makedirs('02_Monthly/05_Teleconnections/01_Graphics/' + str(vs) + '/', exist_ok=True)
        out_file ='02_Monthly/05_Teleconnections/'+str(vs)+'_Teleconnections.xlsx'
        xlswriter = pd.ExcelWriter(out_file)

        var = file_var.parse(vs,index_col=0)


        var = var - var.mean()
        columnas = var.columns

        for c in columnas:
            print(c)
            v = pd.DataFrame(var[c])
            n = v.size / 12

            for i in range(-12, 13, 1):
                # Se prepar<n los rezagos de la variable y se guardan en el dataframe var
                v['{:+03d}'.format(i)] = var[c].shift(periods=i, freq=None, axis=0)
            v.drop([c], axis=1, inplace=True)
            # Añadimos rezagos de la variable hidroclimática a un dataframe temporal
            temp = pd.concat([ind, v], axis=1, join='inner')
            # temp.to_excel('temporal_rezagos.xlsx')
            corr_mat = temp.corr()
            error_r = (1 - corr_mat) / ((n + 1) ** 0.5)
            remover_columnas =  ind.columns #['Mei', 'MEI', 'NINO_1+2', 'NINO_3', 'NINO_34', 'NINO_4', 'ONI', 'SOI', 'WHWP']
            remover_filas = ['-01', '-02', '-03', '-04', '-05', '-06', '-07', '-08', '-09', '-10', '-11', '-12', '+00',
                             '+01', '+02', '+03', '+04', '+05', '+06', '+07', '+08', '+09', '+10', '+11', '+12']

            corr_mat.drop(remover_filas, axis=0, inplace=True)
            corr_mat.drop(remover_columnas, axis=1, inplace=True)
            error_r.drop(remover_filas, axis=0, inplace=True)
            error_r.drop(remover_columnas, axis=1, inplace=True)
            sign_corr = corr_mat.where(np.abs(corr_mat) > error_r, 0)
            sign_corr = pd.DataFrame(sign_corr, columns=corr_mat.columns, index=corr_mat.index)

            plt.subplots(figsize=(18, 12))
            sns.heatmap(sign_corr, annot=True, fmt='.2f', linewidths=.5, cmap="coolwarm_r", vmin=-1, vmax=1,
                        square=False)
            plt.title('Estacion: '+str(c))
            plt.xlabel('Rezagos')
            plt.ylabel('Indices')
            plt.tight_layout()

            if len(str(c)) > 31:
                name = str(c)[:-5]
                if len(str(name)) > 31:
                    name = name[:-10]
            else:
                name = c



            # plt.show()
            plt.savefig('02_Monthly/05_Teleconnections/01_Graphics/'+str(vs)+'/'+ str(vs) +'_'+ str(name) + '_Teleconexiones.png')
            # plt.show()
            plt.close()


            sign_corr.to_excel(xlswriter, sheet_name=str(name))
            # error_r.to_excel(xlswriter, sheet_name=c + '_Er')
        xlswriter.close()