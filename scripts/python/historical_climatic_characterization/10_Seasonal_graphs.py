

# -*- encoding: utf-8 -*-


from pylab import *
import pandas as pd
import os
matplotlib.style.use('ggplot')




os.chdir(r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization\02_Monthly\04_Seasonality/')

variable = ['QL_1','TS_2','TS_3','RS','HR_1','PT_4']
dict_months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sept', 'Oct', 'Nov', 'Dic']



for v in variable[0:]:

    df = pd.read_excel('01_monthly_multiannuals.xlsx', v, index_col=0)# si se desea se plotea 01_mensuales_multianuales

    df.columns = dict_months
    for i in range(np.size(df, axis=0))[0:]:
        print(i, df.index[i])
        namefig = df.index[i]
        plt.rcParams['grid.color'] = 'k'
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['grid.linewidth'] = 0.8

        figsize = (13, 9)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        size_line = 1.5
        type_line = 'None'  # '-'#
        value = df.iloc[i]
        if v =='TS_1' or v =='TS_2' or v =='TS_8' or v =='TS_3' or  v =='HR_1':
            ax.plot(value.index, value,
                    c='black', linewidth=2.5, linestyle='--', alpha=0.9, label='Historico')

            # fig = df.iloc[i].plot(kind='line', color='grey', linewidth=2.0)
            plt.xticks(np.arange(12) ,df.iloc[i].index, rotation=45)
        else:
            value = df.iloc[i]
            plt.bar(value.index, value, color='black', alpha=.7, label='Historico')
            # fig = df.iloc[i].plot(kind='bar', color='grey', linewidth=2.0)
            plt.xticks( rotation=45)


        plt.rcParams['grid.color'] = 'k'
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['grid.linewidth'] = 0.8

        plt.title(str(namefig))
        ax.set_title('Variable: ' + v+ ' Estacion: '+str(namefig),
                     fontdict={'fontsize': 18, 'fontweight': "bold"})
        s_n = 14
        if v == 'TS_1':
            ax.set_ylabel('{} [{}]'.format('TS', 'C'), fontsize=s_n,fontweight="bold")
        elif v == 'ETP_1':
            ax.set_ylabel('{} [{}]'.format('ETP', 'mm'), fontsize=s_n,fontweight="bold")
        elif v == 'ETP_8':
            ax.set_ylabel('{} [{}]'.format('ETP', 'mm'), fontsize=s_n,fontweight="bold")
        elif v == 'ETP_2':
            ax.set_ylabel('{} [{}]'.format('ETP', 'mm'), fontsize=s_n,fontweight="bold")
        elif v == 'TS_2':
            ax.set_ylabel('{} [{}]'.format('TS', 'C'), fontsize=s_n,fontweight="bold")
        elif v == 'TS_8':
            ax.set_ylabel('{} [{}]'.format('TS', 'C'), fontsize=s_n,fontweight="bold")
        elif v == 'TS_3':
            ax.set_ylabel('{} [{}]'.format('TS', 'C'), fontsize=s_n,fontweight="bold")
        elif v == 'PT_4':
            ax.set_ylabel('{} [{}]'.format('PT', 'mm'), fontsize=s_n,fontweight="bold")

        elif v == 'PT_9':
            ax.set_ylabel('{} [{}]'.format('PT_9', 'mm'), fontsize=s_n,fontweight="bold")
        elif v == 'BS_4':
            ax.set_ylabel('{} [{}]'.format('BS', 'Horas'), fontsize=s_n,fontweight="bold")

        elif v == 'RS':
            ax.set_ylabel('{} [{}]'.format('RS', ''), fontsize=s_n,fontweight="bold")

        elif v == 'EV_4':
            ax.set_ylabel('{} [{}]'.format('EV', 'mm'), fontsize=s_n,fontweight="bold")

        elif v == 'HR_1':
            ax.set_ylabel('{} [{}]'.format('HR', '%'), fontsize=s_n,fontweight="bold")

        elif v == 'NV_1':
            ax.set_ylabel('{} [{}]'.format('NV', 'cm'), fontsize=s_n,fontweight="bold")
        elif v == 'NV_2':
            ax.set_ylabel('{} [{}]'.format('NV', 'cm'), fontsize=s_n,fontweight="bold")


        else:
            ax.set_ylabel('{} [{}]'.format('QL', '(m3/s)'))  # fig.set_ylabel('{} [{}]'.format('Y', 'mm'))
        plt.tight_layout()

        if not os.path.exists('02_Graphics/'  +str(v) + '/'):
            os.makedirs('02_Graphics/' +str(v) + '/',exist_ok=True)

        plt.savefig('02_Graphics/'  +str(v) + '/' + str(v) + '_' + str(namefig))
        plt.close()

