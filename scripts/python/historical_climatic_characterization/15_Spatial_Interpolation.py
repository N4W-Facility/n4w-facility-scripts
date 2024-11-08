import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from osgeo import gdal, osr, gdalconst

def pixel(xmax, xmin, ymax, ymin, nx, ny):
    pixelWidth = abs(xmin - xmax) / nx
    pixelHeight = abs(ymin - ymax) / ny
    return pixelWidth, pixelHeight

def array2raster(newRasterfn, array, input_dem):
    dem, _, cols, rows, xmin, xmax, ymin, ymax = LoadRaster(input_dem)
    geotransform = gdal.Open(input_dem).GetGeoTransform()
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    srs = osr.SpatialReference()
    srs.ImportFromWkt(gdal.Open(input_dem).GetProjection())
    epsg = int(srs.GetAttrValue('AUTHORITY', 1)) if srs.GetAttrValue('AUTHORITY', 1) else None
    # pixelWidth and pixelHeight are now obtained from the DEM geotransform directly

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((xmin, pixelWidth, 0, ymax, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    # if array.shape != (rows, cols):
    #     array = array[:rows, :cols]
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    if epsg:
        outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def IDW(x, y, z, input_dem, power=2):
    def distancia(x, y, xi, yi):
        obs = np.vstack((x, y)).T
        interpolar = np.vstack((xi, yi)).T
        d0 = np.subtract.outer(obs[:, 0], interpolar[:, 0])
        d1 = np.subtract.outer(obs[:, 1], interpolar[:, 1])
        return np.hypot(d0, d1)

    def malla(input_dem):
        dem_ds = gdal.Open(input_dem)
        geotransform = dem_ds.GetGeoTransform()
        cols = dem_ds.RasterXSize
        rows = dem_ds.RasterYSize
        xmin = geotransform[0]
        ymax = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]

        # Generate coordinates for each pixel
        xi = np.array([xmin + i * pixelWidth for i in range(cols)])
        yi = np.array([ymax + j * pixelHeight for j in range(rows)])
        xi, yi = np.meshgrid(xi, yi)
        return xi.flatten(), yi .flatten()

    xi, yi = malla(input_dem)
    dem, _, cols, rows, _, _, _, _ = LoadRaster(input_dem)
    dist = distancia(x, y, xi, yi)
    W = 1.0 / dist**power
    W /= W.sum(axis=0)
    zi = np.dot(W.T, z)
    return xi, yi, zi.reshape(rows, cols)

def LoadRaster(file_name):
    g = gdal.Open(file_name, gdalconst.GA_ReadOnly)
    g.GetRasterBand(1)
    nodatavalue = g.GetRasterBand(1).GetNoDataValue()
    cols, rows = g.RasterXSize, g.RasterYSize
    geotransform = g.GetGeoTransform()
    xmin = geotransform[0]
    ymax = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    xmax = xmin + (cols * pixelWidth)
    ymin = ymax + (rows * pixelHeight)
    grid = g.ReadAsArray(0, 0, cols, rows)
    return grid, nodatavalue, cols, rows, xmin, xmax, ymin, ymax

def procesar_datos_interpolacion(input_catalogo, input_mensuales, output_path, variables, variables_pt, variables_ts):
    file = pd.ExcelFile(input_catalogo)
    file_data = pd.ExcelFile(input_mensuales)
    os.makedirs(output_path, exist_ok=True)

    excel_salida = os.path.join(output_path, '01_Datos_Interpolacion_M.xlsx')
    libro_excel = pd.ExcelWriter(excel_salida)

    for var in variables:
        print(var)
        catalogo = file.parse(var, index_col=0)
        df = file_data.parse(var, index_col=0)

        if var in variables_pt:
            df['Anual'] = df.sum(axis=1)
        elif var in ['PT_9', 'TS_2', 'QL_2']:
            df['Anual'] = df.max(axis=1)
        elif var in ['TS_8', 'QL_3','TS_3']:
            df['Anual'] = df.min(axis=1)
        elif var in ['TS_1', 'QL_1']:
            df['Anual'] = df.mean(axis=1)

        df.columns = [var[:-2] + f"_{i:02}" for i in range(1, len(df.columns) + 1)]

        cols_select = ['NORTE', 'ESTE', 'ALTITUD'] if var in variables_ts else ['NORTE', 'ESTE']
        df_new = catalogo[cols_select].merge(df, left_index=True, right_index=True)
        df_new.to_excel(libro_excel, sheet_name=var)
    libro_excel.close()

def interpolar_idw(input_datos_interpolacion, output_path, variables_pt, input_dem):
    df_file = pd.ExcelFile(input_datos_interpolacion)


    for var in variables_pt:
        output_dir = os.path.join(output_path, 'Interpolacion', var)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(var)
        df = df_file.parse(var, index_col=0).interpolate(axis=0).fillna(0)
        meses = df.columns

        for m in meses[2:]:
            print(m)
            # dem, _, cols, rows, xmin, xmax, ymin, ymax = LoadRaster(input_dem)
            # nx, ny = cols, rows #500, 500
            y, x, z = df['NORTE'], df['ESTE'], df[m]
            outRuta = os.path.join(output_dir, f'{m}.tif')

            xi, yi, zi = IDW(x, y, z, input_dem)

            array2raster(newRasterfn=outRuta, array=zi, input_dem=input_dem)

def calcular_coef_isotermas(input_datos_interpolacion, output_path, variables_ts):
    for var in variables_ts:
        xls_save = os.path.join(output_path, f'COEF_ISOTERMAS_{var}.xlsx')
        xls_file = pd.ExcelFile(input_datos_interpolacion)
        data = xls_file.parse(var)
        fechas = data.columns[4:]
        xls_writer = pd.ExcelWriter(xls_save)
        nrl, ndata = len(fechas), data.shape[0]
        x = data['ALTITUD']
        X = sm.add_constant(x, prepend=False)
        header = ['COE_A', 'COE_B', 'MSE', 'RMSE', 'R2', 'Q_DATA', 'ERROR_STD_A', 'ERROR_STD_B',
                  'F-STATISTIC', 'pVALUE_A', 'pVALUE_B', 'T-STUDENT_A', 'T-STUDENT_B',
                  'UPPER_95%A', 'LOWER_95%', 'UPPER_95%_A', 'LOWER_95%_B']
        results = np.zeros([nrl, len(header)])

        for i, f in enumerate(fechas):
            model = sm.OLS(data[f], X, missing='drop')
            statistics = model.fit()
            TS_obs = data.loc[:, ['ALTITUD', f]].dropna(how='any')
            TS_obs['SIMULADOS'] = (TS_obs['ALTITUD'] * statistics.params[0]) + statistics.params[1]
            TS_obs['Errores'] = (TS_obs.loc[:, f] - TS_obs.loc[:, 'SIMULADOS']) ** 2
            mse = np.sum(np.asarray(TS_obs['Errores'])) / len(TS_obs)
            rmse = np.sqrt(mse)
            results[i] = [statistics.params[0], statistics.params[1], mse, rmse, statistics.rsquared,
                          statistics.nobs, statistics.bse[0], statistics.bse[1], statistics.f_pvalue,
                          statistics.pvalues[0], statistics.pvalues[1], statistics.tvalues[0], statistics.tvalues[1],
                          statistics.conf_int(alpha=0.05)[0][0], statistics.conf_int(alpha=0.05)[0][1],
                          statistics.conf_int(alpha=0.05)[1][0], statistics.conf_int(alpha=0.05)[1][1]]

        df_results = pd.DataFrame(results, index=fechas, columns=header)
        df_results.to_excel(xls_writer, sheet_name='ESTADISTICOS', merge_cells=False)
        xls_writer.close()
        print('------------------------------')
        print(df_results.to_string())

def generar_rasters_isotermas(input_dem, output_path, variables_ts,):
    for var in variables_ts:
        output_dir = os.path.join(output_path, 'Interpolacion', var)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        archivo_datos = os.path.join(output_path, f'COEF_ISOTERMAS_{var}.xlsx')
        COE = pd.ExcelFile(archivo_datos).parse('ESTADISTICOS', index_col=0)
        dem, No_Data_Value, _, _, _, _, _, _ = LoadRaster(input_dem)
        dem = ma.masked_equal(dem, No_Data_Value)
        coefficient_A, coefficient_B = COE['COE_A'], COE['COE_B']

        for i in coefficient_A.index:
            name = i
            outRuta = os.path.join(output_dir, f'{name}.tif')
            iso_t = coefficient_A[i] * dem + coefficient_B[i]
            iso_t = np.where(np.isfinite(iso_t), iso_t, No_Data_Value)
            iso_t = np.clip(iso_t, -No_Data_Value, No_Data_Value)  # Limitar los valores extremos para evitar inf de manera más conservadora
            iso_t = np.where(np.isfinite(iso_t), iso_t, No_Data_Value)
            iso_t = ma.masked_equal(np.round(iso_t, 3), No_Data_Value)
            array2raster(newRasterfn=outRuta, array=iso_t, input_dem=input_dem)
        print('Finalizado')



path = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\02_Climate_Caracterization/'
input_catalogo = path +'01_Completitud_Consistencia/03_Catalog_Check.xlsx'
input_mensuales = path + '02_Monthly/04_Seasonality/01_monthly_multiannuals.xlsx'
input_datos_interpolacion = path +'02_Monthly/08_Interpolacion/01_Datos_Interpolacion_M.xlsx'
input_dem = r'C:\Users\miguel.canon\Box\00-N4W_Facility_Projects_(Technical)\SE18-Yaque_del_Norte\03-Working_Folder\SA\04_Data_Model\01-DEM\Dem_Process/Burn_Dem_100.tif'
output_path = path +'02_Monthly/08_Interpolacion/'

# Definir las variables de entrada
variables_pt = ['PT_4']
variables_ts = ['TS_2', 'TS_3']
variables = variables_pt + variables_ts

procesar_datos_interpolacion(input_catalogo, input_mensuales, output_path, variables, variables_pt, variables_ts)
interpolar_idw(input_datos_interpolacion, output_path, variables_pt, input_dem)
calcular_coef_isotermas(input_datos_interpolacion, output_path, variables_ts)
generar_rasters_isotermas(input_dem, output_path, variables_ts)