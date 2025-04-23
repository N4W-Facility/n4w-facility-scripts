import pandas as pd
import xarray as xr
import s3fs
import asyncio
from pathlib import Path
from collections import defaultdict
import numpy as np
from asyncio import as_completed

# ========================
# CONFIGURACIÓN
# ========================
s3 = s3fs.S3FileSystem(anon=True)

# Obtiene la ruta absoluta del directorio donde se encuentra el script actual
script_dir = Path(__file__).resolve().parent

# Construye la ruta al archivo CSV asumiendo que está en el mismo directorio
CATALOG_URL= script_dir / "pangeo-cmip6-local.csv"#"https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv"

# path = Path(r'C:\Users\miguel.canon\Documents\Datos_Cambio_Climatico')
path = Path(r'C:\Users\Miguel Angel Cañon\Documents\Cambio_Climatico')
archivo_excel = path / "Parametros_CC.xlsx"
directorio_salida = path / "01_Series_GCM_raw"

batch_size = 5   # Cantidad de tareas a guardar por batch

# ========================
# FUNCIONES
# ========================

async def obtener_datos_multi_puntos(catalogo, modelo: str, experimento: str, variable: str, df_puntos):
    try:
        print(f"\n🚀 Iniciando {modelo}")
        print(f"🔍 Procesando: {modelo}-{experimento}-{variable} ({len(df_puntos)} puntos)")

        # Búsqueda en catálogo con prioridad gn luego gr
        try:
            query = (
                f"source_id == '{modelo}' and "
                f"experiment_id == '{experimento}' and "
                f"variable_id == '{variable}' and "
                "table_id == 'day' and grid_label == 'gn'"
            )
            entrada = catalogo.query(query).iloc[0]
        except:
            query = (
                f"source_id == '{modelo}' and "
                f"experiment_id == '{experimento}' and "
                f"variable_id == '{variable}' and "
                "table_id == 'day' and grid_label == 'gr'"
            )
            entrada = catalogo.query(query).iloc[0]

        url = f"s3://cmip6-pds/{entrada.zstore}"
        print(f"🌐 Accediendo a: {url}")

        store = s3.get_mapper(entrada.zstore)
        ds = xr.open_zarr(store, consolidated=True)

        # Tratamiento de coordenadas
        lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())

        lats = df_puntos['Lat'].values
        lons = df_puntos['Lon'].values
        names = df_puntos['Name'].values

        lons_ajustados = np.where((lon_min >= 0 and lon_max > 180), lons % 360, ((lons + 180) % 360) - 180)

        # Calculo de resolución
        try:
            res_lat = abs(float(ds.lat[1] - ds.lat[0]))
            res_lon = abs(float(ds.lon[1] - ds.lon[0]))
        except Exception:
            res_lat = abs(float(ds.lat.max() - ds.lat.min()) / ds.lat.size)
            res_lon = abs(float(ds.lon.max() - ds.lon.min()) / ds.lon.size)

        tolerance = max(res_lat, res_lon) * 1.5

        # Selección multipuntos
        ds_sel = ds.sel(
            lat=xr.DataArray(lats, dims="points"),
            lon=xr.DataArray(lons_ajustados, dims="points"),
            method="nearest",
            tolerance=tolerance
        )

        if ds_sel.dims.get('points', 0) == 0:
            print(f"⚠️ No se encontraron puntos dentro del dominio en {modelo}-{experimento}-{variable}")
            return None

        # Filtrado temporal
        time_slice = slice('1980-01-01', None) if experimento == 'historical' else slice('2015-01-01', '2100-12-31')
        ds_sel = ds_sel.sel(time=time_slice)

        # Conversión de unidades
        if variable == 'pr':
            ds_sel[variable] *= 86400  # kg/m²/s → mm/día
        elif variable in ['tasmax', 'tasmin', 'tas']:
            ds_sel[variable] -= 273.15  # K → °C

        # Extracción dataframe multipuntos
        df = await extraer_dataframe_multi(ds_sel, variable, names)

        print(f"✅ Datos obtenidos: {modelo}-{experimento}-{variable} ({len(df.columns)} puntos)")

        return {
            'modelo': modelo,
            'experimento': experimento,
            'variable': variable,
            'data': df
        }

    except Exception as e:
        print(f"❌ Error en {modelo}-{experimento}-{variable}: {str(e)[:200]}")
        return None


async def extraer_dataframe_multi(ds_sel, variable, nombres_puntos):
    """Manejo de calendarios raros + multipuntos robusto"""

    df = ds_sel[variable].to_dataframe().reset_index()

    try:
        df['time'] = np.repeat(ds_sel['time'].values, ds_sel.dims['points'])
    except Exception:
        calendar = ds_sel.time.dt.calendar
        print(f"📅 Calendario especial detectado: {calendar}")

        years = np.repeat(ds_sel.time.dt.year.values, ds_sel.dims['points'])
        months = np.repeat(ds_sel.time.dt.month.values, ds_sel.dims['points'])
        days = np.repeat(
            np.minimum(ds_sel.time.dt.day.values, 30) if calendar == '360_day' else ds_sel.time.dt.day.values,
            ds_sel.dims['points']
        )

        df['time'] = pd.to_datetime(
            [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)],
            errors='coerce'
        )

    df_pivot = df.pivot_table(index='time', columns='points', values=variable)

    if len(df_pivot.columns) != len(nombres_puntos):
        print(f"⚠️ Puntos encontrados: {len(df_pivot.columns)} vs Esperados: {len(nombres_puntos)}")

    df_pivot.columns = nombres_puntos[:len(df_pivot.columns)]

    return df_pivot.dropna()  # Ahora dropna en filas con NaT


async def guardar_batch(resultados):
    for resultado in resultados:
        modelo = resultado['modelo']
        experimento = resultado['experimento']
        variable = resultado['variable']
        df = resultado['data']

        dir_objetivo = directorio_salida / modelo / variable
        dir_objetivo.mkdir(parents=True, exist_ok=True)
        ruta = dir_objetivo / f"Series_{experimento}_{variable}.csv"

        df.sort_index().to_csv(ruta)
        print(f"💾 Guardado: {ruta}")

# ========================
# MAIN
# ========================


async def main():
    print("⏳ Iniciando proceso CMIP6...")

    df_modelos = pd.read_excel(archivo_excel, sheet_name="Consulta")
    df_puntos = pd.read_excel(archivo_excel, sheet_name="Catalogo")
    catalogo = pd.read_csv(CATALOG_URL)

    experimentos = [c for c in df_modelos.columns if c.startswith('ssp') or c == 'historical']
    variables_puntos = [c for c in df_puntos.columns if c not in ['Id', 'Name', 'Lat', 'Lon']]
    variables_modelos = [c for c in df_modelos.columns if c not in ['Modelo'] + experimentos]
    variables_comunes = list(set(variables_modelos) & set(variables_puntos))

    df_modelos_filtrados = df_modelos[(df_modelos[experimentos].sum(axis=1) > 0) |
                                      (df_modelos[variables_comunes].sum(axis=1) > 0)]

    tareas = []
    for _, fila_modelo in df_modelos_filtrados.iterrows():
        modelo = fila_modelo['Modelo']
        experimentos_activos = [e for e in experimentos if fila_modelo[e] == 1]
        vars_modelo = [v for v in variables_comunes if fila_modelo[v] == 1]

        for exp in experimentos_activos:
            for var in vars_modelo:
                df_puntos_filtrado = df_puntos[df_puntos[var] == 1]
                if df_puntos_filtrado.empty:
                    print(f"⚠️ No hay puntos activos para {modelo}-{exp}-{var}")
                    continue
                tareas.append(obtener_datos_multi_puntos(catalogo, modelo, exp, var, df_puntos_filtrado))

    print(f"🚀 Total de tareas a ejecutar: {len(tareas)}")


    exitosos = 0
    batch_resultados = []
    for i in range(0, len(tareas), batch_size):
        tareas_p = tareas[i:i + batch_size]
        resultados_batch = await asyncio.gather(*tareas_p)  # Lista de resultados del batch
        if resultados_batch:
            # Añade cada resultado individualmente (no la lista completa)
            batch_resultados.extend(resultados_batch)
            exitosos += len(resultados_batch)

        if len(batch_resultados) >= batch_size:
            await guardar_batch(batch_resultados)
            batch_resultados = []

    if batch_resultados:
        await guardar_batch(batch_resultados)

    print(f"\n🎉 Proceso terminado: {exitosos }/{len(tareas)} tareas exitosas")
    print(f"📁 Datos guardados en: {directorio_salida}")

if __name__ == "__main__":
    asyncio.run(main())
