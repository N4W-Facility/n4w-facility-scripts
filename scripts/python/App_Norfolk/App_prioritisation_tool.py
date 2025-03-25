import os
from shapely.geometry import Point
import rasterio
from shapely.geometry import box
from rasterstats import zonal_stats
import tkinter.font as tkFont
import warnings
from reportlab.pdfgen import canvas
import rasterio
from rasterio.mask import mask
import pandas as pd
from scipy import stats  # Para calcular la moda
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle, Paragraph
import matplotlib.pyplot as plt

import contextily as ctx
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import geopandas as gpd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from shapely.validation import make_valid
import queue
from shapely.ops import unary_union
from reportlab.lib.colors import red, yellow, green
warnings.filterwarnings("ignore")
import threading
import os
import sys
import sys, os
# Guardar stderr original
stderr_original = sys.stderr
sys.stderr = open(os.devnull, 'w')





# Configurar GDAL_DATA y PROJ_LIB para el ejecutable
#if getattr(sys, 'frozen', False):
#    # Si es un ejecutable, usa la ruta empaquetada
#    base_dir = sys._MEIPASS
#    gdal_data = os.path.join(base_dir, "gdal_data")
#    proj_data = os.path.join(base_dir, "proj_data")
#else:
#    # Si no, usa la ruta normal del entorno
#    base_dir = os.path.dirname(os.path.abspath(__file__))
#    gdal_data = os.path.join(os.environ['CONDA_PREFIX'], "lib", "site-packages", "rasterio", "gdal_data")
#    proj_data = os.path.join(os.environ['CONDA_PREFIX'], "lib", "site-packages", "rasterio", "proj_data")

# Definir variables de entorno críticas
#os.environ['GDAL_DATA'] = gdal_data
#os.environ['PROJ_LIB'] = proj_data
# Crear la cola global para mensajes
# Crear la cola global para mensajes


# Configurar GDAL_DATA y PROJ_LIB
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
    gdal_data = os.path.join(base_dir, "gdal_data")
    proj_data = os.path.join(base_dir, "proj_data")
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gdal_data = os.path.join(os.environ['CONDA_PREFIX'], "Lib", "site-packages", "rasterio", "gdal_data")
    proj_data = os.path.join(os.environ['CONDA_PREFIX'], "Lib", "site-packages", "rasterio", "proj_data")

os.environ['GDAL_DATA'] = gdal_data
os.environ['PROJ_LIB'] = proj_data

# Verificar existencia de archivos críticos (debug)
if getattr(sys, 'frozen', False):
    assert os.path.exists(os.path.join(gdal_data, "gdalvrt.xsd")), "¡gdalvrt.xsd no encontrado!"
    
import logging
gdal_data = os.environ.get('GDAL_DATA', '')

# Configurar logging
#logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def verify_gdal_data():
    """Verifica que GDAL_DATA esté configurado y que gdalvrt.xsd exista."""
    gdal_data = os.environ.get('GDAL_DATA', '')
    logging.debug(f"GDAL_DATA configured in: {gdal_data}")
    
    if not gdal_data:
        logging.warning("GDAL_DATA is not defined.")
        return
    
    gdalvrt_path = os.path.join(gdal_data, "gdalvrt.xsd")
    logging.debug(f"Searching  gdalvrt.xsd in: {gdalvrt_path}")
    
    if os.path.exists(gdalvrt_path):
        logging.info("✅ gdalvrt.xsd found successfully.")
    else:
        logging.error("❌ gdalvrt.xsd NOT found. Directory contains:")
        for f in os.listdir(gdal_data):
            logging.error(f" - {f}")

# Llamar a la función después de configurar GDAL_DATA
verify_gdal_data()
message_queue = queue.Queue()

import os
import sys
import zipfile
import tempfile
import shutil

def setup_data_directory():
    """Configura el directorio de datos, priorizando la carpeta Data sobre el ZIP."""
    global data_dir

    if getattr(sys, 'frozen', False):
        # Ruta base del ejecutable (PyInstaller)
        base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
        
        # Opción 1: Si existe la carpeta Data incluida en el .exe
        data_folder_path = os.path.join(base_dir, "Data")
        if os.path.exists(data_folder_path):
            data_dir = data_folder_path
            return  # Usar la carpeta Data directamente

        # Opción 2: Si no existe Data, buscar Data.zip y extraer
        data_zip_path = os.path.join(base_dir, "Data.zip")
        if os.path.exists(data_zip_path):
            temp_data_dir = os.path.join(tempfile.gettempdir(), "App_Data_Temp")
            
            # Extraer solo si no existe la carpeta temporal
            if not os.path.exists(temp_data_dir):
                # Limpiar y crear directorio
                if os.path.exists(temp_data_dir):
                    shutil.rmtree(temp_data_dir, ignore_errors=True)
                os.makedirs(temp_data_dir, exist_ok=True)
                
                # Extraer el ZIP
                with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_data_dir)
            
            data_dir = temp_data_dir
        else:
            raise FileNotFoundError("No se encontró ni la carpeta Data ni el archivo Data.zip.")
    else:
        # Modo desarrollo: usa la carpeta Data local
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

# Ejecutar al inicio del script
setup_data_directory()

class ConsoleOutput(scrolledtext.ScrolledText):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(state='disabled')

    def write(self, message):
        self.config(state='normal')
        # Asegúrate de que el mensaje termine con un salto de línea
        if not message.endswith("\n"):
            message += "\n"
        self.insert(tk.END, message)
        self.config(state='disabled')
        self.see(tk.END)  # Auto-scroll to the end

    def flush(self):
        pass

    def process_queue(self):
        try:
            while True:
                message = message_queue.get_nowait()
                self.write(message)
        except queue.Empty:
            pass
        # Programar la próxima ejecución de process_queue
        self.after(100, self.process_queue)
def calcular_area_y_subpoligonos(polygon_path):
    """
    Lee un shapefile de polígonos, calcula el área total en hectáreas y cuenta los subpolígonos.

    Parámetros:
    - polygon_path: Ruta del archivo shapefile (.shp).

    Retorna:
    - Lista con el área total en hectáreas y el número de subpolígonos.
    """
    try:
        # Leer el shapefile
        gdf = gpd.read_file(polygon_path)

        if gdf.crs.to_string() != 'EPSG:27700':
            gdf = gdf.to_crs('EPSG:27700')

        # Calcular el área de cada subpolígono en metros cuadrados
        gdf['area_m2'] = gdf.geometry.area

        # Sumar las áreas para obtener el área total en metros cuadrados
        area_total_m2 = gdf['area_m2'].sum()

        # Convertir el área total a hectáreas (1 hectárea = 10,000 m²)
        area_total_ha = area_total_m2 / 10000

        # Contar el número de subpolígonos
        num_subpoligonos = len(gdf)

        # Retornar una lista con el área total en hectáreas y el número de subpolígonos
        return [np.round(area_total_ha,2), num_subpoligonos]

    except Exception as e:
        print(f"Error processing shapefile: {e}")
        return None
def generar_imagen_png(shapefile_path, output_png_path,nombre_proyecto):
    """
    Genera una imagen PNG de un polígono shapefile con bordes rojos y un mapa base.

    Parámetros:
    - shapefile_path: Ruta del archivo shapefile de entrada.
    - output_png_path: Ruta donde se guardará la imagen PNG generada.
    """
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    try:
        # Leer el shapefile
        gdf = gpd.read_file(shapefile_path)
        # if gdf.crs.to_string() != 'EPSG:27700':
        #     gdf = gdf.to_crs('EPSG:27700')
        if gdf.crs.to_string() != 'EPSG:3857':
            gdf = gdf.to_crs('EPSG:3857')

        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Añadir el polígono al gráfico
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2,)

        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png")
        except:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)



        # Ajustar los límites al polígono de referencia con margen
        xmin, ymin, xmax, ymax = gdf.total_bounds
        margen_x = (xmax - xmin) * 0.2
        margen_y = (ymax - ymin) * 0.2
        ax.set_xlim(xmin - margen_x, xmax + margen_x)
        ax.set_ylim(ymin - margen_y, ymax + margen_y)


        north_arrow_path = obtener_ruta_data("Simbolo.png") #'Data/Simbolo.png' 
        # Agregar el símbolo del norte como imagen
        try:
            north_arrow_img = plt.imread(north_arrow_path)
            imagebox = OffsetImage(north_arrow_img, zoom=0.1)
            ab = AnnotationBbox(imagebox, (xmin + margen_x * 0.2, ymax - margen_y * 0.2), frameon=False,
                                box_alignment=(1.1, 0.2))
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error adding the north symbol: {e}")

        # Agregar la leyenda personalizada
        leyenda = mpatches.Patch(color='black', label=nombre_proyecto)
        ax.legend(handles=[leyenda], loc='lower right', title="Legend")
        ax.tick_params(axis='x', which='major', labelsize=10, labelcolor='gray', rotation=0)
        ax.tick_params(axis='y', which='major', labelsize=10, labelcolor='gray', rotation=0)

        ax.grid()

        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
        plt.close()

        # print(f"Imagen PNG generada exitosamente en: {output_png_path}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
def generar_imagen_con_capas(shapefile_path, shapefile_paths, output_png_path, colores, transparencias, buffer_km=0.8):
    """
    Genera una imagen PNG de múltiples capas shapefile con diferentes colores, transparencias, un mapa base y una leyenda.
    Las áreas fuera del polígono de referencia se opacan. Se aplica un buffer cuadrado para limitar el área de visualización.

    Parámetros:
    - shapefile_path: Ruta del shapefile de referencia.
    - shapefile_paths: Lista de rutas de los archivos shapefile de entrada.
    - output_png_path: Ruta donde se guardará la imagen PNG generada.
    - colores: Lista de colores para cada capa.
    - transparencias: Lista de valores de transparencia (entre 0 y 1) para cada capa.
    - buffer_km: Tamaño del buffer en kilómetros (por defecto 5 km).
    """
    try:
        # Leer el shapefile de referencia
        gdfref = gpd.read_file(shapefile_path)

        # Limpiar geometrías inválidas en el shapefile de referencia
        gdfref['geometry'] = gdfref['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

        # Asegurar que el CRS esté en EPSG:3857 (Web Mercator)
        if gdfref.crs.to_string() != 'EPSG:3857':
            gdfref = gdfref.to_crs('EPSG:3857')

        if len(shapefile_paths) != len(colores) or len(shapefile_paths) != len(transparencias):
            raise ValueError("El número de capas debe coincidir con el número de colores y transparencias.")

        # Crear un buffer cuadrado alrededor del polígono de referencia
        xmin, ymin, xmax, ymax = gdfref.total_bounds
        buffer_m = buffer_km * 1000  # Convertir kilómetros a metros
        buffer_box = box(xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)
        gdf_buffer = gpd.GeoDataFrame(geometry=[buffer_box], crs=gdfref.crs)

        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(15, 10))

        leyenda = []

        # Graficar todas las capas completas dentro del buffer (incluyendo áreas fuera del polígono de referencia)
        for idx, files in enumerate(shapefile_paths):
            # Leer cada shapefile
            gdf = gpd.read_file(files)

            # Convertir al CRS de referencia si no está ya
            if gdf.crs.to_string() != gdfref.crs.to_string():
                gdf = gdf.to_crs(gdfref.crs)

            # Recortar la capa al buffer cuadrado
            gdf_clipped = gpd.clip(gdf, gdf_buffer)

            # Graficar la capa recortada con transparencia reducida
            gdf_clipped.plot(
                ax=ax,
                edgecolor=colores[idx],
                facecolor=colores[idx],
                alpha=transparencias[idx] * 0.3,  # Opacidad reducida para áreas fuera del polígono
                linewidth=1
            )

        # Graficar el polígono de referencia con un relleno semitransparente para opacar el exterior
        gdfref.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2)

        # Crear una máscara para opacar el exterior del polígono de referencia
        from matplotlib.patches import Polygon
        import numpy as np

        # Crear un polígono que cubra toda el área del buffer
        exterior_polygon = np.array([
            [xmin - buffer_m, ymin - buffer_m],
            [xmax + buffer_m, ymin - buffer_m],
            [xmax + buffer_m, ymax + buffer_m],
            [xmin - buffer_m, ymax + buffer_m]
        ])

        # Crear un parche semitransparente para el exterior
        exterior_patch = Polygon(exterior_polygon, closed=True, facecolor='white', alpha=0.5, edgecolor='none')
        ax.add_patch(exterior_patch)

        # Graficar las capas dentro del polígono de referencia con colores y transparencias completas
        combined_layer = None
        for idx, files in enumerate(shapefile_paths):
            # Leer cada shapefile
            gdf = gpd.read_file(files)

            # Convertir al CRS de referencia si no está ya
            if gdf.crs.to_string() != gdfref.crs.to_string():
                gdf = gdf.to_crs(gdfref.crs)

            # Recortar la capa al buffer cuadrado
            gdf_clipped = gpd.clip(gdf, gdf_buffer)

            # Realizar la intersección con el polígono de referencia
            intersected = gpd.overlay(gdf_clipped, gdfref, how='intersection')

            # Si ya hay una capa combinada, realizar la diferencia para evitar superposiciones
            if combined_layer is not None:
                intersected = gpd.overlay(intersected, combined_layer, how='difference')

            # Actualizar la capa combinada
            if combined_layer is None:
                combined_layer = intersected.copy()
            else:
                combined_layer = pd.concat([combined_layer, intersected], ignore_index=True)

            # Graficar la capa intersectada con el color y la transparencia específicos
            intersected.plot(
                ax=ax,
                edgecolor=colores[idx],
                facecolor=colores[idx],
                alpha=transparencias[idx],  # Transparencia completa para áreas dentro del polígono
                linewidth=1
            )

            # Agregar a la leyenda
            # try:
            if 'Name' in intersected.columns and not intersected.empty:
                leyenda.append(mpatches.Patch(color=colores[idx], label=intersected['Name'].iloc[0]))
            # except:
            #     if 'Name' in intersected.columns:
            #         leyenda.append(mpatches.Patch(color=colores[idx], label=''))

        # Añadir un mapa base
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png")
        except:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

        # Ajustar los límites al buffer cuadrado
        ax.set_xlim(xmin - buffer_m, xmax + buffer_m)
        ax.set_ylim(ymin - buffer_m, ymax + buffer_m)

        # Eliminar los ejes para una visualización más limpia
        ax.tick_params(axis='x', which='major', labelsize=10, labelcolor='gray', rotation=0)
        ax.tick_params(axis='y', which='major', labelsize=10, labelcolor='gray', rotation=0)
        ax.grid()

        # Agregar el símbolo del norte como imagen (ajustado a la esquina superior izquierda)
        north_arrow_path = obtener_ruta_data("Simbolo.png")#'Data/Simbolo.png'
        try:
            north_arrow_img = plt.imread(north_arrow_path)
            imagebox = OffsetImage(north_arrow_img, zoom=0.1)

            # Posición del símbolo del norte (esquina superior izquierda)
            north_x = xmin - buffer_m + (buffer_m * 0.1)  # 10% del buffer desde el borde izquierdo
            north_y = ymax + buffer_m - (buffer_m * 0.1)  # 10% del buffer desde el borde superior

            ab = AnnotationBbox(imagebox, (north_x, north_y), frameon=False, box_alignment=(0, 1))
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error adding the north symbol: {e}")

        # Agregar la leyenda con el título "Legend"
        ax.legend(handles=leyenda, loc='lower right', title="Legend")

        # Guardar la imagen como PNG
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
        plt.close()

        # print(f"Imagen PNG generada exitosamente en: {output_png_path}")

    except Exception as e:
        print(f"Error: {e}")
def generar_imagen_con_capas_transparecia(shapefile_path, shapefile_paths, output_png_path, colores, transparencias, buffer_km=0.8):
    try:
        # Leer el shapefile de referencia
        gdfref = gpd.read_file(shapefile_path)

        # Limpiar geometrías inválidas en el shapefile de referencia
        gdfref['geometry'] = gdfref['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

        # Asegurar que el CRS esté en EPSG:3857 (Web Mercator)
        if gdfref.crs.to_string() != 'EPSG:3857':
            gdfref = gdfref.to_crs('EPSG:3857')

        if len(shapefile_paths) != len(colores) or len(shapefile_paths) != len(transparencias):
            raise ValueError("The number of layers must match the number of colours and transparencies..")

        # Crear un buffer cuadrado alrededor del polígono de referencia
        xmin, ymin, xmax, ymax = gdfref.total_bounds
        buffer_m = buffer_km * 1000  # Convertir kilómetros a metros
        buffer_box = box(xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)
        gdf_buffer = gpd.GeoDataFrame(geometry=[buffer_box], crs=gdfref.crs)

        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(15, 10))

        leyenda = []

        # Graficar todas las capas completas dentro del buffer (incluyendo áreas fuera del polígono de referencia)
        for idx, files in enumerate(shapefile_paths):
            # Leer cada shapefile
            gdf = gpd.read_file(files)

            # Limpiar geometrías inválidas en la capa actual
            gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

            # Convertir al CRS de referencia si no está ya
            if gdf.crs.to_string() != gdfref.crs.to_string():
                gdf = gdf.to_crs(gdfref.crs)

            # Recortar la capa al buffer cuadrado
            gdf_clipped = gpd.clip(gdf, gdf_buffer)

            # Graficar la capa recortada con transparencia reducida
            if idx == 1:
                # Primera capa: solo líneas (sin relleno)
                gdf_clipped.plot(
                    ax=ax,
                    edgecolor=colores[idx],  # Color del borde
                    facecolor='none',  # Sin relleno
                    alpha=transparencias[idx],  # Transparencia para las líneas
                    linewidth=1.5  # Grosor de las líneas
                )

            elif idx == 2:
                # Segunda capa: relleno amarillo intenso
                gdf_clipped.plot(
                    ax=ax,
                    edgecolor='none',  # Sin borde
                    facecolor=colores[idx],  # Relleno amarillo intenso
                    alpha=transparencias[idx],  # Transparencia para el relleno
                    linewidth=0  # Sin líneas de borde
                )
            elif idx == 0:
                # Tercera capa: relleno verde intenso
                gdf_clipped.plot(
                    ax=ax,
                    edgecolor='none',  # Sin borde
                    facecolor=colores[idx],  # Relleno verde intenso
                    alpha=transparencias[idx],  # Transparencia para el relleno
                    linewidth=0  # Sin líneas de borde
                )

        # Graficar el polígono de referencia con un relleno semitransparente para opacar el exterior
        gdfref.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2.5)

        # Crear una máscara para opacar el exterior del polígono de referencia
        exterior_polygon = np.array([
            [xmin - buffer_m, ymin - buffer_m],
            [xmax + buffer_m, ymin - buffer_m],
            [xmax + buffer_m, ymax + buffer_m],
            [xmin - buffer_m, ymax + buffer_m]
        ])

        # Crear un parche semitransparente para el exterior
        exterior_patch = Polygon(exterior_polygon, closed=True, facecolor='white', alpha=0.5, edgecolor='none')
        ax.add_patch(exterior_patch)

        # Graficar las capas dentro del polígono de referencia con colores y transparencias completas
        combined_layer = None
        for idx, files in enumerate(shapefile_paths):
            # Leer cada shapefile
            gdf = gpd.read_file(files)

            # Limpiar geometrías inválidas en la capa actual
            gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

            # Convertir al CRS de referencia si no está ya
            if gdf.crs.to_string() != gdfref.crs.to_string():
                gdf = gdf.to_crs(gdfref.crs)

            # Recortar la capa al buffer cuadrado
            gdf_clipped = gpd.clip(gdf, gdf_buffer)

            # Realizar la intersección con el polígono de referencia
            intersected = gpd.overlay(gdf_clipped, gdfref, how='intersection')

            # Verificar si la intersección está vacía
            if intersected.empty:
                # print(f"Advertencia: No hay intersección para la capa {files}. Se agregará a la leyenda.")
                # Agregar a la leyenda incluso si no hay intersección
                if 'Name' in gdf.columns:
                    leyenda.append(
                        mpatches.Patch(color=colores[idx], label=gdf['Name'].iloc[0], alpha=transparencias[idx]))
                else:
                    leyenda.append(mpatches.Patch(color=colores[idx], label=f"Layer {idx + 1} (no NbS)",
                                                  alpha=transparencias[idx]))
                continue

            # Si ya hay una capa combinada, realizar la diferencia para evitar superposiciones
            if combined_layer is not None:
                intersected = gpd.overlay(intersected, combined_layer, how='difference')

            # Actualizar la capa combinada
            if combined_layer is None:
                combined_layer = intersected.copy()
            else:
                combined_layer = pd.concat([combined_layer, intersected], ignore_index=True)

            # Graficar la capa intersectada con el color y la transparencia específicos
            if idx == 1:
                # Primera capa: solo líneas (sin relleno)
                intersected.plot(
                    ax=ax,
                    edgecolor=colores[idx],  # Color del borde
                    facecolor='none',  # Sin relleno
                    alpha=transparencias[idx],  # Transparencia para las líneas
                    linewidth=1.5  # Grosor de las líneas
                )
            elif idx == 2:
                # Segunda capa: relleno amarillo intenso
                intersected.plot(
                    ax=ax,
                    edgecolor='none',  # Sin borde
                    facecolor=colores[idx],  # Relleno amarillo intenso
                    alpha=transparencias[idx],  # Transparencia para el relleno
                    linewidth=0  # Sin líneas de borde
                )
            elif idx == 0:
                # Tercera capa: relleno verde intenso
                intersected.plot(
                    ax=ax,
                    edgecolor='none',  # Sin borde
                    facecolor=colores[idx],  # Relleno verde intenso
                    alpha=transparencias[idx],  # Transparencia para el relleno
                    linewidth=0  # Sin líneas de borde
                )

            # Agregar a la leyenda usando el campo 'Name'
            if 'Name' in intersected.columns:
                leyenda.append(
                    mpatches.Patch(color=colores[idx], label=intersected['Name'].iloc[0], alpha=transparencias[idx]))

        # Añadir un mapa base
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png")
        except:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

        # Ajustar los límites al buffer cuadrado
        ax.set_xlim(xmin - buffer_m, xmax + buffer_m)
        ax.set_ylim(ymin - buffer_m, ymax + buffer_m)

        # Eliminar los ejes para una visualización más limpia
        ax.tick_params(axis='x', which='major', labelsize=10, labelcolor='gray', rotation=0)
        ax.tick_params(axis='y', which='major', labelsize=10, labelcolor='gray', rotation=0)
        # ax.grid()

        # Agregar el símbolo del norte como imagen (ajustado a la esquina superior izquierda)
        north_arrow_path = obtener_ruta_data("Simbolo.png")#'Data/Simbolo.png'
        try:
            north_arrow_img = plt.imread(north_arrow_path)
            imagebox = OffsetImage(north_arrow_img, zoom=0.1)

            # Posición del símbolo del norte (esquina superior izquierda)
            north_x = xmin - buffer_m + (buffer_m * 0.1)  # 10% del buffer desde el borde izquierdo
            north_y = ymax + buffer_m - (buffer_m * 0.1)  # 10% del buffer desde el borde superior

            ab = AnnotationBbox(imagebox, (north_x, north_y), frameon=False, box_alignment=(0, 1))
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error adding the north symbol: {e}")

        # Agregar la leyenda con el título "Legend"
        ax.legend(handles=leyenda, loc='lower right', title="Legend")

        # Guardar la imagen como PNG
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
        plt.close()

        # print(f"Imagen PNG generada exitosamente en: {output_png_path}")

    except Exception as e:
        print(f"Error: {e}")
def calcular_area_LULC(polygon_path, input_path, diccionario):
    """
    Calcula el área de las coberturas dentro de un polígono dado un raster o shapefile.

    Parámetros:
    - polygon_path: Ruta del shapefile que contiene el polígono de interés.
    - input_path: Ruta del archivo raster o shapefile de entrada.
    - output_csv_path: Ruta del archivo CSV de salida con los resultados.
    - lulc_diccionario: Diccionario que traduce las categorías del archivo raster.

    Salida:
    - Genera un archivo CSV con las áreas por categoría dentro del polígono.
    """
    try:
        # Leer el shapefile del polígono
        gdf_polygon = gpd.read_file(polygon_path)

        if gdf_polygon.crs.to_string() != 'EPSG:27700':
            gdf_polygon = gdf_polygon.to_crs('EPSG:27700')

        # Verificar el CRS del polígono
        if gdf_polygon.crs is None:
            raise ValueError("Warning: Polygon file has no CRS defined.")

        # Verificar el CRS del polígono
        if gdf_polygon.crs is None:
            raise ValueError("Warning: file has no CRS defined.")

        # Unir todos los polígonos en uno solo
        unified_polygon = unary_union(gdf_polygon.geometry)

        # Crear un GeoDataFrame con el polígono unificado
        gdf_polygon = gpd.GeoDataFrame(geometry=[unified_polygon], crs=gdf_polygon.crs)

        # Caso 2: Si el archivo de entrada es un raster
        if input_path.endswith(('.tif', '.asc')):
            with rasterio.open(input_path) as src:
                if src.crs.to_string() != gdf_polygon.crs.to_string():
                    raise ValueError("The CRS of the raster does not match that of the polygon..")

                # Realizar estadísticas zonales
                stats = zonal_stats(
                    gdf_polygon,
                    input_path,
                    categorical=True,
                    geojson_out=False,
                    nodata=src.nodata
                )

            # Convertir estadísticas a DataFrame
            result = pd.DataFrame(stats[0].items(), columns=['cobertura', 'area_pixels'])

            # Calcular el área en hectáreas (asume píxeles cuadrados)
            pixel_size = src.res[0] * src.res[1]
            result['Area (hectares)'] = (result['area_pixels'] * pixel_size) / 10000
            result.drop(columns=['area_pixels'], inplace=True)

            # Traducir categorías usando el diccionario
            result['Land Use Type'] = result['cobertura'].map(diccionario)
            result = result[['Land Use Type', 'Area (hectares)']]
            total_area = np.round(result['Area (hectares)'].sum(),2)

            result = pd.concat([                result,
                pd.DataFrame([["Total Area", total_area]], columns=result.columns)
            ], ignore_index=True)




        else:
            raise ValueError("Input file must be a shapefile or raster file.")

        # Formatear las áreas en hectáreas en el formato solicitado
        # result['descripcion'] = result.apply(lambda row: f"{row['area_ha']:.2f} hectares of {row['cobertura_nombre']}", axis=1)


    except Exception as e:
        print(f"Error: {e}")

    return result
def calcular_area_slope(polygon_path, raster_path):
    """
    Clasifica los valores de pendiente dentro de un polígono en categorías y calcula el área de cada categoría.

    Parámetros:
    - polygon_path: Ruta del shapefile que contiene el polígono de interés.
    - raster_path: Ruta del archivo raster de pendiente (slope).

    Retorna:
    - DataFrame con las categorías de pendiente y el área correspondiente en hectáreas.
    """
    try:
        # Leer el shapefile del polígono
        gdf_polygon = gpd.read_file(polygon_path)

        # Asegurar que el CRS esté en EPSG:27700 (o el CRS proyectado que uses)
        if gdf_polygon.crs.to_string() != 'EPSG:27700':
            gdf_polygon = gdf_polygon.to_crs('EPSG:27700')

        # Verificar el CRS del polígono
        if gdf_polygon.crs is None:
            raise ValueError("El archivo de polígono no tiene CRS definido.")

        # Leer el raster y verificar su CRS
        with rasterio.open(raster_path) as src:
            if src.crs.to_string() != gdf_polygon.crs.to_string():
                raise ValueError("El CRS del raster no coincide con el del polígono.")

            # Obtener la geometría del polígono en el formato que rasterio espera
            geometries = gdf_polygon.geometry.tolist()

            # Recortar el raster al polígono
            out_image, out_transform = mask(src, geometries, crop=True, all_touched=True)
            out_image = out_image[0]  # Extraer la primera banda

            # Filtrar los valores válidos (ignorar nodata)
            valid_values = out_image[out_image != src.nodata]

            # Obtener el tamaño de píxel en metros (asumiendo que el CRS está en metros)
            pixel_width, pixel_height = src.res
            pixel_area = pixel_width * pixel_height  # Área de un píxel en metros cuadrados

            # Clasificar los valores de pendiente
            flat_mask = (valid_values >= 0) & (valid_values < 1)
            moderate_mask = (valid_values >= 1) & (valid_values < 5)
            steep_mask = valid_values >= 5

            # Calcular el área para cada categoría
            flat_area = np.sum(flat_mask) * pixel_area / 10000  # Convertir a hectáreas
            moderate_area = np.sum(moderate_mask) * pixel_area / 10000  # Convertir a hectáreas
            steep_area = np.sum(steep_mask) * pixel_area / 10000  # Convertir a hectáreas

            # Crear el DataFrame con los resultados
            result_df = pd.DataFrame({
                "Slope": ["Flat (0-1%)", "Moderate slope (1-5%)", "Steep slope (5+%)"],
                "Area (hectares)": [flat_area, moderate_area, steep_area]
            })

            return result_df

    except Exception as e:
        print(f"Error: {e}")
        return None
def calcular_area_files_shp(polygon_path, shp_path, field, count_polygons=False):
    """
    Calcula el área del hábitat dentro de un polígono de referencia y/o cuenta los polígonos por clase.

    Parámetros:
    - polygon_path: Ruta del shapefile que contiene el polígono de referencia.
    - shp_path: Ruta del shapefile que contiene la capa de interés con el campo especificado.
    - field: Campo utilizado para agrupar los resultados.
    - count_polygons: Bandera que determina si se cuentan polígonos en lugar de calcular áreas.

    Salida:
    - Devuelve un DataFrame con el número de polígonos o el área por clase dentro del polígono.
    """
    try:
        # Leer el shapefile del polígono y la capa de interés
        gdf_polygon = gpd.read_file(polygon_path)

        if gdf_polygon.crs.to_string() != 'EPSG:27700':
            gdf_polygon = gdf_polygon.to_crs('EPSG:27700')

        gdf_shp = gpd.read_file(shp_path)

        # Verificar el CRS de ambos shapefiles
        if gdf_polygon.crs is None or gdf_shp.crs is None:
            raise ValueError("Ambos shapefiles deben tener un CRS definido.")

        if gdf_polygon.crs != gdf_shp.crs:
            gdf_shp = gdf_shp.to_crs(gdf_polygon.crs)

        # Realizar la intersección
        intersected = gpd.overlay(gdf_shp, gdf_polygon, how='intersection')

        # Verificar si el campo especificado existe
        if field not in gdf_shp.columns:
            raise ValueError("La capa de interés no contiene el campo " + field)

        # Si no hay intersección, devolver un DataFrame con área cero
        if intersected.empty:
            unique_values = gdf_shp[field].unique()  # Obtener los valores únicos del campo
            result = pd.DataFrame({
                field: unique_values,
                'Area (hectares)': [0] * len(unique_values)  # Área cero para cada valor único
            })
            total_area = 0
            result = pd.concat([
                result,
                pd.DataFrame([["Total Area", total_area]], columns=result.columns)
            ], ignore_index=True)
            return result

        if count_polygons:
            # Calcular el número de polígonos por clase
            intersected['polygon_count'] = 1
            result = intersected.groupby(field).agg({
                'polygon_count': 'count',
                'geometry': 'size'  # Para contabilizar los polígonos
            }).reset_index()
        else:
            # Calcular el área en metros cuadrados
            intersected['area_m2'] = intersected.geometry.area

            # Agrupar por clase y sumar las áreas
            result = intersected.groupby(field)['area_m2'].sum().reset_index()

            # Convertir el área a hectáreas
            result['Area (hectares)'] = np.round(result['area_m2'] / 10000, 2)

            result = result.drop(columns=['area_m2'])

            total_area = result['Area (hectares)'].sum()

            result = pd.concat([                result,
                pd.DataFrame([["Total Area", total_area]], columns=result.columns)
            ], ignore_index=True)

        return result

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()
def procesar_intersecciones_y_mascara(gdfref_path, shapefile_paths, raster_path, Sufix):
    """
        Procesa intersecciones entre múltiples capas shapefile y una capa raster, generando dos resultados:
        - Valor total de la máscara para cada intersección.
        - Valores por cada polígono de referencia para cada intersección.

        Además, asegura que no haya superposición entre las áreas procesadas de los shapefiles.

        Parámetros:
        - gdfref_path: Ruta del shapefile de referencia.
        - shapefile_paths: Lista de rutas de los shapefiles a procesar.
        - raster_path: Ruta del archivo raster (.tif).
        - output_path: Ruta donde se guardarán los resultados.

        Salidas:
        - CSV con el valor total de la máscara para cada intersección.
        - CSV con los valores por polígono de referencia y sus intersecciones.
        """
    import geopandas as gpd
    import rasterio
    import numpy as np
    import pandas as pd
    from rasterio.mask import mask
    import statistics as stat

    diccio_units = {'BNG': 'BNG Units',
                    'Nitrogen': 'kg/yr',
                    'Phosphorus': 'kg/yr',
                    'Recharge': 'mm/yr',
                    'Runoff': 'mm/yr'}
    # Leer el shapefile de referencia
    gdfref = gpd.read_file(gdfref_path)

    # Asegurar que el CRS esté en EPSG:27700 para trabajar con mapas base
    if gdfref.crs.to_string() != 'EPSG:27700':
        gdfref = gdfref.to_crs('EPSG:27700')# Inicializar variables de resultados
    total_mask_values = []
    per_polygon_values = []
    combined_layer = None  # Para evitar superposiciones

    for idx, shp_path in enumerate(shapefile_paths):
        # Leer cada shapefile
        gdf = gpd.read_file(shp_path)

        # Asegurar que el CRS coincida con el de referencia
        if gdf.crs.to_string() != gdfref.crs.to_string():
            gdf = gdf.to_crs(gdfref.crs)

        # Intersección con el polígono de referencia
        intersected = gpd.overlay(gdf, gdfref, how='intersection')

        # Eliminar áreas ya procesadas para evitar superposición
        if combined_layer is not None:
            intersected = gpd.overlay(intersected, combined_layer, how='difference')

        # Actualizar la capa combinada con las nuevas áreas procesadas
        if combined_layer is None:
            combined_layer = intersected.copy()
        else:
            combined_layer = pd.concat([combined_layer, intersected], ignore_index=True)

    # Leer el raster y crear la máscara
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs.to_string()

        # Verificar que el CRS del raster coincida con el CRS de referencia
        if raster_crs != gdfref.crs.to_string():
            raise ValueError("El CRS del raster no coincide con el CRS de los shapefiles.")

        # Generar la máscara para toda la capa intersectada
        geom_list = [geom for geom in intersected.geometry]

        geo_dict = {name: combined_layer.loc[combined_layer['Name'] == name, 'geometry'].values[0] for name in combined_layer['Name'].unique()}

        # Calcular el área de un píxel (en hectáreas)
        pixel_size_m = src.res[0] * src.res[1]  # Tamaño del píxel en metros cuadrados
        pixel_size_ha = pixel_size_m / 10000  # Tamaño del píxel en hectáreas


        for nbs in geo_dict:
            geom_list = geo_dict[nbs]
            out_image, out_transform = mask(src, [geom_list], crop=True)

            # Calcular el valor total de la máscara (ignorar nodata)
            nodata = src.nodata
            if np.isnan(nodata):
                valid_pixels = out_image[~np.isnan(out_image)]#* pixel_size_ha #step 1
            else:
                valid_pixels = out_image[out_image != nodata]#* pixel_size_ha#step 1

            number_of_pixels = len(valid_pixels)  # step 2
            total_area_ha = number_of_pixels * pixel_size_ha  # Step3
            adjust_raster = valid_pixels * pixel_size_ha

            Total_Uplift_2  = np.sum(adjust_raster)


            total_mask_values.append({
                    'NBS': gdf.Name if len(gdf.Name) ==1 else gdf.Name[0],
                    'Total_area_ha': np.round(total_area_ha, 2),
                    f'Total Benefit {diccio_units[Sufix]}': np.round(Total_Uplift_2 ,2)  if len(adjust_raster ) != 0 else 0,
                    # f'Total Benefit {diccio_units[Sufix]}/Ha': np.round(Uplift_ha , 2) if len(valid_pixels) != 0 else 0,
                    f'Mean Value {diccio_units[Sufix]}' :np.round(np.mean(adjust_raster ),2) if len(adjust_raster ) != 0 else 0,
                    f'Mode Value {diccio_units[Sufix]}':np.round(stat.mode(adjust_raster ),2)  if len(adjust_raster ) != 0 else 0,
                    f'Median Value {diccio_units[Sufix]}': np.round(np.median(adjust_raster ),2) if len(adjust_raster ) != 0 else 0,

                })

    # Guardar resultados como CSV
    total_mask_df = pd.DataFrame(total_mask_values)
    total_mask_df = total_mask_df[total_mask_df['Total_area_ha'] > 0]
    # total_mask_df.to_csv(f"{output_path}/{Sufix}_total_values.csv", index=False)


    # print("Resultados guardados en la carpeta de salida.")

    return  total_mask_df
def procesar_nbs_y_raster_individual(polygon_path, shapefile_paths, raster_path, Sufix, buffer_distance=5):
    """
    Procesa las intersecciones de múltiples shapefiles (polígonos o puntos) con un polígono de referencia,
    calcula las áreas correctas y extrae valores del raster o de los campos de puntos.

    Parámetros:
    - polygon_path: Ruta del shapefile que contiene el polígono de referencia.
    - shapefile_paths: Lista de rutas de los shapefiles a procesar (pueden ser polígonos o puntos).
    - raster_path: Ruta del archivo raster (.tif) para capas de polígonos.
    - Sufix: Sufijo para identificar las unidades de los resultados.
    - buffer_distance: Distancia del buffer en metros (por defecto 5 metros).

    Salidas:
    - DataFrame con las áreas y estadísticas del raster o los valores de los puntos.
    """
    import geopandas as gpd
    import rasterio
    import numpy as np
    import pandas as pd
    from rasterio.mask import mask
    from shapely.geometry import Polygon

    diccio_units = {
        'BNG': 'BNG Units',
        'Nitrogen': 'Nitrogen Export Reduction kg/yr',
        'Phosphorus': 'Phosphorus Export Reduction kg/yr',
        'Recharge': 'Infiltration uplift m3/yr',
        'Runoff': 'Runoff Reduction m3/yr'
    }

    # Leer el shapefile de referencia
    gdf_polygon = gpd.read_file(polygon_path)

    # Asegurar que el CRS esté en EPSG:27700
    if gdf_polygon.crs.to_string() != 'EPSG:27700':
        gdf_polygon = gdf_polygon.to_crs('EPSG:27700')

    # Crear un buffer de 5 metros alrededor del polígono
    gdf_polygon['geometry'] = gdf_polygon.geometry.buffer(buffer_distance)

    # Inicializar variables de resultados
    resultados = []

    # Inicializar acumuladores para las capas de puntos
    total_puntos = 0.0

    # Procesar cada shapefile individualmente
    for shp_path in shapefile_paths:
        # Leer el shapefile
        gdf_shp = gpd.read_file(shp_path)

        # Asegurar que el CRS coincida con el de referencia
        if gdf_shp.crs.to_string() != gdf_polygon.crs.to_string():
            gdf_shp = gdf_shp.to_crs(gdf_polygon.crs)

        # Verificar si es una capa de puntos o polígonos
        if gdf_shp.geometry.type.iloc[0] == 'Point':
            # Eliminar puntos duplicados (mismas coordenadas)
            gdf_shp = gdf_shp.drop_duplicates(subset=['geometry'])

            # Procesar capa de puntos
            #puntos_dentro = gdf_shp[gdf_shp.within(gdf_polygon.geometry.iloc[0])]
            
            # Combinar todos los polígonos en una sola geometría
            polygon_union = gdf_polygon.unary_union

            # Filtrar puntos que están dentro de cualquier polígono
            puntos_dentro = gdf_shp[gdf_shp.within(polygon_union)]

            if Sufix == 'Nitrogen':
                campo = 'N_remvl'  # Campo para Nitrógeno
            elif Sufix == 'Phosphorus':
                campo = 'P_remvl'  # Campo para Fósforo

            elif Sufix == 'Runoff':
                campo = 'Wtr_Cp_'
            elif Sufix == 'Recharge':
                campo = 'Recharg'
            else:
                raise ValueError(f"Sufijo no válido para capas de puntos: {Sufix}")

            if campo not in puntos_dentro.columns:
                raise ValueError(f"La capa de puntos no contiene el campo {campo}.")

            # Sumar los valores del campo correspondiente
            total_puntos += puntos_dentro[campo].sum()

        else:
            # Procesar capa de polígonos
            # Intersección con el polígono de referencia
            intersected = gpd.overlay(gdf_shp, gdf_polygon, how='intersection')

            # Si no hay intersección, continuar con la siguiente capa
            if intersected.empty:
                continue

            # Leer el raster
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs.to_string()

                # Verificar que el CRS del raster coincida con el CRS de referencia
                if raster_crs != gdf_polygon.crs.to_string():
                    raise ValueError("El CRS del raster no coincide con el CRS de los shapefiles.")

                # Calcular el área de un píxel (en hectáreas)
                pixel_size_m = src.res[0] * src.res[1]  # Tamaño del píxel en metros cuadrados
                pixel_size_ha = pixel_size_m / 10000  # Tamaño del píxel en hectáreas

                # Generar la máscara para la capa actual
                geom_list = [geom for geom in intersected.geometry]
                out_image, out_transform = mask(src, geom_list, crop=True)

                # Calcular el valor total de la máscara (ignorar nodata)
                nodata = src.nodata
                if np.isnan(nodata):
                    valid_pixels = out_image[~np.isnan(out_image)]
                else:
                    valid_pixels = out_image[out_image != nodata]

                # Calcular estadísticas
                adjust_raster = valid_pixels * pixel_size_ha
                Total_Uplift_2 = np.round(np.sum(adjust_raster),2)

                # Agregar resultados a la lista
                resultados.append({
                    'NBS': intersected['Name'].iloc[0] if 'Name' in intersected.columns else shp_path,
                    f'{diccio_units[Sufix]}': np.round(Total_Uplift_2, 2) if len(adjust_raster) != 0 else 0,
                })

    # Agregar el total de las capas de puntos a la NBS "Runoff Attenuation Features"
    if total_puntos > 0:
        resultados.append({
            'NBS': 'Runoff Attenuation Features',
            f'{diccio_units[Sufix]}': np.round(total_puntos, 2)
        })

    # Crear DataFrame con los resultados
    resultados_df = pd.DataFrame(resultados)
    return resultados_df
def contar_puntos_intersectados(polygon_path, shapefile_paths, buffer_distance=5):
    """
    Cuenta la cantidad de puntos dentro del polígono de referencia (con buffer aplicado),
    para cada shapefile de tipo punto.

    Parámetros:
    - polygon_path: Ruta del shapefile del polígono de referencia.
    - shapefile_paths: Lista de rutas de shapefiles a procesar.
    - buffer_distance: Distancia del buffer en metros (por defecto 5 metros).

    Retorna:
    - Diccionario con la ruta del shapefile como clave y la cantidad de puntos intersectados como valor.
    """
    conteo_intersecciones = {}

    # Leer polígono de referencia y asegurarse del CRS
    gdf_polygon = gpd.read_file(polygon_path)
    if gdf_polygon.crs.to_string() != 'EPSG:27700':
        gdf_polygon = gdf_polygon.to_crs('EPSG:27700')

    # Aplicar buffer
    gdf_polygon['geometry'] = gdf_polygon.geometry.buffer(buffer_distance)
    polygon_union = gdf_polygon.unary_union



    # Iterar sobre shapefiles
    for shp_path in shapefile_paths:
        gdf_shp = gpd.read_file(shp_path)

        if gdf_shp.empty:
            continue

        # Asegurar CRS coincidente
        if gdf_shp.crs.to_string() != gdf_polygon.crs.to_string():
            gdf_shp = gdf_shp.to_crs(gdf_polygon.crs)

        # Verificar si es capa de puntos
        if gdf_shp.geometry.type.iloc[0] == 'Point':
            gdf_shp = gdf_shp.drop_duplicates(subset=['geometry'])  # opcional
            puntos_dentro = gdf_shp[gdf_shp.within(polygon_union)]
            conteo_intersecciones[shp_path] = len(puntos_dentro)

    reglas_nombre = {
        'cRAF': 'Units of In-channel Runoff Attenuation Features',
        'sRAF': 'Units of In-field Runoff Attenuation Features'
    }

    # Crear lista con los textos combinados
    resultado = []

    for ruta, valor in conteo_intersecciones.items():
        for clave, nombre in reglas_nombre.items():
            if clave in ruta:
                resultado.append(f"{valor} {nombre}")
                break  # opcional, para evitar múltiples coincidencias

    return resultado
def procesar_nbs_y_raster_combinado(polygon_path, shapefile_paths, raster_path, Sufix):
    """
    Procesa las intersecciones de múltiples shapefiles con un polígono de referencia,
    calcula las áreas correctas y extrae valores del raster para cada intersección.

    Parámetros:
    - polygon_path: Ruta del shapefile que contiene el polígono de referencia.
    - shapefile_paths: Lista de rutas de los shapefiles a procesar.
                      - La primera capa se procesa como antes.
                      - La segunda capa ya está combinada y tiene un campo llamado 'NbS'.
    - raster_path: Ruta del archivo raster (.tif).
    - Sufix: Sufijo para identificar las unidades de los resultados.

    Salidas:
    - DataFrame con las áreas y estadísticas del raster para cada intersección.
    """
    import geopandas as gpd
    import rasterio
    import numpy as np
    import pandas as pd
    from rasterio.mask import mask
    import statistics as stat

    diccio_units = {'BNG': 'BNG Units',
                    'Nitrogen': 'kg/yr',
                    'Phosphorus': 'kg/yr',
                    'Recharge': 'mm/yr',
                    'Runoff': 'mm/yr'}

    # Leer el shapefile de referencia
    gdf_polygon = gpd.read_file(polygon_path)

    # Asegurar que el CRS esté en EPSG:27700
    if gdf_polygon.crs.to_string() != 'EPSG:27700':
        gdf_polygon = gdf_polygon.to_crs('EPSG:27700')

    # Inicializar variables de resultados
    resultados = []

    # Leer el raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs.to_string()

        # Verificar que el CRS del raster coincida con el CRS de referencia
        if raster_crs != gdf_polygon.crs.to_string():
            raise ValueError("El CRS del raster no coincide con el CRS de los shapefiles.")

        # Calcular el área de un píxel (en hectáreas)
        pixel_size_m = src.res[0] * src.res[1]  # Tamaño del píxel en metros cuadrados
        pixel_size_ha = pixel_size_m / 10000  # Tamaño del píxel en hectáreas

        # Procesar la primera capa (como antes)
        shp_path = shapefile_paths[0]  # Primera capa
        gdf_shp = gpd.read_file(shp_path)

        # Asegurar que el CRS coincida con el de referencia
        if gdf_shp.crs.to_string() != gdf_polygon.crs.to_string():
            gdf_shp = gdf_shp.to_crs(gdf_polygon.crs)

        # Intersección con el polígono de referencia
        intersected = gpd.overlay(gdf_shp, gdf_polygon, how='intersection')

        # Si no hay intersección, continuar con la siguiente capa
        if not intersected.empty:
            # Calcular el área de la intersección
            area_m2 = intersected.geometry.area.sum()
            area_ha = area_m2 / 10000

            # Generar la máscara para la capa actual
            geom_list = [geom for geom in intersected.geometry]
            out_image, out_transform = mask(src, geom_list, crop=True)

            # Calcular el valor total de la máscara (ignorar nodata)
            nodata = src.nodata
            if np.isnan(nodata):
                valid_pixels = out_image[~np.isnan(out_image)]
            else:
                valid_pixels = out_image[out_image != nodata]

            # Calcular estadísticas
            adjust_raster = valid_pixels * pixel_size_ha
            Total_Uplift_2 = np.sum(adjust_raster)

            # Agregar resultados a la lista
            resultados.append({
                'NBS': intersected['Name'].iloc[0] if 'Name' in intersected.columns else shp_path,
                'Total_area_ha': np.round(area_ha, 2),
                f'Total Benefit {diccio_units[Sufix]}': np.round(Total_Uplift_2, 2) if len(adjust_raster) != 0 else 0,
                f'Mean Value {diccio_units[Sufix]}': np.round(np.mean(adjust_raster), 2) if len(adjust_raster) != 0 else 0,
                f'Mode Value {diccio_units[Sufix]}': np.round(stat.mode(adjust_raster), 2) if len(adjust_raster) != 0 else 0,
                f'Median Value {diccio_units[Sufix]}': np.round(np.median(adjust_raster), 2) if len(adjust_raster) != 0 else 0,
            })

        # Procesar la segunda capa (combinada)
        shp_path = shapefile_paths[1]  # Segunda capa (combinada)
        gdf_shp = gpd.read_file(shp_path)

        # Asegurar que el CRS coincida con el de referencia
        if gdf_shp.crs.to_string() != gdf_polygon.crs.to_string():
            gdf_shp = gdf_shp.to_crs(gdf_polygon.crs)

        # Intersección con el polígono de referencia
        intersected = gpd.overlay(gdf_shp, gdf_polygon, how='intersection')

        # Si no hay intersección, continuar con la siguiente capa
        if not intersected.empty:
            # Obtener los valores únicos del campo 'NbS'
            unique_nbs = intersected['NbS'].unique()

            # Procesar cada valor único de 'NbS'
            for nbs in unique_nbs:
                # Filtrar la intersección por el valor actual de 'NbS'
                filtered = intersected[intersected['NbS'] == nbs]

                # Calcular el área de la intersección
                area_m2 = filtered.geometry.area.sum()
                area_ha = area_m2 / 10000

                # Generar la máscara para la capa actual
                geom_list = [geom for geom in filtered.geometry]
                out_image, out_transform = mask(src, geom_list, crop=True)

                # Calcular el valor total de la máscara (ignorar nodata)
                nodata = src.nodata
                if np.isnan(nodata):
                    valid_pixels = out_image[~np.isnan(out_image)]
                else:
                    valid_pixels = out_image[out_image != nodata]

                # Calcular estadísticas
                adjust_raster = valid_pixels * pixel_size_ha
                Total_Uplift_2 = np.sum(adjust_raster)

                # Agregar resultados a la lista
                resultados.append({
                    'NBS': nbs,
                    'Total_area_ha': np.round(area_ha, 2),
                    f'Total Benefit {diccio_units[Sufix]}': np.round(Total_Uplift_2, 2) if len(adjust_raster) != 0 else 0,
                    f'Mean Value {diccio_units[Sufix]}': np.round(np.mean(adjust_raster), 2) if len(adjust_raster) != 0 else 0,
                    f'Mode Value {diccio_units[Sufix]}': np.round(stat.mode(adjust_raster), 2) if len(adjust_raster) != 0 else 0,
                    f'Median Value {diccio_units[Sufix]}': np.round(np.median(adjust_raster), 2) if len(adjust_raster) != 0 else 0,
                })

    # Crear DataFrame con los resultados
    resultados_df = pd.DataFrame(resultados)
    resultados_df = resultados_df[resultados_df['Total_area_ha'] > 0]

    return resultados_df
def extract_area_nbs_I(Polygon):
    # Diccionario de mapeo
    nbs_info = {
        "Riparian Zone Restoration": {
            "Description": "These include measures to intercept or attenuate minor watercourse corridors above the floodplain, particularly where the soils are permeable and there is underlying aquifer.",
            "Local Context": "Features are typically situated over arable hillslope streams. Riparian planting may slow flows to permit more gradual recharge into both superficial and bedrock aquifers together with trapping and filtering potential pollutants."
        },
        "Runoff Attenuation Features": {
            "Description": "The interventions modify areas by providing storage and attenuation on hillslope flow pathways. The activities include restoration and recovery. They are most beneficial over permeable soils and underlying aquifers.",
            "Local Context": "The density of features is highest on till covered areas in the south of the area of interest.  On the chalk, they concentrate along topographical depressions draining into rivers.  The most effective locations to target are likely to be those on the edge of the chalk, which are likely to be active more often capturing the run-off from till.  Those that start on the chalk will likely have less run-off entering them on average."
        },
        "Floodplain Zone": {
            "Description": "The interventions modify areas within the wider floodplain by providing storage and attenuation on hillslope flow pathways. The activities include floodplain restoration or the creation of storage areas, including ponds and scrapes. Elevated groundwater levels within the lower catchment may seasonally reduce recharge capabilitie.",
            "Local Context": "In the upper catchment this shows the area where the chalk stream valleys can be restored.  In the lower catchment it highlights the fen areas where the drainage could be modified to store more water and recreate/improve the fenland habitats."
        }
    }

    lista_nbs = ['RZ_Opmap_PRIO','Wen_CS_Update','FZ_Final_merge_FROP']
    nbs_result = pd.DataFrame()
    for nbs in lista_nbs:

        path_nbs = obtener_ruta_data("NbS", nbs+'.shp')#'Data/NbS/'+nbs+'.shp'

        temp = calcular_area_files_shp(Polygon, path_nbs, 'Name',  count_polygons=False)
        temp = temp[temp["Name"] != "Total Area"]

        nbs_result = pd.concat([nbs_result,temp])

    nbs_result.columns = ['NbS','Area (Hectares)']

    # Agregar las columnas Description y Local Context
    nbs_result["Description"] =   nbs_result ["NbS"].map(lambda x: nbs_info[x]["Description"])
    nbs_result["Local Context"] =   nbs_result ["NbS"].map(lambda x: nbs_info[x]["Local Context"])

    nbs_result['NbS'] = nbs_result['NbS'].replace("Floodplain Zone", "Floodplain reconnection and restoration")

    return nbs_result
def extract_area_nbs_LUC(Polygon):
    # Diccionario de mapeo
    nbs_info = {
        "Soil Management": {
            "Description": "These represent in-field measures to decrease runoff and encourage infiltration via the improvement of soil health and structure. Specific interventions include minimum tillage practices and the introduction of cover crops.",
            "Local Context": "Managing crop types to those utilising lower water budgets may provide greater water availability to the soil and reduce soil moisture deficits improving recharge capabilities and reducing the period channels remain ephemeral"
        },
        "Land Use Change": {
            "Description": "The interventions include specific areas of arable land or improved grassland that should be prioritized for land use change.",
            "Local Context": "The density of features is highest on the till covered hill in the south.  On the chalk, they concentrate along topographical depressions draining into rivers.  The most effective locations to target are likely to be those on the edge of the chalk, which are likely to be active more often capturing the run-off from the till.  Those that start on the chalk will likely have less run-off entering them on average."
        },
        "Slowly Permeable Soils": {
            "Description": "The interventions include areas with impeded soil permeability and superficial till cover.",
            "Local Context": "A significant cover of features are delineated over the expanse of till mainly on the plateau area.  Improved land cover management such as use of tree/grass buffer strips and shelter belts may slow flows and enhance aquifer recharge and provide water quality benefits.  The Very high infiltration areas are on the edge and show where the till is thinning and so may allow more recharge through."
        }
    }

    lista_nbs = ['SPS_Final','SM_Opmap_PRIO','LUC_10-perc']
    nbs_result = pd.DataFrame()
    for nbs in lista_nbs:

        path_nbs = obtener_ruta_data('NbS',nbs+'.shp')

        temp = calcular_area_files_shp(Polygon, path_nbs, 'Name',  count_polygons=False)
        temp = temp[temp['Name'] != "Total Area"]
        nbs_result = pd.concat([nbs_result,temp])

    nbs_result.columns = ['NbS','Area (Hectares)']

    # Agregar las columnas Description y Local Context
    nbs_result["Description"] =   nbs_result ["NbS"].map(lambda x: nbs_info[x]["Description"])
    nbs_result["Local Context"] =   nbs_result ["NbS"].map(lambda x: nbs_info[x]["Local Context"])

    return nbs_result
def determinar_cuenca(polygon_path, cuencas_path):
    """
    Determina la cuenca a la que pertenece el polígono del proyecto.

    Parámetros:
    - polygon_path: Ruta del shapefile del polígono del proyecto.
    - cuencas_path: Ruta del shapefile de las cuencas.

    Retorna:
    - El nombre de la cuenca a la que pertenece el polígono del proyecto.
    """
    try:
        # Leer el polígono del proyecto
        gdf_polygon = gpd.read_file(polygon_path)

        # Leer la capa de cuencas
        gdf_cuencas = gpd.read_file(cuencas_path)

        # Verificar el CRS del polígono del proyecto
        if gdf_polygon.crs is None:
            raise ValueError("El polígono del proyecto no tiene un CRS definido.")

        # Verificar el CRS de la capa de cuencas
        if gdf_cuencas.crs is None:
            raise ValueError("La capa de cuencas no tiene un CRS definido.")

        # Asegurarse de que ambas capas estén en el mismo CRS
        if gdf_polygon.crs != gdf_cuencas.crs:
            print(f"Reproyectando la capa de cuencas al CRS del polígono del proyecto: {gdf_polygon.crs}")
            gdf_cuencas = gdf_cuencas.to_crs(gdf_polygon.crs)

        # Realizar la intersección espacial
        intersected = gpd.overlay(gdf_polygon, gdf_cuencas, how='intersection')

        # Verificar si hay intersección
        if intersected.empty:
            raise ValueError("No se encontró intersección entre el polígono del proyecto y las cuencas.")

        # Obtener el nombre de la cuenca
        nombre_cuenca = intersected.iloc[0]["Catchment"]  # Asume que el campo se llama "Catchment"

        return nombre_cuenca

    except Exception as e:
        print(f"Error al determinar la cuenca: {e}")
        return None
def calcular_totales_por_area(resul_bng, df_combinado, resul_RF, resul_Rg, area_total):
    """
    Calcula los totales por área para biodiversidad, nutrientes (nitrógeno y fósforo),
    escorrentía y recarga.

    Parámetros:
    - resul_bng: DataFrame con los resultados de biodiversidad.
    - df_combinado: DataFrame con los resultados de nitrógeno y fósforo.
    - resul_RF: DataFrame con los resultados de escorrentía.
    - resul_Rg: DataFrame con los resultados de recarga.
    - area_total: Área total del proyecto en hectáreas.

    Retorna:
    - Un diccionario con los totales por área para cada clasificación.
    """
    try:
        # Calcular el total de biodiversidad por área
        total_biodiversidad = resul_bng["BNG Units"].sum() / area_total
        total_biodiversidad_t = resul_bng["BNG Units"].sum()

        # Calcular el total de nitrógeno por área
        total_nitrogeno = df_combinado["Nitrogen Export Reduction kg/yr"].sum() / area_total
        total_nitrogeno_t = df_combinado["Nitrogen Export Reduction kg/yr"].sum()

        # Calcular el total de fósforo por área
        total_fosforo = df_combinado["Phosphorus Export Reduction kg/yr"].sum() / area_total
        total_fosforo_t = df_combinado["Phosphorus Export Reduction kg/yr"].sum()

        # Calcular el total de escorrentía por área
        total_escorrentia = resul_RF["Runoff Reduction m3/yr"].sum() / area_total
        total_escorrentia_t = resul_RF["Runoff Reduction m3/yr"].sum()

        # Calcular el total de recarga por área
        total_recarga = resul_Rg["Infiltration uplift m3/yr"].sum() / area_total
        total_recarga_t = resul_Rg["Infiltration uplift m3/yr"].sum()

        # Crear un diccionario con los totales por área
        totales_por_area = {
            "Biodiversity": total_biodiversidad,
            "Nitrogen": total_nitrogeno,
            "Phosphorous": total_fosforo,
            "Runoff": total_escorrentia,
            "Infiltration": total_recarga
        }

        totales = {
            "Biodiversity": total_biodiversidad_t,
            "Nitrogen": total_nitrogeno_t,
            "Phosphorous": total_fosforo_t,
            "Runoff": total_escorrentia_t,
            "Infiltration": total_recarga_t
        }

        return totales_por_area

    except Exception as e:
        print(f"Error al calcular los totales por área: {e}")
        return None
def determinar_categorias(totales_por_area, cuenca):
    """
    Determina las categorías (Low, Medium, High) para cada clasificación comparando
    los valores del proyecto con los valores de referencia de la cuenca.

    Parámetros:
    - totales_por_area: Diccionario con los totales por área del proyecto.
    - cuenca: Nombre de la cuenca correspondiente al proyecto (ej: "Bure").

    Retorna:
    - Un diccionario con las categorías (Low, Medium, High) para cada clasificación.
    - Un diccionario con los valores de referencia de la cuenca.
    """
    try:
        # Diccionario de referencia de las cuencas
        cuencas = {
            "Wensum": {"Biodiversity": 0.625079, "Nitrogen": 6.579545, "Phosphorous": 0.139116, "Runoff": 509.399328, "Infiltration": 1566.865123},
            "Yare": {"Biodiversity": 0.685228, "Nitrogen": 6.521026, "Phosphorous": 0.168432, "Runoff": 586.780670, "Infiltration": 1674.128651},
            "Bure": {"Biodiversity": 0.447580, "Nitrogen": 4.764691, "Phosphorous": 0.079949, "Runoff": 287.172783, "Infiltration": 806.331120},
            "Ant": {"Biodiversity": 0.416463, "Nitrogen": 4.542870, "Phosphorous": 0.063903, "Runoff": 175.978761, "Infiltration": 127.352635}
        }

        cuencas_T = {
            "Wensum": {"Biodiversity": 36068.7, "Nitrogen": 379657.8, "Phosphorous": 8027.4, "Runoff": 29393742.09, "Infiltration": 20420389.08},
            "Yare": {"Biodiversity": 44821.4, "Nitrogen": 426546.60, "Phosphorous": 11017.30, "Runoff": 38381886.94, "Infiltration": 21818316.02},
            "Bure": {"Biodiversity": 20471.15, "Nitrogen": 217924.82, "Phosphorous": 3656.68, "Runoff": 13134550.81, "Infiltration": 10508623.21},
            "Ant": {"Biodiversity": 5427.61, "Nitrogen": 59205.6, "Phosphorous": 832.82, "Runoff": 2293467.84, "Infiltration": 1659741.04}
        }

        # Obtener los valores de referencia de la cuenca correspondiente
        valores_referencia = cuencas.get(cuenca)
        valores_T_referencia = cuencas_T.get(cuenca)
        if not valores_referencia:
            raise ValueError(f"No reference values were found for the basin: {cuenca}")

        # Función para asignar prioridad basada en el promedio
        def asignar_prioridad1(valor_proyecto, valor_referencia):
            if valor_referencia == 0:
                return "Unknown"  # Evitar división por cero
            promedio = (valor_proyecto / valor_referencia) * 100  # Calcular el porcentaje
            if promedio < 35:
                return "Low"
            elif 35 <= promedio <= 70:
                return "Medium"
            else:
                return "High"

        def asignar_prioridad(valor_proyecto, valor_referencia):
            if valor_referencia == 0:
                return "Unknown"
            porcentaje = (valor_proyecto / valor_referencia) * 100  # Porcentaje relativo

            # Definición de umbrales ajustados
            if porcentaje < 20:
                return "Very Low"
            elif 20 <= porcentaje < 35:
                return "Low"
            elif 35 <= porcentaje < 50:
                return "Medium"
            elif 50 <= porcentaje < 70:
                return "High"
            else:
                return "Very High"
        # Crear un diccionario para almacenar las categorías
        categorias = {}

        # Comparar cada valor del proyecto con los valores de referencia
        for clave in totales_por_area:
            if clave in valores_referencia:
                categorias[clave] = asignar_prioridad(totales_por_area[clave], valores_referencia[clave])
            else:
                print(f"Warning: No reference value was found for {clave}.")
                categorias[clave] = "Unknown"

        return categorias, valores_referencia

    except Exception as e:
        print(f"Error in determining the categories: {e}")
        return None
def exportar_diccionario_a_excel(diccionario, nombre_archivo, ruta):
    """
    Exporta un diccionario de DataFrames a un archivo de Excel, donde cada clave del diccionario
    se convierte en una hoja diferente.

    Parámetros:
    - diccionario: Diccionario donde las claves son nombres de hojas y los valores son DataFrames.
    - nombre_archivo: Nombre del archivo de Excel (sin la extensión .xlsx).
    - ruta: Ruta donde se guardará el archivo de Excel.
    """
    try:
        # Crear un archivo de Excel
        ruta_completa = f"{ruta}/{nombre_archivo}.xlsx"
        with pd.ExcelWriter(ruta_completa, engine='openpyxl') as writer:
            for nombre_hoja, dataframe in diccionario.items():
                # Exportar cada DataFrame a una hoja diferente
                dataframe.to_excel(writer, sheet_name=nombre_hoja, index=False)

        print(f"File successfully saved in: {ruta_completa}")

    except Exception as e:
        print(f"Error exporting dictionary to Excel: {e}")
def create_buffer_from_coords(x, y, radius_km, output_path):
    """
    Crea un buffer alrededor de un punto con coordenadas planas (en metros).

    Parámetros:
    - x: Coordenada X en metros.
    - y: Coordenada Y en metros.
    - radius_km: Radio del buffer en kilómetros.
    - output_path: Ruta donde se guardará el archivo shapefile.

    Retorna:
    - Un GeoDataFrame con el polígono del buffer.
    """
    
    #name_project = project_name.get()

    #os.makedirs('Project_' + name_project, exist_ok=True)

    #for folder in ['FIGURES','RESULTS','SHAPEFILES']:
    #    os.makedirs(r'Project_' + name_project+'/'+folder, exist_ok=True)
    
    # Extrae la ruta del directorio
    #directory = os.path.dirname(output_path)

    # Crea los directorios si no existen
    #os.makedirs(directory, exist_ok=True)
    # Crear un punto a partir de las coordenadas planas (en metros)
    point = Point(float(x), float(y))

    # Crear un GeoDataFrame con el punto y asignar un CRS proyectado (por ejemplo, EPSG:27700)
    gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:27700")  # Cambia el CRS según tu sistema de coordenadas

    # Crear un buffer alrededor del punto (convertir km a metros)
    buffer = gdf.geometry.buffer(radius_km * 1000)

    # Crear un nuevo GeoDataFrame con el polígono del buffer
    buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs="EPSG:27700")  # Mantener el mismo CRS

    # Guardar el shapefile en la ruta especificada

    buffer_gdf.to_file(output_path)

    return buffer_gdf
    
    
def generate_pdf_with_results(pdf_path, images, name_project, general, results, resultB, result_Benefit, categorias, nombre_cuenca,
                              valores_referencia,totales_project,text_points):
    """
    Genera un PDF horizontal con mapas, resultados y tablas organizadas.
    """
    # Contador de páginas

    # Función para agregar imágenes ajustadas


    # Función para agregar el número de página

    def agregar_imagen(img_path, y_pos, img_height=350):
        nonlocal y_position

        if y_pos - img_height < bottom_margin:  # Salto de página si no hay espacio
            c.showPage()
            y_pos = page_height - 50
        c.drawImage(img_path, left_margin, y_pos - img_height, width=available_width, height=img_height)
        y_position = y_pos - img_height - 20

    def agregar_tabla(dataframe, title, flag_title=True):
        nonlocal y_position
        c.setFont("Helvetica-Bold", 14)

        # Función para formatear números con comas como separadores de miles
        def format_number(num):
            try:
                # Intentar formatear el número con comas y dos decimales
                return "{:,.2f}".format(float(num))
            except ValueError:
                # Si no es un número, devolver el valor original
                return num

        # Crear tabla
        dataframe = dataframe.round(2)
        data = [dataframe.columns.tolist()] + dataframe.applymap(format_number).values.tolist()
        num_columns = len(dataframe.columns)
        col_widths = [available_width / num_columns] * num_columns
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 7)
        ]))

        # Calcular altura de la tabla
        table.wrapOn(c, available_width, y_position)
        table_height = table._height

        # Verificar si hay espacio suficiente para título + tabla
        if y_position - table_height - 20 < 50:
            c.showPage()
            y_position = page_height - 50

        # Agregar título y tabla
        if flag_title:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(left_margin, y_position, title)
            y_position -= 20
        table.drawOn(c, left_margin, y_position - table_height)
        y_position -= table_height + 20

    def agregar_texto_descriptivo(c, texto, left_margin, y_position, available_width, font_name="Helvetica",
                                  font_size=11, line_spacing=15):
        """
        Agrega un texto descriptivo al PDF con formato justificado, excepto la última línea.
        """

        c.setFont(font_name, font_size)  # Configurar fuente y tamaño inicial

        # Dividir el texto en líneas para que se ajuste al ancho disponible
        texto_lineas = []
        palabras = texto.split()
        linea_actual = ""
        for palabra in palabras:
            if c.stringWidth(linea_actual + " " + palabra, font_name, font_size) < available_width:
                linea_actual += " " + palabra if linea_actual else palabra
            else:
                texto_lineas.append(linea_actual)
                linea_actual = palabra
        if linea_actual:
            texto_lineas.append(linea_actual)

        # Dibujar cada línea del texto
        for i, linea in enumerate(texto_lineas):
            # Verificar si hay espacio suficiente para la línea actual
            if y_position - line_spacing < bottom_margin:
                c.showPage()  # Salto de página

                y_position = page_height - 50  # Reiniciar la posición vertical
                c.setFont(font_name, font_size)  # Restablecer la fuente después del salto de página

            palabras_linea = linea.split()
            num_palabras = len(palabras_linea)

            if num_palabras > 1 and i < len(texto_lineas) - 1:  # Justificar solo si no es la última línea
                # Calcular el ancho total del texto sin espacios
                ancho_texto = sum(c.stringWidth(palabra, font_name, font_size) for palabra in palabras_linea)
                # Calcular el espacio total disponible para los espacios entre palabras
                espacio_total = available_width - ancho_texto
                # Calcular el espacio entre palabras
                espacio_entre_palabras = espacio_total / (num_palabras - 1)

                # Dibujar cada palabra con el espaciado calculado
                x = left_margin
                for palabra in palabras_linea:
                    c.drawString(x, y_position, palabra)
                    x += c.stringWidth(palabra, font_name, font_size) + espacio_entre_palabras
            else:
                # Si es la última línea o solo hay una palabra, dibujarla sin justificar
                c.drawString(left_margin, y_position, linea)

            y_position -= line_spacing  # Espacio entre líneas

        y_position -= line_spacing  # Espacio adicional después del texto
        return y_position
    def agregar_tabla_larga(c, dataframe, title, left_margin, y_position, available_width, page_height, flag_title=True):
        """
        Agrega una tabla con mucho texto a un PDF, ajustando el ancho de las columnas y dividiendo el texto en varias líneas.

        Parámetros:
        - c: Objeto canvas de ReportLab.
        - dataframe: DataFrame con los datos.
        - title: Título de la tabla.
        - left_margin: Margen izquierdo.
        - y_position: Posición vertical actual.
        - available_width: Ancho disponible para la tabla.
        - page_height: Altura de la página.
        - flag_title: Si es True, agrega un título a la tabla.
        """

        try:
            # Estilos para el texto
            styles = getSampleStyleSheet()
            style_normal = styles['Normal']

            # Convertir el DataFrame en una lista de listas (incluyendo los encabezados)
            data = [dataframe.columns.tolist()] + dataframe.values.tolist()

            # Ajustar el ancho de las columnas según el contenido
            col_widths = [available_width * 0.2, available_width * 0.1, available_width * 0.35,
                          available_width * 0.35]  # Ajusta según tus necesidades

            # Crear una lista de listas con Paragraphs para manejar texto largo
            table_data = []
            for row in data:
                table_row = []
                for item in row:
                    # Usar Paragraph para manejar texto largo y dividirlo en varias líneas
                    table_row.append(Paragraph(str(item), style_normal))
                table_data.append(table_row)

            # Crear la tabla
            table = Table(table_data, colWidths=col_widths,
                          repeatRows=1)  # repeatRows=1 para repetir encabezados en cada página
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Fondo gris para la fila de encabezados
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Texto blanco para la fila de encabezados
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Alinear el texto a la izquierda
                ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Agregar bordes a la tabla
                ('FONTSIZE', (0, 0), (-1, -1), 8),  # Tamaño de fuente
                ('VALIGN', (0, 0), (-1, -1), 'TOP')  # Alinear el texto en la parte superior de la celda
            ]))

            # Calcular la altura de la tabla
            table.wrapOn(c, available_width, y_position)
            table_height = table._height

            # Verificar si hay espacio suficiente para el título y la tabla
            if y_position - table_height - 20 < 50:  # 50 es el margen inferior
                c.showPage()  # Crear una nueva página
                y_position = page_height - 50  # Reiniciar la posición vertical

            # Agregar el título si es necesario
            if flag_title:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(left_margin, y_position, title)
                y_position -= 20  # Ajustar la posición vertical después del título

            # Dibujar la tabla en el PDF
            table.drawOn(c, left_margin, y_position - table_height)
            y_position -= table_height + 25  # Ajustar la posición vertical después de la tabla

            return y_position  # Devolver la nueva posición vertical

        except Exception as e:
            print(f"Error al agregar la tabla: {e}")
            return y_position

    def generar_texto_recomendaciones(resultB):
        """
        Genera un texto de recomendaciones basado en los resultados de las tablas resultB["NBS_01"] y resultB["NBS_02"].

        Parámetros:
        - resultB: Diccionario que contiene las tablas resultB["NBS_01"] y resultB["NBS_02"].

        Retorna:
        - Un texto de recomendaciones basado en las áreas mayores que cero.
        """
        # Inicializar una lista para acumular las recomendaciones
        recomendaciones = []

        # Verificar si "Floodplain" tiene un área mayor que cero en resultB["NBS_01"]
        if "Floodplain Zone" in resultB["NBS_01"]["NbS"].values:
            area_floodplain = resultB["NBS_01"].loc[resultB["NBS_01"]["NbS"] == "Floodplain Zone", "Area (Hectares)"].values[0]
            if area_floodplain > 0:
                recomendaciones.append(
                    "Floodplain reconnection and restoration, including (where possible) channel bed raising, the removal of drainage on the floodplain and the installation of scrapes and ponds where possible.")

        # Verificar si "Riparian zone restoration" tiene un área mayor que cero en resultB["NBS_01"]
        if "Riparian Zone Restoration" in resultB["NBS_01"]["NbS"].values:
            area_riparian = \
            resultB["NBS_01"].loc[resultB["NBS_01"]["NbS"] == "Riparian Zone Restoration", "Area (Hectares)"].values[0]
            if area_riparian > 0:
                recomendaciones.append(
                    "Riparian zone restoration and expansion to buffer water courses, including targeted grassland restoration and tree planting where deemed appropriate.")

        if "Runoff Attenuation Features" in resultB["NBS_01"]["NbS"].values:
            area_raf = \
            resultB["NBS_01"].loc[resultB["NBS_01"]["NbS"] == "Runoff Attenuation Features", "Area (Hectares)"].values[0]
            if area_raf > 0:
                recomendaciones.append(
                    "Construction of small runoff attenuation features (ponds and scrapes) in topographical depressions to capture and store surface runoff, reducing peak flows and enhancing groundwater recharge")
                recomendaciones.append(
                    "Construction of larger runoff attenuation features in drainage network or on significant flow pathways to capture and store surface runoff, reducing peak flows and enhancing groundwater recharge")

        # Verificar si "Soil Management" tiene un área mayor que cero en resultB["NBS_02"]
        if "Soil Management" in resultB["NBS_02"]["NbS"].values:
            area_soil_management = \
            resultB["NBS_02"].loc[resultB["NBS_02"]["NbS"] == "Soil Management", "Area (Hectares)"].values[0]
            if area_soil_management > 0:
                recomendaciones.append(
                    "Soil management activities in arable areas, including practising cover cropping and minimum tillage agricultural practices.")

        if "Land Use Change" in resultB["NBS_02"]["NbS"].values:
            area_luc = \
            resultB["NBS_02"].loc[resultB["NBS_02"]["NbS"] == "Land Use Change", "Area (Hectares)"].values[0]
            if area_luc > 0:
                recomendaciones.append(
                    "Land use change initiatives, focusing on converting arable land to grassland or woodland, particularly in areas with high runoff potential, also implementation of agroforestry systems, combining tree planting with agricultural practices, to stabilize soil structure and increase biodiversity in the catchment, .")

        # Unir las recomendaciones en un solo texto, separadas por saltos de línea
        texto_final = "\n".join(recomendaciones)

        return texto_final

    def agregar_texto_vinetas(c, texto, left_margin, y_position, available_width, font_name="Helvetica", font_size=11,
                              line_spacing=15, bullet="•"):
        """
        Agrega un texto con viñetas al PDF, manejando saltos de página y posición vertical.

        Parámetros:
        - c: Objeto canvas de ReportLab.
        - texto: Texto con viñetas (cada viñeta separada por un salto de línea).
        - left_margin: Margen izquierdo.
        - y_position: Posición vertical actual.
        - available_width: Ancho disponible para el texto.
        - font_name: Nombre de la fuente (por defecto "Helvetica").
        - font_size: Tamaño de la fuente (por defecto 11).
        - line_spacing: Espacio entre líneas (por defecto 15).
        - bullet: Símbolo de la viñeta (por defecto "•").

        Retorna:
        - y_position: Nueva posición vertical después de agregar el texto.
        """

        c.setFont(font_name, font_size)  # Configurar fuente y tamaño

        # Dividir el texto en viñetas (cada viñeta separada por un salto de línea)
        viñetas = texto.split("\n")

        # Dibujar cada viñeta
        for viñeta in viñetas:
            # Verificar si hay espacio suficiente para la viñeta actual
            if y_position - line_spacing < bottom_margin:
                c.showPage()  # Salto de página
                y_position = page_height - 50  # Reiniciar la posición vertical
                c.setFont(font_name, font_size)  # Restablecer la fuente después del salto de página

            # Dibujar el símbolo de la viñeta
            c.drawString(left_margin, y_position, bullet)
            x = left_margin + 10  # Espacio entre la viñeta y el texto

            # Dividir la viñeta en palabras para ajustarla al ancho disponible
            palabras = viñeta.strip().split()
            linea_actual = ""
            for palabra in palabras:
                if c.stringWidth(linea_actual + " " + palabra, font_name,
                                 font_size) < available_width - 10:  # Restar espacio de la viñeta
                    linea_actual += " " + palabra if linea_actual else palabra
                else:
                    # Dibujar la línea actual
                    c.drawString(x, y_position, linea_actual)
                    y_position -= line_spacing  # Espacio entre líneas
                    linea_actual = palabra  # Comenzar una nueva línea con la palabra actual

                    # Verificar si hay espacio suficiente para la nueva línea
                    if y_position - line_spacing < bottom_margin:
                        c.showPage()  # Salto de página
                        y_position = page_height - 50  # Reiniciar la posición vertical
                        c.setFont(font_name, font_size)  # Restablecer la fuente después del salto de página

            # Dibujar la última línea de la viñeta
            if linea_actual:
                c.drawString(x, y_position, linea_actual)
                y_position -= line_spacing  # Espacio entre líneas

        y_position -= line_spacing  # Espacio adicional después del texto
        return y_position

    def dibujar_bloque_color(c, x, y, ancho, alto, categoria):
        from reportlab.lib import colors
        """
        Dibuja un bloque de color (rojo, amarillo, verde) según la categoría.

        Parámetros:
        - c: Objeto canvas de ReportLab.
        - x, y: Posición del bloque.
        - ancho, alto: Tamaño del bloque.
        - categoria: Categoría (bajo, medio, alto).
        """
        # Definir colores según la categoría
        # color = {
        #     "Low": red,
        #     "Medium": yellow,
        #     "High": green
        # }.get(categoria, red)  # Por defecto, rojo si la categoría no es válida

        # Definir colores para todas las categorías
        colores = {
            "Very Low":colors.red ,  # Rojo claro
            "Low":colors.HexColor("#de7321"),  # Rojo estándar
            "Medium": colors.yellow,  # Amarillo
            "High": colors.HexColor("#2ebf2e"),  # Verde claro
            "Very High": green
        }

        # Obtener el color (gris por defecto si no existe)
        color = colores.get(categoria, colors.lightgrey)

        # Dibujar el bloque de color
        c.setFillColor(color)
        c.rect(x, y, ancho, alto, fill=1, stroke=0)

        # Dibujar el texto de la categoría en el bloque
        c.setFillColor(colors.black)  # Texto en negro
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + 5, y + 5, categoria)  # Ajustar la posición del texto

    def construir_dataframe(resultB, area_total):
        # Lista para almacenar los datos

        datos = []
        for nbs_key in ["NBS_01", "NBS_02"]:
            if nbs_key in resultB:
                # Extraer la lista de datos
                nbs_data = resultB[nbs_key]

                if "NbS" in nbs_data.columns and "Area (Hectares)" in nbs_data.columns:
                    for _, row in nbs_data.iterrows():
                        nbs_name = row["NbS"]
                        area = row["Area (Hectares)"]
                        datos.append([nbs_name, area])

        # Crear el DataFrame


        df = pd.DataFrame(datos, columns=["NbS", "Area (Hectares)"])
        # Eliminar la fila donde NbS es "Slowly Permeable Soils"
        df = df[df["NbS"] != "Slowly Permeable Soils"]
        # Calcular el porcentaje de área
        df["Area Percentage (%)"] = (df["Area (Hectares)"] / area_total) * 100

        df["Area Percentage (%)"] = (df["Area (Hectares)"] / area_total * 100).round(2)

        total_area = df["Area (Hectares)"].sum()
        total_row = pd.DataFrame({
            "NbS": ["Total Area"],
            "Area (Hectares)": [total_area],
            "Area Percentage (%)": [np.nan]  # Usamos NaN
        })
        df = pd.concat([df, total_row], ignore_index=True)
        df["Area Percentage (%)"] = df["Area Percentage (%)"].fillna("-")

        return df

    def agregar_logos_inicio(c, page_width, page_height, logo_size=140):
        """
        Agrega dos logos en la parte superior izquierda y derecha de la primera página.
        """
        # Logo izquierdo (esquina superior izquierda)
        file_1 = obtener_ruta_data("logos", "N4W Logo_Tagline Logo Colour.png")
        file_2 = obtener_ruta_data("logos", "Norfolk.png")
        from reportlab.lib.utils import ImageReader
        # logo1 = ImageReader(file_1)
        # logo2 = ImageReader(file_2)
        c.drawImage(file_1, 40, page_height - logo_size - 10,
                    width=logo_size, height=logo_size, preserveAspectRatio=True)

        c.drawImage(file_2, page_width - logo_size - 40, page_height - logo_size - 10,
                    width=logo_size, height=logo_size, preserveAspectRatio=True)

    c = canvas.Canvas(pdf_path, pagesize=letter)  # Orientación vertical
    page_width, page_height = letter  # Dimensiones de la página en orientación vertical
    agregar_logos_inicio(c, page_width, page_height)


    left_margin = 50
    right_margin = 50
    available_width = page_width - left_margin - right_margin
    bottom_margin = 50

    # Título principal
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, page_height - 120, "NbS for water security prioritisation tool – site assessment report")



    c.setFont("Helvetica", 12)

    c.drawString(left_margin, page_height - 150, f"Site Assessment: {name_project}")
    y_position = page_height - 170
    texto_0= (f"This assessment report details the background characteristics of "+f"{name_project}"+"  and provides an automatically generated assessment of opportunities to deliver Nature-based Solutions (NbS) on the site. It also provides high-level calculations of potential uplifts which could be achieved through the implementation of recommended interventions. Specifically, it evaluates:."
    )

    y_position = agregar_texto_descriptivo(c, texto_0 , left_margin, y_position, available_width)

    list_initial= ['Existing biophysical conditions, including habitats, topography, geology, and soil type.',
                   'Opportunities for implementing NbS, such as riparian restoration, runoff attenuation features, floodplain reconnection, and land cover management.',
                   'Potential benefits, including estimated biodiversity net gain credits, reductions in phosphorus and nitrogen export for nutrient neutrality, runoff reduction, and groundwater infiltration enhancement.',
                      ]
    texto_0001 = "\n".join(list_initial)
    y_position = agregar_texto_vinetas(c, texto_0001, left_margin, y_position, available_width)

    ################################## Discleimer ############################################

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, "Disclaimer")
    y_position -= 20
    c.setFont("Helvetica", 12)
    c.drawString(left_margin, y_position, "Important notice: Indicative assessment tool")
    y_position -= 20
    texto_21 = ("This tool provides a high-level scoping of biophysical conditions, the feasibility of implementing nature-based solutions (NbS), and the potential environmental benefits that can be achieved through the implementation of recommended interventions. Specifically, it evaluates:")

    y_position = agregar_texto_descriptivo(c, texto_21 , left_margin, y_position, available_width)
    list_disclaimer = ['Existing biophysical conditions, including habitats, topography, geology, and soil type.',
                       'Opportunities for implementing NbS, such as riparian restoration, runoff attenuation features, floodplain reconnection, and land cover management',
                       'Potential benefits, including estimated biodiversity net gain credits, reductions in phosphorus and nitrogen export for nutrient neutrality, flood mitigation, runoff reduction, and groundwater infiltration enhancement.']
    texto_final = "\n".join(list_disclaimer)
    y_position = agregar_texto_vinetas(c, texto_final, left_margin, y_position, available_width)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, "Limitations and legal notice")
    y_position -= 20
    texto_21 = ("The results generated by this tool are indicative only and intended for preliminary assessment. They do not constitute a definitive feasibility study, financial projection, or regulatory compliance assessment.")
    y_position = agregar_texto_descriptivo(c, texto_21 , left_margin, y_position, available_width)

    texto_23 = ("Users are responsible for ensuring compliance with all relevant regulations and should seek independent professional advice before proceeding with any project. By using this tool, users acknowledge that the outputs are non-binding and that neither the tool's developers nor any associated parties accept liability for decisions made based on its results.")
    y_position = agregar_texto_descriptivo(c, texto_23 , left_margin, y_position, available_width)

    ###################### punto 1 ####################################################################################
    c.showPage()
    page_width, page_height = letter  # Dimensiones de la página en orientación vertical
    y_position = page_height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position , f"1. Project site background")
    y_position -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position , f"Background / Baseline Map")
    y_position -= 25

    area = general[0]
    fields = general[1]
    texto_1 = (f"{name_project} covers a total area of "+ f"{str(area)} hectares. It is composed of"+f" {str(fields)} field parcels and is located in the"+ f" {nombre_cuenca} catchment."
    )

    y_position = agregar_texto_descriptivo(c, texto_1 , left_margin, y_position, available_width)

    # Insertar la imagen principal
    agregar_imagen(images['Location'], y_position)

    #Land use
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position , f"Land use")
    y_position -= 25

    texto_1 = ("Current land use helps identify where projects could be delivered, and where protection and expansion of existing habitat can be achieved. In general, agricultural land types including arable and horticulture, or improved grassland, represent areas in which NbS could – given the right circumstances – be delivered. Land uses of urban or suburban areas and existing (semi)natural habitat types are unlikely to be possible for the NbS delivery." )

    y_position = agregar_texto_descriptivo(c, texto_1 , left_margin, y_position, available_width)

    texto_2 = ("Current land use in the project area includes:" )

    y_position = agregar_texto_descriptivo(c, texto_2 , left_margin, y_position, available_width)
    agregar_tabla(results["Land Use"], '',flag_title=False)


    # Habitats
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Habitats")
    y_position -= 25

    texto_3 = (
        "Existing areas of priority habitats (for example, broadleaved woodland or lowland fens) should be protected or restored. Where possible, new habitat should be created adjacent to existing habitat types to improve landscape scale connectivity. Buffering of existing habitat areas (including rivers) by delivering NbS is also beneficial to improving their state. ")

    y_position = agregar_texto_descriptivo(c, texto_3, left_margin, y_position, available_width)

    texto_4 = ("Note that network enhancement and expansion, alongside fragmentation action zones, represent priority areas for delivery to achieve connectivity and buffering to existing habitats.")

    y_position = agregar_texto_descriptivo(c, texto_4, left_margin, y_position, available_width)

    texto_5= ("In terms of habitats, the site contains:" )

    y_position = agregar_texto_descriptivo(c, texto_5 , left_margin, y_position, available_width)

    results["Habitats"].columns = ['Habitat Type','Area (hectares)']

    agregar_tabla(results["Habitats"], '', flag_title=False)


    # Topography, Geology and Soil Type
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Topography, geology and soil Type")
    y_position -= 20

    texto_6= ("Topography, geography and soil type all represent constraining or enabling factors for the delivery of NbS. Topography may dictate what sorts of features should be delivered in the area, with areas of high slope representing high priority for the delivery of specific NbS." )

    y_position = agregar_texto_descriptivo(c, texto_6 , left_margin, y_position, available_width)

    texto_7= ("The topography of the area is the following:" )

    y_position = agregar_texto_descriptivo(c, texto_7 , left_margin, y_position, available_width)

    agregar_tabla(results["Slope"], '', flag_title=False)

    texto_72= ("**Total area calculated from the raster may not match the polygon area due to raster resolution and clipping. Raster pixels have a fixed size, and when clipping to the polygon, all touching pixels are included, even those partially outside the polygon. This can include adjacent pixels not fully within the area. Additionally, raster resolution (pixel size) affects precision: larger pixels reduce accuracy. Small discrepancies between the polygon area and raster-derived area are expected." )

    y_position = agregar_texto_descriptivo(c, texto_72 , left_margin, y_position, available_width)

    texto_8= ("Geology and soil type dictate the hydrology of the site through influencing partitioning between runoff and infiltration, and determining whether water infiltrated will reach groundwater. Areas overlying highly permeable bedrock geologies like chalk should be priority for delivering infiltration-related features. Heavy soil types and superficial geologies will generate runoff, meaning features to reduce runoff and store water should be prioritised." )

    y_position = agregar_texto_descriptivo(c, texto_8 , left_margin, y_position, available_width)

    texto_9= ("The soil type and geography of the area is the following:." )

    y_position = agregar_texto_descriptivo(c, texto_9 , left_margin, y_position, available_width)

    results["Bedrock"].columns = ['Bedrock Geology Type','Area (hectares)']
    agregar_tabla(results["Bedrock"], '', flag_title=False)

    results["Superficial Geology"].columns = ['Superficial Geology Type','Area (hectares)']
    agregar_tabla(results["Superficial Geology"], '', flag_title=False)


    results["Soil"].columns = ['Soil Type','Area (hectares)']
    agregar_tabla(results["Soil"], '', flag_title=False)


    ############### NBS OPPORTUNITIES ##########
    # y_position -= 25
    c.showPage()
    page_width, page_height = letter  # Dimensiones de la página en orientación vertical
    y_position = page_height - 50


    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position , f"2. NbS opportunities")
    y_position -= 25

    # Land use
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Feature-level Interventions")
    y_position -= 25

    texto_10 = ("These interventions aim to restore or recreate natural processes in target areas. These often include measures targeting hydrologically important areas including floodplains, riparian areas and areas of runoff accumulation. They aim to buffer hydrological systems and store water, encouraging runoff reduction and infiltration.")

    y_position = agregar_texto_descriptivo(c, texto_10, left_margin, y_position, available_width)

    agregar_imagen(images['NBS_01'], y_position)

    y_position = agregar_tabla_larga(c, resultB["NBS_01"], "", left_margin, y_position, available_width, page_height,flag_title=False)


    c.showPage() #nueva pagina
    y_position = page_height - 50  # Reinicia la posición vertical en la nueva página

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Land cover management")
    y_position -= 20

    texto_11 = ("Land cover management interventions aim to improve land use practices or provide wholesale land use change. When delivered in targeted areas, these interventions can help increase infiltration and decrease runoff, whilst also reducing nutrient and sediment export from fields.")

    y_position = agregar_texto_descriptivo(c, texto_11, left_margin, y_position, available_width)

    texto_11lm = ("Note that soil management is recommended for delivery in arable systems across the catchment, whereas wholescale land use change is only recommended in high priority areas. High priority areas for land use change were identified based on connectivity with existing habitats – hence complementing targeting in the Local Nature Recovery Strategy. ")

    y_position = agregar_texto_descriptivo(c, texto_11lm, left_margin, y_position, available_width)

    texto_12= ("The following figure presents the land cover management interventions, an important point is that many areas of Slowly permeable soils are overlapping in soil management areas:" )

    y_position = agregar_texto_descriptivo(c, texto_12 , left_margin, y_position, available_width)

    agregar_imagen(images['NBS_02'], y_position)

    y_position = agregar_tabla_larga(c, resultB["NBS_02"], "", left_margin, y_position, available_width, page_height,
                                     flag_title=False)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Summary of interventions")
    y_position -= 25

    texto_13= ("Based on the information presented in the tables and maps above, a summary of the proposed NBS interventions is presented below:" )

    y_position = agregar_texto_descriptivo(c, texto_13 , left_margin, y_position, available_width)
    texto_14 = generar_texto_recomendaciones(resultB)

    y_position = agregar_texto_vinetas(c, texto_14, left_margin, y_position, available_width)

    ####################### oportunidades NBS beneficios
    # y_position -= 25
    c.showPage()
    page_width, page_height = letter  # Dimensiones de la página en orientación vertical
    y_position = page_height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position , f"3. NbS benefits assessment")
    y_position -= 25

    texto_15= ("This section provides an overview of the potential benefits that the project could deliver based on simplified calculations. These calculations are high level and should only be used for a quick understanding of potential benefits that could be achieved – more detailed assessments should be completed at a later project stage. " )

    y_position = agregar_texto_descriptivo(c, texto_15 , left_margin, y_position, available_width)

    #
    # y_position -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Summary")
    y_position -= 20
    tex_sum1 = ("Note that the following assessments are performed based on the assumption that all of the NbS opportunities identified in the section above are implemented. ")

    y_position = agregar_texto_descriptivo(c, tex_sum1 , left_margin, y_position, available_width,"Helvetica-Bold")

    tex_sum2 = ("Note also that runoff attenuation features are delivered as a combination of large features located in-channel or on flow pathways; and small features delivered in topographic depressions. These are all delivered based on an average density within the opportunity area identified within the maps above. ")
    y_position = agregar_texto_descriptivo(c, tex_sum2 , left_margin, y_position, available_width, )
    tex_sum3 = ("It is assumed that within the target area, the following is delivered:")
    y_position = agregar_texto_descriptivo(c, tex_sum3 , left_margin, y_position, available_width, )

    df_summary = construir_dataframe(resultB, area)
    agregar_tabla(df_summary, '', flag_title=False)


    tex_sum33 = ("** It is important to mention that these nbs suggest the potential for implementation, in the case of ‘Soil Management’ this covers a large part of the territory and in some cases overlaps other NBS, in the determination of benefits the different NBS are prioritised without overcoming any intervention.")
    y_position = agregar_texto_descriptivo(c, tex_sum33 , left_margin, y_position, available_width, )


    # y_position -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Full results")
    y_position -= 20


    tex_kpis_2 = ('Market benefits:')
    y_position = agregar_texto_descriptivo(c, tex_kpis_2, left_margin, y_position, available_width, )

    def format_number(num):
        try:
            # Intentar formatear el número con comas y dos decimales
            return "{:,.2f}".format(float(num))
        except ValueError:
            # Si no es un número, devolver el valor original
            return num

    val_market_BNG = np.round(totales_project["Biodiversity"]*area,2)
    val_market_N = np.round(totales_project["Nitrogen"]*area,2)
    val_market_P = np.round(totales_project["Phosphorous"]*area,2)

    #"{:,.2f}".format(float(num))
    list_market = [f"{format_number(val_market_BNG)} BNG units delivered.",
                 f"{format_number(val_market_P)} kg per year of Phosphorus export mitigated.",
                 f"{format_number(val_market_N)} kg per year of Nitrogen export mitigated ", ]
    texto_market = "\n".join(list_market )

    y_position = agregar_texto_vinetas(c, texto_market, left_margin, y_position, available_width)


    tex_kpis_3 = ('Non-market benefits:')
    y_position = agregar_texto_descriptivo(c, tex_kpis_3, left_margin, y_position, available_width, )


    val_nomarket_R = np.round(totales_project["Runoff"]*area,2)
    val_market_g = np.round(totales_project["Infiltration"]*area,2)
    list_nomarket = ([f"{format_number(val_nomarket_R)} per year of avoided runoff.",
                 f"{format_number(val_market_g)} per year of additional infiltration to groundwater.",
                      text_points[0],
                      text_points[1]])

    # list_nomarket.append(text_points)
    texto_nomarket = "\n".join(list_nomarket )

    y_position = agregar_texto_vinetas(c, texto_nomarket, left_margin, y_position, available_width)

    y_position -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Biodiversity net gain")
    y_position -= 20
    texto_16= ("NbS not only address water security challenges but also contribute significantly to biodiversity enhancement. The following table outlines the estimated Biodiversity Net Gain (BNG) units generated by different NbS interventions. These units reflect the potential for habitat creation and restoration, which are critical for supporting local wildlife and improving ecological resilience.")
    y_position = agregar_texto_descriptivo(c, texto_16 , left_margin, y_position, available_width)

    texto_16_2= ("Note that this is based on simplifications of the statutory biodiversity metric calculations, based on the following assumptions:")
    y_position = agregar_texto_descriptivo(c, texto_16_2 , left_margin, y_position, available_width)


    list_16_2 = ['Habitats delivered are used to generate offsite units',
                 'Habitats are delivered to a moderate condition',
                 'No habitat is created in advance and there is no delay in habitat creation']
    texto_16_2 = "\n".join(list_16_2)

    y_position = agregar_texto_vinetas(c, texto_16_2 , left_margin, y_position, available_width)

    agregar_tabla(result_Benefit["BNG"], '', flag_title=False)


    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Nutrient neutrality")
    y_position -= 20
    texto_17= ("Nutrient pollution, particularly from nitrogen and phosphorus, poses a significant threat to water quality in Norfolk. The table below quantifies the reduction in nutrient exports achieved through various NbS interventions in the project area. These reductions are essential for meeting regulatory requirements, such as Nutrient Neutrality, and for protecting sensitive ecosystems like the Norfolk Broads and River Wensum." )
    y_position = agregar_texto_descriptivo(c, texto_17 , left_margin, y_position, available_width)

    texto_17_2= ("Note that delivery calculations are based on simplifications of the Farmscoper tool and modelling conducted by the Environment Agency and Norfolk Water Strategy Programme." )
    y_position = agregar_texto_descriptivo(c, texto_17_2 , left_margin, y_position, available_width)
    agregar_tabla(result_Benefit["NN"], '', flag_title=False)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Flooding and runoff")
    y_position -= 20
    texto_18= ("Managing flood risk and reducing surface runoff are key objectives of the Norfolk Water Fund. The following table presents the estimated reduction in runoff volume achieved by different NbS interventions in the project area. By slowing down and storing water, these solutions help mitigate flood risk, improve water availability, and enhance the resilience of local communities and ecosystems." )
    y_position = agregar_texto_descriptivo(c, texto_18 , left_margin, y_position, available_width)
    texto_18_2= ("Note that delivery calculations are based on simplifications of modelling conducted by the Environment Agency and Norfolk Water Strategy Programme." )
    y_position = agregar_texto_descriptivo(c, texto_18_2 , left_margin, y_position, available_width)

    agregar_tabla(result_Benefit["runoff"], '', flag_title=False)


    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, f"Infiltration to groundwater")
    y_position -= 20
    texto_19= ("Enhancing groundwater recharge is a critical component of improving water security in Norfolk. The table below shows the estimated uplift in groundwater infiltration resulting from various NbS interventions. These interventions help replenish aquifers, support baseflows in rivers, and ensure a more sustainable water supply for both people and nature." )
    y_position = agregar_texto_descriptivo(c, texto_19 , left_margin, y_position, available_width)

    texto_19_2= ("Note that delivery calculations are based on simplifications of modelling conducted by the Environment Agency and Norfolk Water Strategy Programme." )
    y_position = agregar_texto_descriptivo(c, texto_19_2 , left_margin, y_position, available_width)
    agregar_tabla(result_Benefit["recharge"], '', flag_title=False)

    # agregar esta parte
    c.showPage() #nueva pagina
    y_position = page_height - 50  # Reinicia la posición vertical en la nueva página
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position , f"4. Prioritisation of potential benefits")
    y_position -= 20

    texto_20 = ("This section presents an assessment of the project’s level of priority for delivery, helping to highlight projects that will have the greatest positive impact on biodiversity, water resources, and nutrient reduction.")

    y_position = agregar_texto_descriptivo(c, texto_20 , left_margin, y_position, available_width)

    texto_20_1 = ("Prioritisation is achieved by comparing the potential benefits delivered by this project with the catchment-averaged opportunity to achieve benefits through NbS delivery. If the project can deliver benefits over and above the average for the catchment, then it is deemed a high priority for delivery – this will likely be due to an increased total opportunity to deliver NbS, or the fact that NbS delivered are in priority areas for achieving uplift. ")

    y_position = agregar_texto_descriptivo(c, texto_20_1 , left_margin, y_position, available_width)


    # Agregar categorías con bloques de colores y flechas alineadas
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, "Prioritisation categories")
    y_position -= 30

    # Agrupar las categorías
    categorias_agrupadas = {
        "Potential Biodiversity Net Gain": {"Biodiversity": categorias["Biodiversity"]},
        "Potential Nutrient Neutrality": {
            "Nitrogen": categorias["Nitrogen"],
            "Phosphorous": categorias["Phosphorous"]
        },
        "Potential Water Resources": {
            "Runoff": categorias["Runoff"],
            "Infiltration": categorias["Infiltration"]
        }
    }


    def dibujar_categorias_con_flechas(c, categorias_agrupadas, left_margin, y_position, page_height,
                                       espacio_entre_grupos=25):
        """
        Dibuja las categorías con flechas alineadas y maneja el espacio de la página.

        Parámetros:
        - c: Objeto canvas de ReportLab.
        - categorias_agrupadas: Diccionario con las categorías agrupadas por título.
        - left_margin: Margen izquierdo de la página.
        - y_position: Posición vertical inicial.
        - page_height: Altura de la página.
        - espacio_entre_grupos: Espacio vertical entre grupos de categorías.

        Retorna:
        - y_position: Posición vertical final después de dibujar las categorías.
        """
        # Calcular la longitud máxima de las palabras
        palabras = ["Biodiversity", "Nitrogen", "Phosphorous", "Runoff", "Infiltration"]
        longitud_maxima = max(len(palabra) for palabra in palabras) * 6  # Ajustar según el tamaño de la fuente

        # Definir el espacio mínimo requerido antes de pasar a una nueva página
        espacio_minimo = 50  # Espacio mínimo en la parte inferior de la página

        # Espacio adicional después de las categorías (1-2 cm, equivalente a 28-56 puntos)
        espacio_adicional = 28  # 1 cm = 28.35 puntos (usamos 28 para simplificar)

        # Dibujar las categorías con flechas alineadas
        for titulo, datos in categorias_agrupadas.items():
            # Verificar si hay suficiente espacio para el título y al menos una categoría
            espacio_requerido = 20 + (len(datos) * 25) + espacio_entre_grupos
            if y_position - espacio_requerido < espacio_minimo:
                c.showPage()  # Pasar a una nueva página
                y_position = page_height - 50  # Reiniciar la posición vertical

            # Dibujar el título del grupo
            c.setFont("Helvetica-Bold", 12)
            c.drawString(left_margin, y_position, titulo)
            y_position -= 20

            # Dibujar cada categoría dentro del grupo
            for clave, valor in datos.items():
                # Verificar si hay suficiente espacio para la categoría actual
                if y_position - 25 < espacio_minimo:
                    c.showPage()  # Pasar a una nueva página
                    y_position = page_height - 50  # Reiniciar la posición vertical

                # Dibujar el texto de la categoría
                c.setFont("Helvetica", 10)
                c.drawString(left_margin, y_position, clave)

                # Dibujar la flecha alineada
                x_flecha = left_margin + longitud_maxima
                c.drawString(x_flecha, y_position, "--------------------->")

                # Dibujar el bloque de color
                x_bloque = x_flecha + 100  # Espacio después de la flecha
                dibujar_bloque_color(c, x_bloque, y_position - 5, 55, 20, valor)

                y_position -= 20  # Ajustar la posición vertical

            y_position -= espacio_entre_grupos  # Espacio adicional entre grupos

        # Dejar un espacio razonable después de las categorías (1-2 cm)
        y_position -= espacio_adicional

        return y_position  # Devolver la posición vertical final


    y_position = dibujar_categorias_con_flechas(c, categorias_agrupadas, left_margin, y_position, page_height)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, "Full breakdown")
    y_position -= 20


    texto_interactivo = (
        f"The project {name_project} is located in the {nombre_cuenca} catchment. "
        "The prioritisation is therefore developed using the reference values per hectare for this catchment (based on the total uplifts resulting from delivering all NbS identified across the catchment, divided by catchment area). These values are then compared to the project-specific values in the following table, "
    )

    y_position = agregar_texto_descriptivo(c, texto_interactivo, left_margin, y_position, available_width)

    # Crear el DataFrame
    data = {
        "Metric": [
            "Runoff Reduction (m3/yr)",
            "Infiltration Enhancement (m3/yr)",
            "Nitrogen Export Reduction (kg/yr)",
            "Phosphorus Reduction (kg/yr)",
            "Biodiversity Credits (-)"
        ],
        "Catchment Scale Delivery Per Hectare": [
            valores_referencia["Runoff"],
            valores_referencia["Infiltration"],
            valores_referencia["Nitrogen"],
            valores_referencia["Phosphorous"],
            valores_referencia["Biodiversity"]
        ],
        "Project Delivery Per Hectare": [
            totales_project["Runoff"] ,
            totales_project["Infiltration"] ,
            totales_project["Nitrogen"] ,
            totales_project["Phosphorous"] ,
            totales_project["Biodiversity"]
        ]
    }

    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Calcular la columna de porcentaje
    df["Project Delivery as a Percentage of Catchment Scale Delivery (%)"] = (
            df["Project Delivery Per Hectare"] / df["Catchment Scale Delivery Per Hectare"] * 100
    )

    def agregar_tabla_adjust(dataframe, title, flag_title=True, header_height=30):
        """
        Agrega una tabla al PDF con control sobre la altura de la primera fila (encabezados).
        Usa Paragraph para ajustar el texto de la primera fila.

        Parámetros:
        - dataframe: DataFrame con los datos.
        - title: Título de la tabla.
        - flag_title: Si es True, agrega un título a la tabla.
        - header_height: Altura de la primera fila (encabezados).
        """
        nonlocal y_position
        c.setFont("Helvetica-Bold", 14)

        # Crear tabla
        dataframe = dataframe.round(2)
        data = dataframe.round(2).astype(str).values.tolist()
        num_columns = len(dataframe.columns)

        # Definir anchos personalizados para las columnas
        col_widths = [120, 120, 120, 160]  # Ajusta estos valores según tus necesidades

        # Estilos para el texto
        styles = getSampleStyleSheet()
        header_style = styles['Normal']
        header_style.fontName = 'Helvetica-Bold'
        header_style.fontSize = 8
        header_style.leading = 10  # Espaciado entre líneas

        # Convertir los encabezados en Paragraphs para manejar texto largo
        headers = [Paragraph(str(col), header_style) for col in dataframe.columns]

        # Combinar encabezados con los datos
        table_data = [headers] + data

        # Definir la altura de la primera fila (encabezados) y dejar el resto automático
        row_heights = [header_height] + [None] * (len(data))

        # Crear la tabla
        table = Table(table_data, colWidths=col_widths, rowHeights=row_heights)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Fondo gris para la fila de encabezados
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Texto blanco para la fila de encabezados
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),  # Centrar encabezados
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),  # Centrar valores numéricos
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear verticalmente al centro
            ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Agregar bordes a la tabla
            ('FONTSIZE', (0, 0), (-1, -1), 8),  # Tamaño de fuente
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Fuente en negrita para los encabezados
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica')  # Fuente normal para los datos
        ]))

        # Calcular altura de la tabla
        table.wrapOn(c, available_width, y_position)
        table_height = table._height

        # Verificar si hay espacio suficiente para título + tabla
        if y_position - table_height - 20 < 50:
            c.showPage()
            y_position = page_height - 50

        # Agregar título y tabla
        if flag_title:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(left_margin, y_position, title)
            y_position -= 20
        table.drawOn(c, left_margin, y_position - table_height)
        y_position -= table_height + 20

    agregar_tabla_adjust(df, '', flag_title=False)

    # c.setFont("Helvetica-Bold", 12)
    # c.drawString(left_margin, y_position, "Suggested conclusion")
    # y_position -= 25
    #
    # texto_conclusion = ("The assignment of scores was based on the obtained categories (Low, Medium, High), where each category was assigned a specific value: High (25 points), Medium (15 points), and Low (5 points). These scores were summed to evaluate the project's potential in three key areas: water security, biodiversity, and nutrient neutrality. Finally, if the total score exceeds the minimum requirement of 50 points, the project is considered viable (Proceed to feasibility); otherwise, it is classified as Not eligible. This represents a suggestion for a clear and standardised assessment of the project status’. ")
    #
    # y_position = agregar_texto_descriptivo(c, texto_conclusion , left_margin, y_position, available_width)

    def asignar_puntajes(categorias):
        """
        Asigna puntajes basados en las categorías (Low, Medium, High).

        Parámetros:
        - categorias: Diccionario con las categorías para cada clasificación.

        Retorna:
        - Un diccionario con los puntajes asignados para cada categoría.
        """
        # Definir los puntajes para cada categoría
        puntajes_categoria = {
            "High": 25,
            "Medium": 15,
            "Low": 5,
            "Unknown": 0,  # Si no hay datos, se asigna 0
        }

        # Asignar puntajes basados en las categorías
        puntajes = {
            "Biodiversidad": puntajes_categoria.get(categorias.get("Biodiversity", "Unknown"), 0),
            "Fósforo": puntajes_categoria.get(categorias.get("Phosphorous", "Unknown"), 0),
            "Infiltración": puntajes_categoria.get(categorias.get("Infiltration", "Unknown"), 0),
            "Escorrentía": puntajes_categoria.get(categorias.get("Runoff", "Unknown"), 0),
        }

        return puntajes

    def evaluar_proyecto(categorias):
        """
        Evalúa el proyecto basado en las categorías y asigna puntajes.

        Parámetros:
        - categorias: Diccionario con las categorías para cada clasificación.

        Retorna:
        - Un DataFrame con la tabla de puntajes.
        - Un mensaje indicando si el proyecto es viable o no.
        """
        # Asignar puntajes basados en las categorías
        puntajes = asignar_puntajes(categorias)

        # Definir los máximos puntajes para cada categoría
        max_puntajes = {
            "Potential to generate water security outcomes": 50,  # Infiltración + Escorrentía
            "Potential to generate biodiversity and socio-economic outcomes": 25,  # Biodiversidad
            "Potential to generate Nutrient Neutrality offsets": 25,  # Fósforo
        }

        # Calcular los puntajes para cada categoría
        puntajes_calculados = {
            "Potential to generate water security outcomes": (
                    puntajes.get("Infiltración", 0) + puntajes.get("Escorrentía", 0)
            ),
            "Potential to generate biodiversity and socio-economic outcomes": (
                puntajes.get("Biodiversidad", 0)
            ),
            "Potential to generate Nutrient Neutrality offsets": (
                puntajes.get("Fósforo", 0)
            ),
        }

        # Estandarizar los puntajes para que no excedan el máximo
        for categoria, puntaje in puntajes_calculados.items():
            max_puntaje = max_puntajes[categoria]
            if puntaje > max_puntaje:
                puntajes_calculados[categoria] = max_puntaje

        # Calcular los porcentajes
        porcentajes = {
            categoria: (puntaje / max_puntajes[categoria]) * 100
            for categoria, puntaje in puntajes_calculados.items()
        }

        # Calcular el puntaje total del proyecto
        project_score = sum(puntajes_calculados.values())

        # Determinar si el proyecto es viable
        min_requirement = 50
        estado_proyecto = (
            "Proceed to feasibility" if project_score >= min_requirement else "Not eligible"
        )

        # Crear la tabla
        # tabla = {
        #     "Score": list(puntajes_calculados.values()) + [project_score, min_requirement],
        #     "Project": list(puntajes_calculados.keys()) + ["Project score", "Min. requirement for project selection"],
        #     "Max": list(max_puntajes.values()) + [100, None],
        #     "%": list(porcentajes.values()) + [(project_score / 100) * 100, None
        #     ],
        # }
        tabla = {
            "Score": list(puntajes_calculados.values()) + [project_score, min_requirement],
            "Project": list(puntajes_calculados.keys()) + ["Project score", "Min. requirement for project selection"],
            "Max": list(max_puntajes.values()) + [100, "not apply"],  # Reemplazar None por "not apply"
            "%": list(porcentajes.values()) + [(project_score / 100) * 100, "not apply"],
            # Reemplazar None por "not apply"
        }
        # Convertir a DataFrame
        df_tabla = pd.DataFrame(tabla)

        df_tabla = df_tabla.rename(columns={"Project": "Items"})
        df_tabla= df_tabla[["Items", "Score", "Max", "%"]]

        # Renombrar las columnas para que coincidan con lo solicitado
        df_tabla.columns = ["Items", "Score Project", "MAX", "%"]
        return df_tabla, estado_proyecto

    # Evaluar el proyecto
    # tabla_resultado, estado = evaluar_proyecto(categorias)

    def agregar_tabla_final(dataframe, title, flag_title=True):
        nonlocal y_position
        c.setFont("Helvetica-Bold", 14)

        # Función para formatear números con comas como separadores de miles
        def format_number(num):
            try:
                # Intentar formatear el número con comas y dos decimales
                return "{:,.2f}".format(float(num))
            except ValueError:
                # Si no es un número, devolver el valor original
                return num

        # Crear tabla
        dataframe = dataframe.round(2)
        data = [dataframe.columns.tolist()] + dataframe.applymap(format_number).values.tolist()

        # Calcular el ancho de la primera columna (Items) basado en el texto más largo
        max_text_width = max(
            [len(str(data[row][0])) for row in range(len(data))]
        ) * 7  # Ajustar el factor multiplicativo según el tamaño de la fuente

        # Asegurarse de que el ancho de la primera columna no exceda el 50% del ancho disponible
        max_text_width = min(max_text_width, available_width * 0.5)

        # Calcular el ancho de las demás columnas
        num_columns = len(data[0])
        remaining_width = available_width - max_text_width
        other_col_widths = [remaining_width / (num_columns - 1)] * (num_columns - 1)

        # Combinar los anchos de las columnas
        col_widths = [max_text_width] + other_col_widths

        # Crear la tabla con los anchos de columna calculados
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('WORDWRAP', (0, 0), (0, -1), 'CENTER'),  # Ajustar texto en la primera columna
        ]))

        # Calcular altura de la tabla
        table.wrapOn(c, available_width, y_position)
        table_height = table._height

        # Verificar si hay espacio suficiente para título + tabla
        if y_position - table_height - 20 < 50:
            c.showPage()
            y_position = page_height - 50

        # Agregar título y tabla
        if flag_title:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(left_margin, y_position, title)
            y_position -= 20
        table.drawOn(c, left_margin, y_position - table_height)
        y_position -= table_height + 20

    # agregar_tabla_final(tabla_resultado,'', flag_title=False)
    #
    # texto_conclusion= (f"Suggested conclusion :{estado}")
    #
    # y_position = agregar_texto_descriptivo(c, texto_conclusion , left_margin, y_position, available_width,"Helvetica-Bold")





    c.showPage()
    page_width, page_height = letter  # Dimensiones de la página en orientación vertical
    y_position = page_height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, "Appendix – data Sources")
    y_position -= 20

    # Datos de la tabla
    data = [
        ["Land Cover Map 2023", "UK Centre for Ecology and Hydrology",
         "https://catalogue.ceh.ac.uk/documents/73ecb85e-c55a-4505-9c39-526b464e1efd"],
        ["Habitat Networks (England)", "Natural England",
         "https://www.data.gov.uk/dataset/0ef2ed26-2f04-4e0f-9493-ffbdbfaeb159/habitat-networks-england"],
        ["National LiDAR Programme", "Environment Agency",
         "https://www.data.gov.uk/dataset/f0db0249-f17b-4036-9e65-309148c97ce4/national-lidar-programme"],
        ["BGS Geology - 625k Bedrock version 5", "British Geological Survey",
         "https://www.data.gov.uk/dataset/0867de44-a9d1-4459-84af-87016ae31e26/bgs-geology-625k-digmapgb-625-bedrock-version-5"],
        ["BGS Geology - 625k Superficial version 4", "British Geological Survey",
         "https://www.data.gov.uk/dataset/9bec0be4-c552-4769-b6dc-4073d1de9f46/bgs-geology-625k-digmapgb-625-superficial-version-4"],
        ["National Soil Parent Material", "British Geological Survey",
         "https://www.data.gov.uk/dataset/b5ceb96b-2828-4cca-a410-ad516ccc3fb3/national-soil-parent-material"]
    ]

    # Crear el DataFrame
    df_apendix = pd.DataFrame(data, columns=["Dataset Name (in order of use)", "Provider", "Link"])

    def agregar_tabla_larga_adjust(c, dataframe, title, left_margin, y_position, available_width, page_height,
                            flag_title=True):
        """
        Agrega una tabla con mucho texto a un PDF, ajustando el ancho de las columnas y dividiendo el texto en varias líneas.

        Parámetros:
        - c: Objeto canvas de ReportLab.
        - dataframe: DataFrame con los datos.
        - title: Título de la tabla.
        - left_margin: Margen izquierdo.
        - y_position: Posición vertical actual.
        - available_width: Ancho disponible para la tabla.
        - page_height: Altura de la página.
        - flag_title: Si es True, agrega un título a la tabla.
        """

        try:
            # Estilos para el texto
            styles = getSampleStyleSheet()
            style_normal = styles['Normal']
            style_normal.fontSize = 8  # Reducir el tamaño de la fuente
            style_normal.leading = 10  # Espaciado entre líneas

            # Convertir el DataFrame en una lista de listas (incluyendo los encabezados)
            data = [dataframe.columns.tolist()] + dataframe.values.tolist()

            # Ajustar el ancho de las columnas según el contenido
            col_widths = [
                available_width * 0.35,  # Dataset Name (35% del ancho disponible)
                available_width * 0.25,  # Provider (25% del ancho disponible)
                available_width * 0.40  # Link (40% del ancho disponible)
            ]

            # Crear una lista de listas con Paragraphs para manejar texto largo
            table_data = []
            for row in data:
                table_row = []
                for item in row:
                    # Usar Paragraph para manejar texto largo y dividirlo en varias líneas
                    table_row.append(Paragraph(str(item), style_normal))
                table_data.append(table_row)

            # Crear la tabla
            table = Table(table_data, colWidths=col_widths,
                          repeatRows=1)  # repeatRows=1 para repetir encabezados en cada página
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Fondo gris para la fila de encabezados
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Texto blanco para la fila de encabezados
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Alinear el texto a la izquierda
                ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Agregar bordes a la tabla
                ('FONTSIZE', (0, 0), (-1, -1), 8),  # Tamaño de fuente
                ('VALIGN', (0, 0), (-1, -1), 'TOP')  # Alinear el texto en la parte superior de la celda
            ]))

            # Calcular la altura de la tabla
            table.wrapOn(c, available_width, y_position)
            table_height = table._height

            # Verificar si hay espacio suficiente para el título y la tabla
            if y_position - table_height - 20 < 50:  # 50 es el margen inferior
                c.showPage()  # Crear una nueva página
                y_position = page_height - 50  # Reiniciar la posición vertical

            # Agregar el título si es necesario
            if flag_title:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(left_margin, y_position, title)
                y_position -= 20  # Ajustar la posición vertical después del título

            # Dibujar la tabla en el PDF
            table.drawOn(c, left_margin, y_position - table_height)
            y_position -= table_height + 25  # Ajustar la posición vertical después de la tabla

            return y_position  # Devolver la nueva posición vertical

        except Exception as e:
            print(f"Error al agregar la tabla: {e}")
            return y_position
    y_position = agregar_tabla_larga_adjust(c, df_apendix, "", left_margin, y_position, available_width, page_height,
                                     flag_title=False)

    c.save()
    print("PDF generated successfully.")

import sys
import os

def obtener_ruta_data1(nombre_archivo):
    if getattr(sys, 'frozen', False):
        # Para PyInstaller
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        # Para cx_Freeze
        else:
            base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, "Data", nombre_archivo)
    
def obtener_ruta_data(*args):
    """Construye la ruta a un archivo dentro de los datos."""
    return os.path.join(data_dir, *args)
    
    
def process_information():
    lulc_diccionario = {1: 'Broadleaved woodland', 2: 'Coniferous woodland', 3: 'Arable and horticulture'
        , 4: 'Improved grassland', 5: 'Neutral grassland', 6: 'Calcareous grassland',
                        7: 'Acid grassland', 8: 'Fen, Marsh and Swamp', 9: 'Heather',
                        10: 'Heather grassland', 11: 'Bog', 12: 'Inland rock', 13: 'Saltwater',
                        14: 'Freshwater', 15: 'Supra-littoral rock', 16: 'Supra-littoral sediment',
                        17: 'Littoral rock', 18: 'Littoral sediment', 19: 'Saltmarsh',
                        20: 'Urban', 21: 'Suburban'}

    soil_group_dict = {
        "LIGHT; LIGHT(SILTY); LIGHT(SANDY); LIGHT(SANDY) TO MEDIUM(SANDY)": "LIGHTEST SOILS",
        "LIGHT(SILTY) TO MEDIUM(SILTY) TO HEAVY; LIGHT TO MEDIUM; LIGHT(SILTY) TO MEDIUM(SILTY); MEDIUM TO LIGHT; MEDIUM TO LIGHT(SILTY)": "MEDIUM AND/TO LIGHT",
        "LIGHT(SANDY) TO MEDIUM(SANDY) TO HEAVY AND MEDIUM; MEDIUM(SILTY) TO LIGHT(SILTY); MEDIUM(SILTY); MEDIUM": "MEDIUM SOILS",
        "MEDIUM TO HEAVY; MEDIUM TO LIGHT(SILTY) TO HEAVY": "MEDIUM AND/TO HEAVY",
        "MEDIUM(SILTY) TO HEAVY; HEAVY AND MEDIUM TO LIGHT(SILTY); HEAVY TO MEDIUM(SANDY) TO LIGHT(SANDY)": "HEAVIEST SOILS",
        "ALL": "MIXED or ORGANIC",
        "NA": "NA"}





    if shapefile_path.get() or (x_coord.get() and y_coord.get() and radius_km.get()):

        try:

            name_project = project_name.get()

            os.makedirs('Project_' + name_project, exist_ok=True)

            for folder in ['FIGURES','RESULTS','SHAPEFILES']:
                os.makedirs(r'Project_' + name_project+'/'+folder, exist_ok=True)



            if shapefile_path.get():
                polygon_path = shapefile_path.get()

                # Leer el shapefile
                gdf = gpd.read_file(polygon_path)

                # Verificar el CRS
                if gdf.crs != "EPSG:27700":
                    # Crear la carpeta para almacenar el shapefile reproyectado
                    new_path_shp = os.path.join('Project_' + name_project, 'SHAPEFILES')
                    #os.makedirs(new_path_shp, exist_ok=True)
                    #new_path_shp = os.path.join(os.path.expanduser('~'), 'Project_' + name_project, 'SHAPEFILES')
                    
                    
                    # Definir la ruta del nuevo shapefile reproyectado
                    reprojected_shp_path = os.path.join(new_path_shp, 'reprojected_shapefile.shp')

                    # Reproyectar a EPSG:27700
                    gdf = gdf.to_crs("EPSG:27700")

                    # Guardar el shapefile reproyectado
                    gdf.to_file(reprojected_shp_path)

                    # Actualizar polygon_path con la ruta del shapefile reproyectado
                    polygon_path = reprojected_shp_path

                #agregar una condicion que lea la capa y sus crs si es diferente a la brutanica que la reproyecte y exporte y asi lea la nueva
            else:
                # Crear el polígono de buffer
                new_path_shp = os.path.join('Project_' + name_project, 'SHAPEFILES') 
                output_shp_path = os.path.join(new_path_shp, "output_buffer.shp") 

                polygon_path = output_shp_path


            #General Inputs


            new_path_fig = 'Project_' + name_project+'/'+'FIGURES'+'/'
            new_path_files = 'Project_' + name_project + '/' + 'RESULTS' + '/'


            
            input_path = obtener_ruta_data("LULC", "gblcm10m2021_NORFOLK.tif")
            habitat_path = obtener_ruta_data("Habitat", "Habitat_Networks_NORFOLK.shp")
            BedRock_path = obtener_ruta_data("Geology", "625k_V5_BEDROCK_Geology_Polygons.shp")
            Superficial_path = obtener_ruta_data("Geology", "UK_625k_SUPERFICIAL_Geology_Polygons.shp")
            Soil_path = obtener_ruta_data("Soil", "SoilParentMateriall_V1_portal1km_clip.shp")
            raster_slope = obtener_ruta_data("Slope", "slope_Norfolk.tif")
            path_AOI = obtener_ruta_data("NbS", "AOI_Norfolk.shp")
            image_path = new_path_fig+ "Map_Location.png"
            pdf_path = 'Project_' + name_project+'/'+ r'assessment_report_'+name_project+'.pdf'


            message_queue.put('step 1, processing location map and extraction of cover and generalities....')
            generar_imagen_png(polygon_path, image_path, name_project)
            result_lulc = calcular_area_LULC(polygon_path, input_path, lulc_diccionario)
            result_habitat = calcular_area_files_shp(polygon_path, habitat_path, 'Class')
            result_BedRock = calcular_area_files_shp(polygon_path, BedRock_path, 'RCS_D')
            result_Superficial = calcular_area_files_shp(polygon_path, Superficial_path, 'RCS', )
            tem_soil = calcular_area_files_shp(polygon_path, Soil_path, 'SOIL_GROUP', )

            result_slope = calcular_area_slope(polygon_path, raster_slope)
            generalidades = calcular_area_y_subpoligonos(polygon_path)

            results = {"Land Use": result_lulc,
                       "Habitats": result_habitat,
                       'Slope': result_slope,
                       "Bedrock": result_BedRock,
                       "Superficial Geology": result_Superficial,
                       "Soil": tem_soil}

            exportar_diccionario_a_excel(
                diccionario=results,
                nombre_archivo="01_General_features_"+name_project,
                ruta=new_path_files
            )


            shapefile_paths = [
                obtener_ruta_data("NbS_3857","RZ_Opmap_PRIO.shp"),
                obtener_ruta_data("NbS_3857","Wen_CS_Update.shp"),
                obtener_ruta_data("NbS_3857","FZ_Final_merge_FROP.shp")
            ]

            colores = ["darkred", "olivedrab", "steelblue", ]
            transparencias = [0.7, 0.7, 0.7, ]

            NBS_01 =new_path_fig +"Map_nbs01.png"
            console_output.write('Step 2, processing NbS 01 map, extracting area values...')
            generar_imagen_con_capas(polygon_path, shapefile_paths, NBS_01, colores, transparencias)

            nbs_result = extract_area_nbs_I(polygon_path)


            shapefile_paths_s = [
                obtener_ruta_data("NbS_3857","LUC_10-perc.shp"),
                obtener_ruta_data("NbS_3857","SPS_Final.shp"),
                obtener_ruta_data("NbS_3857","SM_Opmap_PRIO.shp")
            ]

            colores_s = ['green', 'teal',
                         'darkgoldenrod', ]  # Azul para las líneas, amarillo para la segunda capa, verde para la tercera
            transparencias_s = [0.5, 0.7, 0.7]

            NBS_02 = new_path_fig +"Map_nbs02.png"
            console_output.write('Step 3, processing NbS 02 map, extracting area values...')
            generar_imagen_con_capas_transparecia(polygon_path, shapefile_paths_s, NBS_02, colores_s, transparencias_s)

            Imagenes = {'Location': image_path,
                        'NBS_01': NBS_01,
                        'NBS_02': NBS_02}

            nbs_result_2 = extract_area_nbs_LUC(polygon_path)

            resultsNBS = {"NBS_01": nbs_result,
                          "NBS_02": nbs_result_2}

            exportar_diccionario_a_excel(
                diccionario=resultsNBS,
                nombre_archivo="02_NbS_"+name_project,
                ruta=new_path_files
            )

            #Beneficios NBS

            console_output.write('Step 4, processing biodiversity benefits...')

            shapefile_paths  = [
                obtener_ruta_data("NbS","FZ_Final_merge_FROP.shp"),
                obtener_ruta_data("NbS","RZ_Opmap_PRIO.shp"),
                obtener_ruta_data("NbS","Wen_CS_Update.shp")
            ]


            raster_BNG = obtener_ruta_data("Benefit","Distributed_BNG_uplift.tif") #'Data/Benefit/Distributed_BNG_uplift.tif'
            resul_bng = procesar_nbs_y_raster_individual(polygon_path, shapefile_paths, raster_BNG,
                                                         'BNG')  # Funciona bien



            shapefile_paths_NN = [
                obtener_ruta_data("NbS","FZ_Final_merge_FROP.shp"),
                obtener_ruta_data("NbS","RZ_Opmap_PRIO.shp"),
                obtener_ruta_data("NbS","SM_Opmap_PRIO.shp"),
                obtener_ruta_data("Benefit","cRAF_Nutrient_uplift.shp"),  # Capa de puntos
                obtener_ruta_data("Benefit","sRAF_Nutrient_uplift.shp")  # Capa de puntos
            ]

            console_output.write('Step 5, processing Nutrient Neutrality benefits...')


            raster_N =obtener_ruta_data("Benefit","Distributed_Nexp_uplift.tif") # 'Data/Benefit/Distributed_Nexp_uplift.tif'

            resul_N = procesar_nbs_y_raster_individual(polygon_path, shapefile_paths_NN, raster_N,
                                                       'Nitrogen')  # Funciona bien

            raster_P = obtener_ruta_data("Benefit","Distributed_Pexp_uplift.tif") #'Data/Benefit/Distributed_Pexp_uplift.tif'

            resul_P = procesar_nbs_y_raster_individual(polygon_path, shapefile_paths_NN, raster_P,
                                                       'Phosphorus')  # Funciona bien

            df_combinado = pd.merge(resul_N, resul_P, on='NBS', how='outer')
            df_combinado.loc[df_combinado['NBS'] != 'Runoff Attenuation Features', df_combinado.select_dtypes(include=['number']).columns] *= -1


            shapefile_paths_RF = [
                obtener_ruta_data("NbS","FZ_Final_merge_FROP.shp"),
                obtener_ruta_data("NbS","RZ_Opmap_PRIO.shp"),
                obtener_ruta_data("NbS","SM_Opmap_PRIO.shp"),
                obtener_ruta_data("Benefit","cRAF_Nutrient_uplift.shp"),  # Capa de puntos
                obtener_ruta_data("Benefit","sRAF_Nutrient_uplift.shp")  # Capa de puntos
            ]

            console_output.write('Step 6, processing Flooding and Runoff benefits...')
            raster_RF = obtener_ruta_data("Benefit","Distributed_runoff_uplift.tif") #'Data/Benefit/Distributed_runoff_uplift.tif'

            resul_RF = procesar_nbs_y_raster_individual(polygon_path, shapefile_paths_RF, raster_RF,'Runoff')  # Funciona bien
            resul_RF.loc[resul_RF['NBS'] != 'Runoff Attenuation Features', resul_RF.select_dtypes(include=['number']).columns] *= -1


            console_output.write('Step 7, processing Infiltration to Groundwater benefits...')
            raster_Rg =obtener_ruta_data("Benefit","Distributed_recharge_uplift.tif") # 'Data/Benefit/Distributed_recharge_uplift.tif'

            resul_Rg = procesar_nbs_y_raster_individual(polygon_path, shapefile_paths_RF, raster_Rg,
                                                        'Recharge')  # Funciona bien

            shapefile_paths_points = [
                obtener_ruta_data("Benefit","cRAF_Nutrient_uplift.shp"),  # Capa de puntos
                obtener_ruta_data("Benefit","sRAF_Nutrient_uplift.shp")  # Capa de puntos
            ]
            text_points = contar_puntos_intersectados(polygon_path, shapefile_paths_points)


            Result_oportunidad = {'BNG': resul_bng,
                                  'NN': df_combinado,
                                  'runoff': resul_RF,
                                  'recharge': resul_Rg}

            exportar_diccionario_a_excel(
                diccionario=Result_oportunidad ,
                nombre_archivo="03_NbS_Benefits_"+name_project,
                ruta=new_path_files
            )


            area_total = calcular_area_y_subpoligonos(polygon_path)[0]  # El primer valor es el área en hectáreas
            totales_por_area= calcular_totales_por_area(resul_bng, df_combinado, resul_RF, resul_Rg, area_total)

            nombre_cuenca = determinar_cuenca(polygon_path, path_AOI)

            categorias, val_ref = determinar_categorias(totales_por_area, nombre_cuenca)

            generate_pdf_with_results(
                pdf_path=pdf_path,
                images=Imagenes,
                name_project=name_project,
                general=generalidades,
                results=results,
                resultB=resultsNBS,
                result_Benefit=Result_oportunidad,
                categorias=categorias,
                nombre_cuenca=nombre_cuenca,
                valores_referencia=val_ref,
                totales_project=totales_por_area,
                text_points=text_points
            )

            # generate_pdf(pdf_path)
            messagebox.showinfo("Success", f"PDF generated successfully at {pdf_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF: {str(e)}")
    else:
        messagebox.showwarning("Input Error", "Please provide a shapefile or coordinates and radius.")



def browse_shapefile():
    file_path = filedialog.askopenfilename(filetypes=[("Shapefiles", "*.shp")])
    if file_path:
        shapefile_path.set(file_path)


def process_buffer():
    
    if (project_name.get() and x_coord.get() and y_coord.get() and radius_km.get()):

        try:

            name_project = project_name.get()

            os.makedirs('Project_' + name_project, exist_ok=True)
            
            for folder in ['FIGURES','RESULTS','SHAPEFILES']:
                os.makedirs(r'Project_' + name_project+'/'+folder, exist_ok=True)
                 
            new_path_shp = os.path.join('Project_' + project_name.get(), 'SHAPEFILES')
            output_path = os.path.join(new_path_shp, "output_buffer.shp")
            
            create_buffer_from_coords(x_coord.get(),y_coord.get(),float(radius_km.get()),  output_path)
                 
            
            messagebox.showinfo("Success", f"Shapefile generated successfully at {new_path_shp}")
        
        except Exception as e:
            messagebox.showwarning("Error in process", "Please provide a shapefile or coordinates and radius.")
    else:
        messagebox.showwarning("Input Error", "Please provide a Name project or shapefile or coordinates and radius.")
        
    
    


def start_process():

    thread = threading.Thread(target=process_information)
    thread.daemon = True  # Permite que el hilo se cierre al cerrar la aplicación
    thread.start()
    
def start_process_Buffer():

    thread = threading.Thread(target=process_buffer)
    thread.daemon = True  # Permite que el hilo se cierre al cerrar la aplicación
    thread.start()





# Main application window
root = tk.Tk()
root.title("NbS for water security prioritisation tool")
root.geometry("650x700")

# Variables
shapefile_path = tk.StringVar()
x_coord = tk.StringVar()
y_coord = tk.StringVar()
radius_km = tk.StringVar()
project_name = tk.StringVar()

# Fuente en negrita
bold_font = tkFont.Font(family="Helvetica", size=12, weight="bold")

# Título en negrita
title_label = tk.Label(root, text="NbS for water security prioritisation tool", font=bold_font)
title_label.pack(pady=10)

# Frame para la carga del archivo .shp
shp_frame = tk.Frame(root)
shp_frame.pack(pady=10, anchor="w")

# Campos de entrada para la carga del archivo .shp
tk.Label(shp_frame, text="Option 1. upload the shapefile .shp polygon of your project:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
shapefile_entry = tk.Entry(shp_frame, textvariable=shapefile_path, width=40)
shapefile_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
browse_button = tk.Button(shp_frame, text="Browse", command=browse_shapefile)
browse_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

# Frame para el nombre del proyecto
project_frame = tk.Frame(root)
project_frame.pack(pady=10, anchor="w")

# Campo de entrada para el nombre del proyecto
tk.Label(project_frame, text="Project Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
project_entry = tk.Entry(project_frame, textvariable=project_name, width=40)
project_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# Frame para coordenadas y radio
coord_frame = tk.Frame(root)
coord_frame.pack(pady=10, anchor="w")

# Campos de entrada para coordenadas y radio
tk.Label(coord_frame, text="Option 2.  X (Meters):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
coord_entry_x = tk.Entry(coord_frame, textvariable=x_coord, width=10)
coord_entry_x.grid(row=0, column=1, padx=5, pady=5, sticky="w")

tk.Label(coord_frame, text="Y (Meters):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
coord_entry_y = tk.Entry(coord_frame, textvariable=y_coord, width=10)
coord_entry_y.grid(row=0, column=3, padx=5, pady=5, sticky="w")

tk.Label(coord_frame, text="Radius (kilometers):").grid(row=0, column=4, padx=5, pady=5, sticky="w")
radius_entry = tk.Entry(coord_frame, textvariable=radius_km, width=10)
radius_entry.grid(row=0, column=5, padx=5, pady=5, sticky="w")

# Botón "Crear capa"



# Crear nombre del archivo con ruta completa

create_layer_button = tk.Button(coord_frame,text="Create shapefile", command=start_process_Buffer)

#create_layer_button = tk.Button(coord_frame, text="Create shapefile", command=lambda: create_buffer_from_coords(x_coord.get(), y_coord.get(), float(radius_km.get()), new_path_shp + '/'+"output_buffer.shp"))

create_layer_button.grid(row=0, column=6, padx=5, pady=5, sticky="w")

# Console output
console_output = ConsoleOutput(root, wrap=tk.WORD, width=70, height=10)
console_output.pack(pady=10)
# Iniciar el procesamiento de la cola
console_output.process_queue()

process_button = tk.Button(root, text="Start Process", command=start_process)
process_button.pack(pady=20)



# Frame para los logos
logo_frame = tk.Frame(root)
logo_frame.pack(pady=10, anchor="center")

# Cargar imágenes de los logos (asegúrate de tener los archivos en la misma carpeta)
try:
    # Guardar las imágenes como atributos de la ventana para evitar que se eliminenobtener_ruta_data("Benefit/Distributed_recharge_uplift.tif")
    file_1 = obtener_ruta_data("logos", "N4W Logo_Tagline Logo Colour.png")
    file_2 = obtener_ruta_data("logos", "Norfolk.png")
    root.logo1 = tk.PhotoImage(file=file_1).subsample(7, 7)  # Ajusta los valores según sea necesario tk.PhotoImage(file="logos/N4W Logo_Tagline Logo Colour.png").subsample(7, 7)
    root.logo2 = tk.PhotoImage(file=file_2).subsample(1, 1)
    # root.logo3 = tk.PhotoImage(file="logos/TNCLogoPrimary_RGB.png").subsample(11, 11)

    # Mostrar los logos
    tk.Label(logo_frame, image=root.logo1).grid(row=0, column=1, padx=10, pady=10)  # Logo 1 en el centro
    tk.Label(logo_frame, image=root.logo2).grid(row=1, column=1, padx=10, pady=10)  # Logo 2 a la derecha abajo del 1
    # tk.Label(logo_frame, image=root.logo3).grid(row=1, column=0, padx=10, pady=10)  # Logo 3 a la izquierda abajo del 1
except tk.TclError as e:
    console_output.write(f"Error al cargar los logos: {str(e)}\n")
    messagebox.showerror("Error", f"Error al cargar los logos: {str(e)}")

# Redirect stdout to the console output
import sys


# console_output.write('Hello, This is the Prioritisation tool, do not close the window once the button has been clicked: Start Process ')
message_queue.put ('Hello, This is the Prioritisation tool, do not close the window once the button has been clicked: Start Process ')
sys.stdout = console_output

# Run the application
root.mainloop()
