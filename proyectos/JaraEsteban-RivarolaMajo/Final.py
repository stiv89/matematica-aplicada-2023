# Proyecto final
# Esteban Manuel Jara Noceda
# Maria Jose Rivarola Orihuela

# Importamos las librerías necesarias para el procesamiento de datos, cálculo de similitudes, lógica difusa y presentación de datos.
import pandas as pd
import pandas as pd
import gzip
import json
from sklearn.metrics.pairwise import cosine_similarity
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import argparse
from prettytable import PrettyTable

# Función para cargar datos desde un archivo comprimido con gzip y convertirlos a un DataFrame de pandas.
def cargarDatosG(path):
    try:
        with gzip.open(path, 'rb') as file:
            return pd.DataFrame([json.loads(line) for line in file])
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

# Función para calcular la calificación promedio de cada producto basándose en las reseñas.
def getcalificacionProm(reviews):
    return reviews.groupby('asin')['overall'].mean().to_dict()

# Función para crear perfiles de usuario basados en los productos que han revisado.
def crearPerfiles(reviews):
    unicos_asins = reviews['asin'].unique()
    perfiles = []
    for idUsuario, group in reviews.groupby('reviewerID'):
        profile = pd.Series(0, index=unicos_asins)
        profile[group['asin']] = 1
        perfiles.append(profile)
    return perfiles

# Función para calcular la similitud de los ítems utilizando la similitud de coseno entre los perfiles de usuario.
def getitemsSimilitud(perfiles):
    return cosine_similarity(np.array(perfiles))

# Función para mapear índices a ASINs y viceversa para facilitar el acceso a los datos.
def mapindicesAsin(reviews):
    unicos_asins = reviews['asin'].unique()
    return dict(enumerate(unicos_asins)), {v: k for k, v in enumerate(unicos_asins)}

# Función para encontrar los ítems que un usuario ha revisado.
def getitemsRevisados(user_index, perfiles):
    return [i for i, reviewed in enumerate(perfiles[user_index]) if reviewed]

# Función para encontrar ítems relacionados con un ítem específico.
def getrelacionadosItems(metadata, asin_index_map, item_index, related_key):
    related_asins = metadata.loc[metadata['asin'] == asin_index_map[item_index], related_key]
    return related_asins.explode().dropna().unique()

# Función para recomendar ítems a un usuario basándose en los ítems que ha revisado y la similitud entre ítems.
def itemsRecomendar(metadata, similarity_matrix, reviewed_items, asin_index_map, index_asin_map):
    recomendaciones = []
    for item_index in reviewed_items:
        for related_asin in getrelacionadosItems(metadata, index_asin_map, item_index, 'also_view'):
            if related_asin not in asin_index_map or related_asin in reviewed_items:
                continue
            related_index = asin_index_map[related_asin]
            similarity_score = similarity_matrix[item_index][related_index]
            avg_rating = metadata.loc[metadata['asin'] == related_asin, 'average_rating'].iloc[0]
            recomendaciones.append((related_asin, avg_rating, similarity_score))

    return recomendaciones

# Función para configurar el sistema de recomendación utilizando lógica difusa.
def sistemaRecomLog():
    # Definimos variables difusas para las entradas y la salida del sistema.
    rating = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'rating')
    similarity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'similarity')
    recommendation = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'recommendation')
    # Automáticamente generamos funciones de membresía.
    rating.automf(3)
    similarity.automf(3)

    # Definimos funciones de membresía para la salida.
    recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 0.5])
    recommendation['medium'] = fuzz.trimf(recommendation.universe, [0, 0.5, 1])
    recommendation['high'] = fuzz.trimf(recommendation.universe, [0.5, 1, 1])

    # Establecemos reglas difusas para determinar la recomendación.
    rule1 = ctrl.Rule(rating['poor'] | similarity['poor'], recommendation['low'])
    rule2 = ctrl.Rule(rating['average'], recommendation['medium'])
    rule3 = ctrl.Rule(rating['good'] | similarity['good'], recommendation['high'])
    
    # Creamos y retornamos el sistema de control difuso.
    recommendation_system = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(recommendation_system)

# Función principal para ejecutar el sistema de recomendación.
def ejecsistRecom(idUsuario, reviewd, metadatad):
    # Cargamos los datos de las reseñas y los metadatos.
    reviews = cargarDatosG(reviewd)
    metadata = cargarDatosG(metadatad)

    # Calculamos la calificación promedio de los productos.
    average_ratings = getcalificacionProm(reviews)
    metadata['average_rating'] = metadata['asin'].map(average_ratings)

    # Creamos perfiles de usuario y calculamos la similitud de los ítems.
    user_perfiles = crearPerfiles(reviews)
    item_similarities = getitemsSimilitud(user_perfiles)

    # Mapeamos índices a ASINs y encontramos los ítems que el usuario ha revisado.
    index_asin_map, asin_index_map = mapindicesAsin(reviews)
    reviewed_items = getitemsRevisados(idUsuario, user_perfiles)

    # Obtenemos recomendaciones para el usuario y configuramos el sistema de recomendación difuso.
    recomendaciones = itemsRecomendar(
        metadata, item_similarities, reviewed_items, asin_index_map, index_asin_map
    )

    # Inicializamos el sistema de recomendación difuso y preparamos la tabla para la salida.
    fuzzy_system = sistemaRecomLog()
    table = PrettyTable()
    table.field_names = ["Product Name", "ASIN", "Average Rating", "Similarity", "Fuzzy Recommendation"]
    
    # Procesamos cada recomendación a través del sistema difuso y agregamos los resultados a la tabla.
    for asin, rating, similarity in recomendaciones:
        product_name = metadata.loc[metadata['asin'] == asin, 'title'].iloc[0] if not metadata.loc[metadata['asin'] == asin, 'title'].empty else "Unknown Product"
        fuzzy_system.input['rating'] = rating
        fuzzy_system.input['similarity'] = similarity
        fuzzy_system.compute()
        recommendation_value = fuzzy_system.output['recommendation']
        table.add_row([product_name, asin, round(rating, 2), round(similarity, 2), round(recommendation_value, 2)])
    
    # Imprimimos la tabla con las recomendaciones.
    print(table)

# Configuramos los parámetros y ejecutamos la función principal.
idUsuario = 1  # Índice del usuario para el cual queremos obtener recomendaciones.
reviewd = r'C:\Users\ejara\Desktop\Matematica Aplicada\TP\Software.json.gz'  # Ruta al archivo de reseñas.
metadatad = r'C:\Users\ejara\Desktop\Matematica Aplicada\TP\meta_Software.json.gz'  # Ruta al archivo de metadatos.

# Llamamos a la función principal con los parámetros definidos.
ejecsistRecom(idUsuario, reviewd, metadatad)