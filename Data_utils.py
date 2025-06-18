# Data_utils.py
import tensorflow as tf

def create_batch(data_list):
    """
    Convierte una lista de diccionarios de casos en un batch con tensores.
    Cada entrada del diccionario debe tener las claves esperadas por el modelo.
    """
    # Creamos un batch vacío
    batch = {}

    # Recorremos las claves del primer diccionario para construir el batch
    for key in data_list[0]:
        # Obtenemos todos los valores correspondientes a la clave en cada muestra
        values = [sample[key] for sample in data_list]

        # Convertimos a tensor, detectando si son enteros o flotantes
        if isinstance(values[0], int):
            batch[key] = tf.convert_to_tensor(values, dtype=tf.int32)
        elif isinstance(values[0], float):
            batch[key] = tf.convert_to_tensor(values, dtype=tf.float32)
        elif isinstance(values[0], list) or isinstance(values[0], tuple):
            batch[key] = tf.convert_to_tensor(values, dtype=tf.float32)
        else:
            # Si ya es un array o tensor, se agrupan directamente
            batch[key] = tf.stack(values)

    return batch
