import os
import sys
import numpy as np
import tensorflow as tf

# Asegura que TGNN_PF está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TGNN_PF')))

from Get_Data import DataImporter

class Args:
    def __init__(self):
        self.dataset = "Case30Net"
        self.train_size = 0.8
        self.batch_size = 4
        self.shuffle = True
        self.normalize = True
        self.test_type = "PF"
        self.seed = 42
        self.gpu = False
        self.device = "cpu"

def test_data_importer_structure_and_noise_generation():
    args = Args()
    data = DataImporter(args)

    # Comprobaciones básicas
    assert isinstance(data.input_data, dict), "input_data no es un diccionario"
    assert "bus" in data.input_data, "Falta la clave 'bus'"
    assert "branch" in data.input_data, "Falta la clave 'branch'"
    assert "gen" in data.input_data, "Falta la clave 'gen'"

    assert hasattr(data, 'n_samples'), "DataImporter no define 'n_samples'"
    assert isinstance(data.n_samples, tf.Variable), "'n_samples' no es una variable de TensorFlow"

    print("✅ DataImporter estructuralmente correcto y 'n_samples' definido.")

if __name__ == "__main__":
    test_data_importer_structure_and_noise_generation()