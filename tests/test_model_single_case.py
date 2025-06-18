import os
import sys
import tensorflow as tf
import numpy as np

# Añadir TGNN_PF y Data_files al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TGNN_PF')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_files')))

from Get_Data import DataImporter
from model_PF import TGN_PF

class Args:
    def __init__(self):
        self.dataset = "Case30Net_from_mat"
        self.train_size = 0.8
        self.batch_size = 4
        self.shuffle = True
        self.normalize = True
        self.test_type = "PF"
        self.seed = 42
        self.gpu = False
        self.device = "cpu"
        self.glob_norm = 1.0
        self.max_iter = 5
        self.track_validation = 1
        self.save_path = "results/test_train_run"
        self.resume = True

def test_model_single_case_base():
    args = Args()
    data = DataImporter(args)
    data.load_data()  # ← Añade esta línea

    model = TGN_PF(
        lr=1e-3,
        batch_size=args.batch_size,
        dim_e=10,
        dim_pv=10,
        dim_pq=10,
        time_steps=2,
        tgn_layers=10,
        non_lin='tanh',
        name='TGN_PFsolverSingle',
        directory=args.save_path,
        model_to_restore=None
    )
    model.set_data_importer(data)

    try:
        input_data = data.input_data

        print("🔍 Tipo de input_data:", type(input_data))
        if isinstance(input_data, dict):
            case_base = input_data
        elif isinstance(input_data, (list, tuple)):
            case_base = input_data[0]
        else:
            raise TypeError(f"❌ Tipo inesperado de input_data: {type(input_data)}")

        Vmag_pred, Vang_pred = model.predict_single_case(case_base)

        print("✅ Predicción para un único caso completada.\n")
        for i in range(len(Vmag_pred)):
            print(f"Nodo {i:2d}: Vmag = {Vmag_pred[i]:.4f}, Vang = {Vang_pred[i]:.4f}")

    except Exception as e:
        print(f"❌ Error durante la evaluación de una muestra: {e}")
        assert False, "Falló el test de caso único"

if __name__ == "__main__":
    test_model_single_case_base()
