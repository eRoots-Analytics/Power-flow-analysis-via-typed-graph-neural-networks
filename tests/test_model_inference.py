import os
import sys
import tensorflow as tf
import numpy as np
from scipy.io import loadmat

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

def test_model_against_matpower_solution():
    args = Args()
    data = DataImporter(args)

    model = TGN_PF(
        lr=1e-3,
        batch_size=args.batch_size,
        dim_e=10,
        dim_pv=10,
        dim_pq=10,
        time_steps=2,
        tgn_layers=10,
        non_lin='tanh',
        name='TGN_PFsolverTop',
        directory=args.save_path,
        model_to_restore=None
    )

    model.set_data_importer(data)

    try:
        # Ejecutar test del modelo
        loss, Vmag_pred, Vang_pred, _, _ = model.test()

        # Cargar solución real de MATPOWER desde .mat
        mat_file = os.path.join("..", "matpower_files", "matpower_solutions", "case30_solution.mat")
        mat_data = loadmat(mat_file)
        Vmag_real = mat_data['V_mag'].flatten()
        Vang_real = mat_data['V_ang'].flatten()

        print("✅ Comparación con resultados de MATPOWER (sólo muestra 0):\n")

        print("Magnitud de tensión (predicho vs real):")
        for j in range(len(Vmag_pred[0])):
            pred = float(Vmag_pred[0][j])
            real = float(Vmag_real[j])
            print(f"Nodo {j:2d}: {pred:.4f} vs {real:.4f}")

        print("\nÁngulo de tensión (predicho vs real):")
        for j in range(len(Vang_pred[0])):
            pred = float(Vang_pred[0][j])
            real = float(Vang_real[j])
            print(f"Nodo {j:2d}: {pred:.4f} vs {real:.4f}")

    except Exception as e:
        print(f"❌ Error durante la inferencia o la comparación: {e}")
        assert False, "Falló el test de comparación con solución real"

if __name__ == "__main__":
    test_model_against_matpower_solution()
