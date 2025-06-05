import os
import sys
import tensorflow as tf

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
        self.resume = False

def test_model_training_runs_successfully():
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

    #jad model.data_importer = data
    model.set_data_importer(data)

    try:
        model.train(
            max_iter=args.max_iter,
            glob_norm=args.glob_norm,
            save_step=args.track_validation
        )
        print("✅ Entrenamiento ejecutado sin errores.")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        assert False, "Falló el entrenamiento del modelo"

if __name__ == "__main__":
    test_model_training_runs_successfully()