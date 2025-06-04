# Crear un test que importa el archivo generado y comprueba la estructura devuelta por ref_grid()
test_file_path = "/mnt/data/test_ref_grid.py"

test_code = '''
import importlib.util
import os
import numpy as np

# Ruta al archivo generado
module_path = "Case30Net_from_mat.py"
module_name = "Case30Net_from_mat"

# Cargar dinámicamente el módulo
spec = importlib.util.spec_from_file_location(module_name, module_path)
case_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case_module)

# Ejecutar ref_grid y validar
def test_ref_grid():
    data = case_module.ref_grid()

    assert isinstance(data, dict), "ref_grid() debe devolver un diccionario"
    for key in ['buses', 'branches', 'generators', 'baseMVA', 'init_V']:
        assert key in data, f"Falta la clave '{key}' en el resultado"

    assert isinstance(data['buses'], np.ndarray)
    assert isinstance(data['branches'], np.ndarray)
    assert isinstance(data['generators'], np.ndarray)
    assert isinstance(data['baseMVA'], (float, int))
    assert isinstance(data['init_V'], dict)
    assert 'mag' in data['init_V'] and 'ang' in data['init_V']

    print("✅ ref_grid() ha pasado todas las comprobaciones.")

if __name__ == "__main__":
    test_ref_grid()
'''

with open(test_file_path, "w") as f:
    f.write(test_code)

test_file_path
