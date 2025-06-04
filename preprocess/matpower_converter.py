# matpower_converter.py

import scipy.io
import numpy as np


def convert_matpower_to_python(mat_file_path, output_py_path):
    """
    Convierte un archivo .mat con estructura MATPOWER a un archivo .py con función ref_grid().

    Args:
        mat_file_path (str): Ruta al archivo .mat (guardado desde MATLAB con 'mpc' como variable).
        output_py_path (str): Ruta del archivo .py de salida con la función ref_grid().
    """
    # Cargar el archivo .mat
    mat = scipy.io.loadmat(mat_file_path, squeeze_me=True)
    mpc_struct = mat["mpc"]

    # Extraer las matrices de buses, ramas, generadores y baseMVA
    buses = mpc_struct["bus"].item()
    branches = mpc_struct["branch"].item()
    generators = mpc_struct["gen"].item()
    baseMVA = mpc_struct["baseMVA"].item() if "baseMVA" in mpc_struct.dtype.names else 100

    # Escribir el archivo Python
    with open(output_py_path, "w") as f:
        f.write("import numpy as np\n\n")
        f.write("def ref_grid():\n")
        f.write(f"    baseMVA = {baseMVA}\n")
        f.write(f"    bus_info = np.array({repr(buses.tolist())})\n")
        f.write(f"    branch_info = np.array({repr(branches.tolist())})\n")
        f.write(f"    gen_info = np.array({repr(generators.tolist())})\n")
        f.write("    return {\n")
        f.write("        'buses': bus_info,\n")
        f.write("        'branches': branch_info,\n")
        f.write("        'generators': gen_info,\n")
        f.write("        'baseMVA': baseMVA,\n")
        f.write("        'init_V': {'mag': np.ones(len(bus_info)), 'ang': np.zeros(len(bus_info))}\n")
        f.write("    }\n")


# uso de la función de conversión
# from preprocess.matpower_converter import convert_matpower_to_python
# convert_matpower_to_python("case30.mat", "Case30Net.py")
