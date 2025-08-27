from flask import Blueprint, request, jsonify
import numpy as np
from quantum_tools import simulate_circuit, get_single_qubit_dm, dm_to_bloch
from visualizer import bloch_plot

bloch_api = Blueprint('bloch_api', __name__)

@bloch_api.route('/bloch3d_html', methods=['GET'])
def bloch3d_html():
    try:
        idx = int(request.args.get('qubit', 0))
    except Exception:
        idx = 0
    from app import _get_qc
    qc = _get_qc()
    rho = simulate_circuit(qc)
    bloch = dm_to_bloch(get_single_qubit_dm(rho, idx, qc.num_qubits))
    fig = bloch_plot(bloch, title=f"Qubit {idx} Bloch Sphere")
    return fig.to_html(full_html=True, include_plotlyjs='cdn')

@bloch_api.route('/bloch3d', methods=['GET'])
def bloch3d():
    # Get qubit index from query param
    try:
        idx = int(request.args.get('qubit', 0))
    except Exception:
        idx = 0
    from app import _get_qc
    qc = _get_qc()
    rho = simulate_circuit(qc)
    bloch = dm_to_bloch(get_single_qubit_dm(rho, idx, qc.num_qubits))
    fig = bloch_plot(bloch, title=f"Qubit {idx} Bloch Sphere")
    return jsonify(fig.to_dict())
