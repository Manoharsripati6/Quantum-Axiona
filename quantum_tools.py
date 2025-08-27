import numpy as np
from qiskit_aer import Aer
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Pauli, Statevector
import re

def import_qasm_file(file_path):
    """Import a QASM file and return a QuantumCircuit object."""
    try:
        with open(file_path, 'r') as file:
            qasm_content = file.read()
        return parse_qasm_string(qasm_content)
    except FileNotFoundError:
        raise Exception(f"QASM file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading QASM file: {str(e)}")

def parse_qasm_string(qasm_string):
    """Parse a QASM string and return a QuantumCircuit object."""
    try:
        # Remove comments and empty lines
        qasm_lines = []
        for line in qasm_string.split('\n'):
            line = line.strip()
            if line and not line.startswith('//'):
                qasm_lines.append(line)
        
        # Parse header
        num_qubits = 0
        num_clbits = 0
        
        for line in qasm_lines:
            if line.startswith('OPENQASM'):
                continue
            elif line.startswith('include'):
                continue
            elif line.startswith('qreg'):
                match = re.match(r'qreg\s+(\w+)\[(\d+)\]', line)
                if match:
                    num_qubits = int(match.group(2))
            elif line.startswith('creg'):
                match = re.match(r'creg\s+(\w+)\[(\d+)\]', line)
                if match:
                    num_clbits = int(match.group(2))
            elif line.startswith('barrier'):
                continue
            elif line.startswith('measure'):
                continue
            elif line.startswith('reset'):
                continue
            elif line.startswith('if'):
                continue
            elif line.startswith('gate'):
                continue
            elif line.startswith('opaque'):
                continue
        
        # Create circuit
        if num_qubits > 0:
            qc = QuantumCircuit(num_qubits, num_clbits)
        else:
            qc = QuantumCircuit(2, 2)  # Default circuit
        
        # Parse gates
        for line in qasm_lines:
            if line.startswith('OPENQASM') or line.startswith('include'):
                continue
            elif line.startswith('qreg') or line.startswith('creg'):
                continue
            elif line.startswith('barrier'):
                continue
            elif line.startswith('measure'):
                continue
            elif line.startswith('reset'):
                continue
            elif line.startswith('if'):
                continue
            elif line.startswith('gate'):
                continue
            elif line.startswith('opaque'):
                continue
            elif line.startswith('h'):
                # Hadamard gate
                match = re.match(r'h\s+(\w+)\[(\d+)\]', line)
                if match:
                    qubit = int(match.group(2))
                    if qubit < num_qubits:
                        qc.h(qubit)
            elif line.startswith('x'):
                # X gate
                match = re.match(r'x\s+(\w+)\[(\d+)\]', line)
                if match:
                    qubit = int(match.group(2))
                    if qubit < num_qubits:
                        qc.x(qubit)
            elif line.startswith('y'):
                # Y gate
                match = re.match(r'y\s+(\w+)\[(\d+)\]', line)
                if match:
                    qubit = int(match.group(2))
                    if qubit < num_qubits:
                        qc.y(qubit)
            elif line.startswith('z'):
                # Z gate
                match = re.match(r'z\s+(\w+)\[(\d+)\]', line)
                if match:
                    qubit = int(match.group(2))
                    if qubit < num_qubits:
                        qc.z(qubit)
            elif line.startswith('cx'):
                # CNOT gate
                match = re.match(r'cx\s+(\w+)\[(\d+)\],\s*(\w+)\[(\d+)\]', line)
                if match:
                    control = int(match.group(2))
                    target = int(match.group(4))
                    if control < num_qubits and target < num_qubits:
                        qc.cx(control, target)
            elif line.startswith('rx'):
                # RX gate
                match = re.match(r'rx\s*\(([^)]+)\)\s+(\w+)\[(\d+)\]', line)
                if match:
                    try:
                        angle = float(match.group(1))
                        qubit = int(match.group(3))
                        if qubit < num_qubits:
                            qc.rx(angle, qubit)
                    except ValueError:
                        continue
            elif line.startswith('ry'):
                # RY gate
                match = re.match(r'ry\s*\(([^)]+)\)\s+(\w+)\[(\d+)\]', line)
                if match:
                    try:
                        angle = float(match.group(1))
                        qubit = int(match.group(3))
                        if qubit < num_qubits:
                            qc.ry(angle, qubit)
                    except ValueError:
                        continue
            elif line.startswith('rz'):
                # RZ gate
                match = re.match(r'rz\s*\(([^)]+)\)\s+(\w+)\[(\d+)\]', line)
                if match:
                    try:
                        angle = float(match.group(1))
                        qubit = int(match.group(3))
                        if qubit < num_qubits:
                            qc.rz(angle, qubit)
                    except ValueError:
                        continue
        
        return qc
        
    except Exception as e:
        raise Exception(f"Error parsing QASM: {str(e)}")

def simulate_circuit(circuit, noise_model=None):
    """Simulate a quantum circuit and return density matrix."""
    try:
        # Use statevector_simulator for statevector method
        statevector_backend = Aer.get_backend("statevector_simulator")
        compiled_statevector = transpile(circuit, statevector_backend)
        
        result = statevector_backend.run(compiled_statevector).result()
        
        statevector = result.get_statevector()
        rho = DensityMatrix(statevector)
        return rho
    except Exception as e:
        raise Exception(f"Simulation failed: {str(e)}")

def get_single_qubit_dm(rho, qubit, total_qubits):
    """Return reduced density matrix for one qubit."""
    try:
        if total_qubits == 1:
            return rho
        else:
            reduced = partial_trace(rho, [i for i in range(total_qubits) if i != qubit])
            return reduced
    except Exception as e:
        raise Exception(f"Failed to get reduced density matrix: {str(e)}")

def dm_to_bloch(dm):
    """Convert density matrix to Bloch vector (x,y,z)."""
    try:
        X, Y, Z = Pauli("X"), Pauli("Y"), Pauli("Z")
        x = np.real(np.trace(dm.data @ X.to_matrix()))
        y = np.real(np.trace(dm.data @ Y.to_matrix()))
        z = np.real(np.trace(dm.data @ Z.to_matrix()))
        return [x, y, z]
    except Exception as e:
        raise Exception(f"Failed to convert to Bloch vector: {str(e)}")

def get_state_probabilities(statevector):
    """Get probabilities for each computational basis state."""
    try:
        if hasattr(statevector, 'data'):
            state_data = statevector.data
        else:
            state_data = statevector
            
        probabilities = np.abs(state_data) ** 2
        return probabilities
    except Exception as e:
        raise Exception(f"Failed to get probabilities: {str(e)}")

def calculate_entanglement_entropy(rho, qubit, total_qubits):
    """Calculate von Neumann entropy for a qubit subsystem."""
    try:
        if total_qubits == 1:
            return 0.0
        
        reduced_dm = get_single_qubit_dm(rho, qubit, total_qubits)
        eigenvalues = np.linalg.eigvalsh(reduced_dm.data)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
        
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return np.real(entropy)
    except Exception as e:
        raise Exception(f"Failed to calculate entanglement entropy: {str(e)}")

if __name__ == "__main__":
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0)
    rho = simulate_circuit(qc)
    dm = get_single_qubit_dm(rho, 0, 1)
    print("Bloch vector:", dm_to_bloch(dm))
