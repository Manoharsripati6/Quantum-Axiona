# 🔮 Quantum State Visualizer

An interactive web application for visualizing quantum states, building quantum circuits, and exploring quantum mechanics concepts through Bloch sphere representations. **Now with QASM file import support!**

## Features

- **QASM File Import**: Upload and parse QASM files to visualize existing quantum circuits
- **Interactive Circuit Builder**: Create quantum circuits with up to 5 qubits
- **Real-time Simulation**: Simulate quantum circuits and visualize results
- **Bloch Sphere Visualization**: 3D interactive Bloch spheres for each qubit
- **Multiple Gate Types**: Support for single-qubit and two-qubit gates
- **State Analysis**: View state vectors, density matrices, and purity calculations
- **Example Circuits**: Pre-built examples like Bell states, GHZ states, and W states
- **Export Functionality**: Download your circuits as QASM files

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd quantum_state_visualizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### Importing QASM Files

1. **Upload QASM**: Use the file uploader in the sidebar to select a `.qasm` or `.txt` file
2. **Automatic Parsing**: The application will automatically parse the QASM and create the circuit
3. **View Content**: Expand the "View QASM Content" section to see the original QASM code
4. **Visualize**: The circuit will be displayed with full visualization capabilities

### Building Circuits

1. **Configure Circuit**: Use the sidebar to set the number of qubits and classical bits
2. **Add Gates**: Click on gate buttons to add operations to your circuit
3. **Select Target**: Choose which qubit to apply gates to
4. **View Circuit**: See your circuit diagram and statistics in real-time

### Available Gates

- **Single-qubit gates**: H (Hadamard), X, Y, Z, RX, RY, RZ
- **Two-qubit gates**: CX (CNOT), CY, CZ
- **Rotation gates**: Adjustable rotation angles

### Visualization Features

- **Bloch Spheres**: 3D representation of qubit states
- **State Vectors**: View the complete quantum state
- **Density Matrices**: Examine the quantum state representation
- **Purity Analysis**: Determine if the state is pure or mixed
- **Circuit Diagrams**: Visual representation of quantum circuits

## QASM File Format Support

The application supports standard QASM 2.0 files with the following features:

- **Basic Gates**: h, x, y, z, cx, rx, ry, rz
- **Registers**: qreg (quantum) and creg (classical)
- **Comments**: Lines starting with `//` are ignored
- **Standard Includes**: `qelib1.inc` is supported

### Sample QASM Files

**Bell State** (`sample_bell_state.qasm`):
```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
```

**GHZ State** (`sample_ghz_state.qasm`):
```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
```

## Example Circuits

### Bell State
Creates the maximally entangled state: `(|00⟩ + |11⟩)/√2`

### GHZ State
Creates a three-qubit entangled state: `(|000⟩ + |111⟩)/√2`

### W State
Creates a three-qubit entangled state: `(|001⟩ + |010⟩ + |100⟩)/√3`

## Technical Details

- **Backend**: Qiskit Aer simulator for quantum circuit simulation
- **Visualization**: Plotly for interactive 3D Bloch spheres
- **Web Framework**: Streamlit for the user interface
- **Quantum Framework**: Qiskit for quantum computing operations
- **QASM Parser**: Custom parser supporting standard QASM 2.0 syntax

## File Structure

```
quantum_state_visualizer/
├── app.py                    # Main Streamlit application
├── quantum_tools.py          # Quantum simulation and QASM parsing functions
├── visualizer.py             # Bloch sphere and plotting functions
├── requirements.txt          # Python dependencies
├── sample_bell_state.qasm    # Sample Bell state QASM file
├── sample_ghz_state.qasm    # Sample GHZ state QASM file
└── README.md                # This file
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Simulation Failures**: Check that your circuit has valid gate combinations
3. **Visualization Issues**: Ensure Plotly is properly installed and updated
4. **QASM Parsing Errors**: Check that your QASM file follows standard 2.0 syntax

### QASM Import Tips

- Use standard QASM 2.0 syntax
- Ensure proper register declarations
- Check that gate names match supported operations
- Verify qubit indices are within register bounds

### Performance Tips

- Keep circuits under 5 qubits for optimal performance
- Use the reset button to clear complex circuits
- Start with simple examples to understand the interface
- Import existing QASM files instead of building from scratch

## Contributing

Feel free to contribute by:
- Adding new gate types
- Improving visualizations
- Adding more example circuits
- Enhancing error handling
- Extending QASM parser support

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [Qiskit](https://qiskit.org/) - IBM's quantum computing framework
- Powered by [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- Visualizations created with [Plotly](https://plotly.com/) - Interactive plotting library
- QASM support based on OpenQASM 2.0 specification
