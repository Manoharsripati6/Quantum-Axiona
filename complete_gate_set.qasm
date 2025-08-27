OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Qubit 0: Full gate set demonstration
h q[0];           // Hadamard - creates superposition
x q[0];           // X gate - flips state
ry(0.7854) q[0]; // π/4 rotation around Y
rz(0.5236) q[0]; // π/6 rotation around Z

// Qubit 1: Different combination of gates
y q[1];           // Y gate - applies iY transformation
h q[1];           // Hadamard after Y
rx(1.0472) q[1]; // π/3 rotation around X
z q[1];           // Z gate - applies phase shift
ry(0.3927) q[1]; // π/8 rotation around Y

// Qubit 2: Another gate combination
z q[2];           // Z gate first
h q[2];           // Hadamard after Z
x q[2];           // X gate after H
rx(0.5236) q[2]; // π/6 rotation around X
ry(0.7854) q[2]; // π/4 rotation around Y
rz(0.2618) q[2]; // π/12 rotation around Z

// Create full entanglement network with CNOT gates
cx q[0],q[1];     // CNOT: control=0, target=1
cx q[1],q[2];     // CNOT: control=1, target=2
cx q[2],q[0];     // CNOT: control=2, target=0 (creates 3-qubit entanglement)

// Additional CNOT patterns for maximum complexity
cx q[0],q[2];     // CNOT: control=0, target=2
cx q[1],q[0];     // CNOT: control=1, target=0

// Final single-qubit gates to ensure non-zero coordinates
x q[0];           // Final X on qubit 0
y q[1];           // Final Y on qubit 1
z q[2];           // Final Z on qubit 2

// Additional rotations to make it even more complex
rx(0.1745) q[0]; // π/18 rotation around X
ry(0.2618) q[1]; // π/12 rotation around Y
rz(0.3927) q[2]; // π/8 rotation around Z
