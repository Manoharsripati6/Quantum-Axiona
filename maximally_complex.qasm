OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Qubit 0: Complex superposition with multiple rotations
h q[0];           // Hadamard creates superposition
ry(0.7854) q[0]; // π/4 rotation around Y
rz(0.5236) q[0]; // π/6 rotation around Z

// Qubit 1: Different rotation pattern
rx(1.0472) q[1]; // π/3 rotation around X
ry(0.5236) q[1]; // π/6 rotation around Y
h q[1];           // Hadamard for superposition

// Qubit 2: Another rotation pattern
rz(0.7854) q[2]; // π/4 rotation around Z
rx(0.5236) q[2]; // π/6 rotation around X
ry(1.0472) q[2]; // π/3 rotation around Y

// Create entanglement between all qubits
cx q[0],q[1];    // CNOT 0->1
cx q[1],q[2];    // CNOT 1->2
cx q[2],q[0];    // CNOT 2->0 (creates 3-qubit entanglement)

// Final rotations to ensure non-zero coordinates
rx(0.2618) q[0]; // π/12 rotation around X
ry(0.3927) q[1]; // π/8 rotation around Y
rz(0.1745) q[2]; // π/18 rotation around Z
