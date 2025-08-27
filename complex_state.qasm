OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Qubit 0: Put in superposition with H gate (Z = 0, X = 0, Y = 0 initially, but will have non-zero components)
h q[0];

// Qubit 1: Apply rotation around Y-axis to get non-zero Y and Z components
ry(1.5708) q[1];  // π/2 rotation around Y-axis

// Qubit 2: Apply rotation around X-axis to get non-zero X and Z components  
rx(0.7854) q[2];  // π/4 rotation around X-axis

// Add some entanglement between qubits
cx q[0],q[1];     // CNOT between qubits 0 and 1
cx q[1],q[2];     // CNOT between qubits 1 and 2

// Additional rotations to make coordinates more interesting
rz(0.5236) q[0];  // π/6 rotation around Z-axis
ry(0.7854) q[1];  // π/4 rotation around Y-axis
