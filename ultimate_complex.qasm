OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// PHASE 1: Initialize all qubits with different states
h q[0];           // Qubit 0: Hadamard
y q[1];           // Qubit 1: Y gate
z q[2];           // Qubit 2: Z gate

// PHASE 2: Apply rotation gates to each qubit
rx(0.7854) q[0]; // π/4 rotation around X
ry(0.5236) q[0]; // π/6 rotation around Y
rz(0.3927) q[0]; // π/8 rotation around Z

rx(1.0472) q[1]; // π/3 rotation around X
ry(0.7854) q[1]; // π/4 rotation around Y
rz(0.2618) q[1]; // π/12 rotation around Z

rx(0.5236) q[2]; // π/6 rotation around X
ry(1.0472) q[2]; // π/3 rotation around Y
rz(0.7854) q[2]; // π/4 rotation around Z

// PHASE 3: Apply Pauli gates
x q[0];           // X gate on qubit 0
y q[1];           // Y gate on qubit 1
z q[2];           // Z gate on qubit 2

// PHASE 4: More Hadamards and rotations
h q[0];           // Another Hadamard
h q[1];           // Another Hadamard
h q[2];           // Another Hadamard

rx(0.2618) q[0]; // π/12 rotation around X
ry(0.1745) q[1]; // π/18 rotation around Y
rz(0.3927) q[2]; // π/8 rotation around Z

// PHASE 5: Create maximum entanglement with CNOT gates
cx q[0],q[1];     // CNOT: 0->1
cx q[1],q[2];     // CNOT: 1->2
cx q[2],q[0];     // CNOT: 2->0

cx q[0],q[2];     // CNOT: 0->2
cx q[1],q[0];     // CNOT: 1->0
cx q[2],q[1];     // CNOT: 2->1

// PHASE 6: Final gate applications
x q[0];           // Final X
y q[1];           // Final Y
z q[2];           // Final Z

h q[0];           // Final Hadamard
h q[1];           // Final Hadamard
h q[2];           // Final Hadamard

// PHASE 7: Ultimate rotations
rx(0.0873) q[0]; // π/36 rotation around X
ry(0.1309) q[1]; // π/24 rotation around Y
rz(0.1745) q[2]; // π/18 rotation around Z

// PHASE 8: Final entanglement
cx q[0],q[1];     // Final CNOT: 0->1
cx q[1],q[2];     // Final CNOT: 1->2
cx q[2],q[0];     // Final CNOT: 2->0
