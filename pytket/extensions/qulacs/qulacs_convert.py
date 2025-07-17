# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
#     http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from pytket.circuit import Circuit, OpType
from pytket.passes import FlattenRegisters
from qulacs import QuantumCircuit, gate

_ONE_QUBIT_GATES = {
    OpType.X: gate.X,
    OpType.Y: gate.Y,
    OpType.Z: gate.Z,
    OpType.H: gate.H,
    OpType.S: gate.S,
    OpType.Sdg: gate.Sdag,
    OpType.T: gate.T,
    OpType.Tdg: gate.Tdag,
}

_ONE_QUBIT_ROTATIONS = {OpType.Rx: gate.RX, OpType.Ry: gate.RY, OpType.Rz: gate.RZ}

_MEASURE_GATES = {OpType.Measure: gate.Measurement}

_TWO_QUBIT_GATES = {OpType.CX: gate.CNOT, OpType.CZ: gate.CZ, OpType.SWAP: gate.SWAP}

_IBM_GATES = {OpType.U1: gate.U1, OpType.U2: gate.U2, OpType.U3: gate.U3}


def tk_to_qulacs(
    circuit: Circuit, reverse_index: bool = False, replace_implicit_swaps: bool = False
) -> QuantumCircuit:
    """Convert a pytket circuit to a qulacs circuit object."""
    circ = circuit.copy()

    if not circ.is_simple:
        FlattenRegisters().apply(circ)

    if replace_implicit_swaps:
        circ.replace_implicit_wire_swaps()
    n_qubits = circ.n_qubits
    qulacs_circ = QuantumCircuit(circ.n_qubits)
    index_map = {
        i: (i if not reverse_index else n_qubits - 1 - i) for i in range(n_qubits)
    }
    for com in circ:
        optype = com.op.type
        if optype in _IBM_GATES:
            qulacs_gate = _IBM_GATES[optype]
            index = index_map[com.qubits[0].index[0]]

            if optype == OpType.U1:
                param = com.op.params[0]
                add_gate = qulacs_gate(index, param * np.pi)  # type: ignore
            elif optype == OpType.U2:
                param0, param1 = com.op.params
                add_gate = qulacs_gate(index, param0 * np.pi, param1 * np.pi)  # type: ignore
            elif optype == OpType.U3:
                param0, param1, param2 = com.op.params
                add_gate = qulacs_gate(  # type: ignore
                    index, param0 * np.pi, param1 * np.pi, param2 * np.pi
                )

        elif optype in _ONE_QUBIT_GATES:
            qulacs_gate = _ONE_QUBIT_GATES[optype]
            index = index_map[com.qubits[0].index[0]]
            add_gate = qulacs_gate(index)

        elif optype in _ONE_QUBIT_ROTATIONS:
            qulacs_gate = _ONE_QUBIT_ROTATIONS[optype]
            index = index_map[com.qubits[0].index[0]]
            param = com.op.params[0] * np.pi
            add_gate = qulacs_gate(index, -param)  # parameter negated for qulacs

        elif optype in _TWO_QUBIT_GATES:
            qulacs_gate = _TWO_QUBIT_GATES[optype]
            id1 = index_map[com.qubits[0].index[0]]
            id2 = index_map[com.qubits[1].index[0]]
            add_gate = qulacs_gate(id1, id2)

        elif optype in _MEASURE_GATES or optype == OpType.Barrier:
            continue

        else:
            raise NotImplementedError(f"Gate: {optype} Not Implemented in Qulacs!")
        qulacs_circ.add_gate(add_gate)

    return qulacs_circ
