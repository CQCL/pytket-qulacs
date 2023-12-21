# Copyright 2019-2023 Cambridge Quantum Computing
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

"""Conversion from to tket circuits to Qulacs circuits
"""
import numpy as np
from qulacs import QuantumCircuit, gate
from pytket.circuit import Circuit, OpType
from pytket.unit_id import Qubit


def get_register_offset_map(circuit: Circuit) -> dict[str, int]:
    """Creates a map that accounts for the offset required to access the
    `pytket._tket.unit_id.Qubit`s of a `pytket._tket.unit_id.QubitRegister`
    with an absolute index.

    Args:
        circuit (Circuit): The circuit for which to create the map.

    Returns:
        dict[str, int]: A dictionary where the keys are the names of the quantum registers
            of the circuit and the values are the total number of qubits in preceeding registers.
    """
    qubits_preceeding = 0
    offset_dict: dict[str, int] = {}

    for register in circuit.q_registers:
        # The number of qubits to offset by is the number
        # Of qubits in all register preceeding this one
        offset_dict[register.name] = qubits_preceeding

        # Increase the number of qubits
        # By the number of qubits this register contains
        qubits_preceeding += register.size

    return offset_dict


def get_qubit_index_map(circuit: Circuit, reverse_order: bool) -> dict[Qubit, int]:
    """Creates a map that contains the absolute position of a qubit in a given circuit,
    accounting for each qubit's register as well.

    Args:
        circuit (Circuit): The circuit for which to create the map.
        reverse_order (bool): Whether the circuits' qubit positions are reversed.

    Returns:
        dict[Qubit, int]: A dictionary where the keys are the circuit's qubits
            and the values are the absolute positions of qubits, accounting for the register's position.
    """
    # Get the dictionary accounting for register positions
    offset_map = get_register_offset_map(circuit)
    index_map: dict[Qubit, int] = {}

    for register in circuit.q_registers:
        for qubit_index, qubit in enumerate(register.to_list()):
            # The position of the qubit is the sum of the offset (preceeding qubits)
            # And the position of the qubit within its register
            qubit_position = offset_map[register.name] + qubit_index

            # Invert the if the qubits are stored in reverse order
            index_map[qubit] = (
                qubit_position
                if not reverse_order
                else circuit.n_qubits - 1 - qubit_position
            )

    return index_map


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
    if replace_implicit_swaps:
        circ.replace_implicit_wire_swaps()
    n_qubits = circ.n_qubits
    qulacs_circ = QuantumCircuit(circ.n_qubits)

    # Dictionary mapping qubits to their absolute position
    # Within the quantum circuit
    index_map = get_qubit_index_map(circ, reverse_index)

    for com in circ:
        optype = com.op.type
        if optype in _IBM_GATES:
            qulacs_gate = _IBM_GATES[optype]
            index = index_map[com.qubits[0]]

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
            index = index_map[com.qubits[0]]
            add_gate = qulacs_gate(index)

        elif optype in _ONE_QUBIT_ROTATIONS:
            qulacs_gate = _ONE_QUBIT_ROTATIONS[optype]
            index = index_map[com.qubits[0]]
            param = com.op.params[0] * np.pi
            add_gate = qulacs_gate(index, -param)  # parameter negated for qulacs

        elif optype in _TWO_QUBIT_GATES:
            qulacs_gate = _TWO_QUBIT_GATES[optype]
            id1 = index_map[com.qubits[0]]
            id2 = index_map[com.qubits[1]]
            add_gate = qulacs_gate(id1, id2)

        elif optype in _MEASURE_GATES:
            continue
            # gate = _MEASURE_GATES[optype]
            # qubit = com.qubits[0].index[0]
            # bit = com.bits[0].index[0]
            # add_gate = (gate(qubit, bit))

        elif optype == OpType.Barrier:
            continue

        else:
            raise NotImplementedError(
                "Gate: {} Not Implemented in Qulacs!".format(optype)
            )
        qulacs_circ.add_gate(add_gate)

    return qulacs_circ
