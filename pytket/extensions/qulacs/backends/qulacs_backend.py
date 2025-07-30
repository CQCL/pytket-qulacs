# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from logging import warning
from random import Random
from typing import Optional, Union, cast
from uuid import uuid4

import numpy as np
from sympy import Expr

from pytket.backends import (
    Backend,
    CircuitNotRunError,
    CircuitStatus,
    ResultHandle,
    StatusEnum,
)
from pytket.backends.backend import KwargTypes
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, OpType, Pauli
from pytket.extensions.qulacs._metadata import __extension_version__
from pytket.extensions.qulacs.qulacs_convert import (
    _IBM_GATES,
    _MEASURE_GATES,
    _ONE_QUBIT_GATES,
    _ONE_QUBIT_ROTATIONS,
    _TWO_QUBIT_GATES,
    tk_to_qulacs,
)
from pytket.passes import (
    AutoRebase,
    BasePass,
    DecomposeBoxes,
    FlattenRegisters,
    FullPeepholeOptimise,
    SequencePass,
    SynthesiseTket,
)
from pytket.pauli import QubitPauliString
from pytket.predicates import (
    DefaultRegisterPredicate,
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.outcomearray import OutcomeArray
from qulacs import DensityMatrix, Observable, QuantumState, check_build_for_mpi

if check_build_for_mpi():
    from mpi4py import MPI  # type: ignore  # noqa: F401

_GPU_ENABLED = True
try:
    from qulacs import QuantumStateGpu  # type: ignore
except ImportError:
    _GPU_ENABLED = False


def _tk1_to_u(a: float, b: float, c: float) -> Circuit:
    circ = Circuit(1)
    circ.add_gate(OpType.U3, [b, a - 0.5, c + 0.5], [0])
    circ.add_phase(-0.5 * (a + c))
    return circ


_1Q_GATES = (
    set(_ONE_QUBIT_ROTATIONS)
    | set(_ONE_QUBIT_GATES)
    | set(_MEASURE_GATES)
    | set(_IBM_GATES)
)


class QulacsBackend(Backend):
    """
    Backend for running simulations on the Qulacs simulator
    """

    _supports_shots = True
    _supports_counts = True
    _supports_state = True
    _supports_expectation = True
    _expectation_allows_nonhermitian = False
    _persistent_handles = False
    _GATE_SET = {  # noqa: RUF012
        *_TWO_QUBIT_GATES.keys(),
        *_1Q_GATES,
        OpType.Barrier,
    }

    def __init__(
        self,
        result_type: str = "state_vector",
    ) -> None:
        """
        Backend for running simulations on the Qulacs simulator

        :param result_type: Indicating the type of the simulation result
            to be returned. It can be either "state_vector" or "density_matrix".
            Defaults to "state_vector"
        """
        super().__init__()
        self._backend_info = BackendInfo(
            type(self).__name__,
            None,
            __extension_version__,
            None,
            self._GATE_SET,
        )
        self._result_type = result_type
        self._sim: type[QuantumState | DensityMatrix | QuantumStateGpu]
        if result_type == "state_vector":
            self._sim = QuantumState
        elif result_type == "density_matrix":
            self._sim = DensityMatrix
            self._supports_state = False
            self._supports_density_matrix = True
        else:
            raise ValueError(f"Unsupported result type {result_type}")

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    @property
    def backend_info(self) -> Optional["BackendInfo"]:
        return self._backend_info

    @property
    def required_predicates(self) -> list[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(self._GATE_SET),
            DefaultRegisterPredicate(),
        ]

    def rebase_pass(self) -> BasePass:
        return AutoRebase(set(_TWO_QUBIT_GATES) | _1Q_GATES)

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return SequencePass(
                [DecomposeBoxes(), FlattenRegisters(), self.rebase_pass()]
            )
        if optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    FlattenRegisters(),
                    SynthesiseTket(),
                    self.rebase_pass(),
                ]
            )
        return SequencePass(
            [
                DecomposeBoxes(),
                FlattenRegisters(),
                FullPeepholeOptimise(),
                self.rebase_pass(),
            ]
        )

    def process_circuits(  # noqa: PLR0912
        self,
        circuits: Sequence[Circuit],
        n_shots: None | int | Sequence[int | None] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(  # noqa: SLF001
            n_shots,
            len(circuits),
            optional=True,
        )

        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=False)

        seed = cast("int | None", kwargs.get("seed"))
        rng = Random(seed) if seed else None

        handle_list = []
        for circuit, n_shots_circ in zip(circuits, n_shots_list, strict=False):
            if isinstance(self._sim, QuantumState):
                qulacs_state = self._sim(circuit.n_qubits, use_multi_cpu=True)  # type: ignore
            else:
                qulacs_state = self._sim(circuit.n_qubits)
            qulacs_state.set_zero_state()
            qulacs_circ = tk_to_qulacs(
                circuit, reverse_index=True, replace_implicit_swaps=True
            )
            qulacs_circ.update_quantum_state(qulacs_state)
            if self._result_type == "state_vector":
                state = qulacs_state.get_vector()  # type: ignore
            else:
                state = qulacs_state.get_matrix()  # type: ignore
            qubits = sorted(circuit.qubits, reverse=False)
            shots = None
            bits = None
            if n_shots_circ is not None:
                # tk_to_qulacs might add SWAPs after measurements,
                # hence we need to push the measurements through the
                # SWAPs.
                wire_map = circuit.implicit_qubit_permutation()
                bits2index = list(  # noqa: C400
                    (com.bits[0], qubits.index(wire_map[com.qubits[0]]))
                    for com in circuit
                    if com.op.type == OpType.Measure
                )
                if len(bits2index) == 0:
                    bits = circuit.bits
                    shots = OutcomeArray.from_ints([0] * n_shots_circ, len(bits))
                else:
                    bits, choose_indices = zip(*bits2index, strict=False)  # type: ignore

                    samples = self._sample_quantum_state(
                        qulacs_state, n_shots_circ, rng
                    )
                    shots = OutcomeArray.from_ints(samples, circuit.n_qubits)
                    shots = shots.choose_indices(choose_indices)  # type: ignore
            if self._result_type == "state_vector":
                try:
                    phase = float(circuit.phase)
                    coeff = np.exp(phase * np.pi * 1j)
                    state *= coeff
                except TypeError:
                    warning(  # noqa: LOG015
                        "Global phase is dependent on a symbolic parameter, so cannot "
                        "adjust for phase"
                    )
            handle = ResultHandle(str(uuid4()))
            if self._result_type == "state_vector":
                self._cache[handle] = {
                    "result": BackendResult(
                        state=state, shots=shots, c_bits=bits, q_bits=qubits
                    )
                }
            else:
                self._cache[handle] = {
                    "result": BackendResult(
                        density_matrix=state, shots=shots, c_bits=bits, q_bits=qubits
                    )
                }
            handle_list.append(handle)
            del qulacs_state
            del qulacs_circ
        return handle_list

    def _sample_quantum_state(
        self,
        quantum_state: Union[QuantumState, DensityMatrix, "QuantumStateGpu"],
        n_shots: int,
        rng: Random | None,
    ) -> list[int]:
        if rng:
            return quantum_state.sampling(n_shots, rng.randint(0, 2**32 - 1))
        return quantum_state.sampling(n_shots)

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)

    def get_operator_expectation_value(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        n_shots: int = 0,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> complex:
        if valid_check:
            self._check_all_circuits([state_circuit], nomeasure_warn=False)

        observable = Observable(state_circuit.n_qubits)
        for qps, coeff in operator._dict.items():  # noqa: SLF001
            _items = []
            if qps != QubitPauliString():
                for qubit, pauli in qps.map.items():
                    if pauli == Pauli.X:
                        _items.append("X")
                    elif pauli == Pauli.Y:
                        _items.append("Y")
                    elif pauli == Pauli.Z:
                        _items.append("Z")
                    _items.append(str(qubit.index[0]))

            qulacs_qps = " ".join(_items)
            if isinstance(coeff, Expr):
                qulacs_coeff = complex(coeff.evalf())
            else:
                qulacs_coeff = complex(coeff)
            observable.add_operator(qulacs_coeff, qulacs_qps)

        expectation_value = self._expectation_value(state_circuit, observable)
        del observable
        return expectation_value.real

    def _expectation_value(self, circuit: Circuit, operator: Observable) -> complex:
        state = self._sim(circuit.n_qubits)
        state.set_zero_state()
        ql_circ = tk_to_qulacs(circuit)
        ql_circ.update_quantum_state(state)
        expectation_value = operator.get_expectation_value(state)
        del state
        del ql_circ
        return complex(expectation_value)


if _GPU_ENABLED:

    class QulacsGPUBackend(QulacsBackend):
        """
        Backend for running simulations on the Qulacs GPU simulator
        """

        def __init__(self) -> None:
            """
            Backend for running simulations on the Qulacs GPU simulator
            """
            super().__init__()
            self._backend_info.name = type(self).__name__
            self._sim = QuantumStateGpu
