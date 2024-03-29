"""
evaluator.py - Evaluate LinearCombinaisonPauliString on state functions
               (quantum circuit)

Copyright © 2021 Brett Henderson <brettrhenderson25@gmail.com>,
                 Igor Benek-Lins <physics@ibeneklins.com>,
                 Melvin Mathews <mel.matt007@gmail.com>,
Copyright © 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>

This file is part of “The Three Qiskiteers H₂ ground state finder” (T3QH2).
For the licence, see the LICENCE file.
"""

from qiskit import QuantumCircuit, execute
import time
import numpy as np
from pauli_string import PauliString


class Evaluator(object):
    def __init__(self, varform, backend, execute_opts={},
                 measure_filter=None, record=None):
        """
        An Evaluator allows to transform a LCPS into a callable function.
        The LCPS is not set at the initialization. The evaluator will build
        the quantum circuit necessary to estimate the expected value of the
        LCPS. Upon using the `eval` method, it will execute these circuits and
        interpret the results to return an estimate of the expected value of
        the LCPS.

        Parameters
        ----------
        varform : qiskit.QuantumCircuit
            A parametrised QuantumCircuit.
        backend : qiskit.backend
            A Qiskit backend. Could be a simulator are an actual quantum computer.
        execute_opts : dict, optional, default={}
            Optional arguments to be passed to the qiskit.execute function.
        measure_filter : qiskit.ignis...MeasureFilter, optional, default=None
            A measure filter that can be used on the result of an execution to
            mitigate readout errors. Defaults to None.
        record : object, optional, default=None
            And object that could be called on each evaluation to record the
            results.
        """

        self.varform = varform
        self.backend = backend
        self.execute_opts = execute_opts

        self.measure_filter = measure_filter
        self.record = record

        # To be set attributes
        self.n_qubits = None
        self.measurement_circuits = list()
        self.interpreters = list()

    def set_linear_combinaison_pauli_string(self, lcps):
        """
        Set the LCPS to be evaluated. Further LCPS can be later provided still
        using the same Evaluator. This sets the value of the attribute
        `n_qubits`. The measurement circuits and the interpreters are
        generated right away with
        `prepare_measurement_circuits_and_interpreters` (defined at the
        subclass level).

        Parameters
        ----------
        lcps : LinearCombinaisonPauliString)
            The LCPS to be evaluated.
        """

        self.n_qubits = lcps.n_qubits
        self.measurement_circuits, self.interpreters = self.prepare_measurement_circuits_and_interpreters(lcps)

    def eval(self, params):
        """
        Evaluate an estimate of the expectation value of the set LCPS.

        Parameters
        ----------
        params : list or np.array
            Parameter values at which the expectation value should be evaluated. Will be fed to the `varform` parametrised
            QuantumCircuit.

        Returns
        -------
        float
            The value of the estimated expectation value of the LCPS.
        """

        t0 = time.time()
        eval_circuits = self.prepare_eval_circuits(params)
        counts_arrays = list()

        # Activity 3.2.
        # TODO: Record the result with the record object
        # TODO: Monitor the time of execution.
        job = execute(eval_circuits, backend=self.backend, **self.execute_opts)
        result = job.result()
        for i in range(len(eval_circuits)):
            counts = result.get_counts(eval_circuits[i])
            if self.measure_filter is not None:
                counts = self.measure_filter.apply(counts)
            counts_arrays.append(self.counts2array(counts))

        output = self.interpret_count_arrays(counts_arrays)

        eval_time = time.time()-t0

        return output

    def prepare_eval_circuits(self, params):
        """
        Assign parameter values to the variational circuit (`varfom`) to set
        the wave function. Combine `varform` circuit with each of the
        measurement circuits.

        Parameters
        ----------
        params : list or np.array
            Parameters to be assigned to the `varform` QuantumCircuit.

        Returns
        -------
        list<QuantumCircuit>
            All the QuantumCircuit necessary to the evaluation of the LCPS.
        """

        # Activity 3.2.
        eval_circuits = list()

        param_dict = dict(zip(self.varform.parameters, params))
        parameterized_qc = self.varform.assign_parameters(param_dict)
        for measurement_circuit in self.measurement_circuits:
            eval_circuits.append(parameterized_qc.copy() + measurement_circuit)

        return eval_circuits

    def counts2array(self, counts):
        """
        Transform a counts dictionary into an array.

        Parameters
        ----------
        counts : dict
            The counts dict as return by `qiskit.result.Result.get_counts()`.

        Returns
        -------
        np.array<int>
            Counts vector sorted in the usual way:
            `0...00, 0...01, 0...10, ..., 1...11`
        """

        # TODO: Optimise this routine using dictionaries.
        array = np.zeros((2**self.n_qubits,), dtype=int)
        for state, count in counts.items():
            i = int(state,base=2)
            array[i] = count

        return array

    def interpret_count_arrays(self, counts_arrays):
        """
        Interprets all the counts_arrays resulting from the execution of all
        the eval_circuits. This computes the :math:`\sum_i h_i <P_i>`.

        Parameters
        ----------
        counts_arrays : list<np.array>
            counts_arrays resulting from the execution of all the
            `eval_circuits`.

        Returns
        -------
        float
            Sum of all the interpreted values of counts_arrays.
            Mathematically returns :math:`\sum_i h_i <P_i>`.
        """

        cumulative = 0
        for interpreter, counts_array in zip(self.interpreters, counts_arrays):
            cumulative += self.interpret_count_array(interpreter,counts_array)

        return np.real(cumulative)

    @staticmethod
    def interpret_count_array(interpreter, counts_array):
        """
        Interprets the counts_array resulting from the execution of one
        `eval_circuit`.
        This computes the :math:`h_i <P_i>` either for one `PauliString` or a
        `Clique`.

        Parameters
        ----------
        interpreter : np.array
            Array of the eigenvalues of the measurable `PauliStrings`
            associated with the circuit that returned the `counts_array`.
        counts_array : np.array
            `count_arrays` resulting from the execution of the `eval_circuit`
            associated with the `interpreter`.

        Returns
        -------
        float
            the interpreted values for the `PauliStrings` associated with the
            `eval_circuit` that gave counts_arrays. Mathematically returns
            :math:`h_i <P_i>`.
        """

        # Activity 3.2.
        value = interpreter.dot(counts_array) / counts_array.sum()

        return value

    @staticmethod
    def pauli_string_based_measurement(pauli_string):
        """
        Build a `QuantumCircuit` that measures the qubits in the basis given by
        a `PauliString`.

        Parameters
        ----------
        pauli_string : PauliString

        Returns
        -------
        qiskit.QuantumCircuit
            A quantum circuit starting with rotation of the qubit following
            the `PauliString` and finishing with the measurement of all the
            qubits.
        """

        n_qubits = len(pauli_string)
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Activity 3.2.
        for i, (x, z) in enumerate(zip(pauli_string.x_bits, pauli_string.z_bits)):
            if x == 1 and z == 0:
                qc.h(i)
            elif x == 1 and z == 1:
                qc.sdg(i)
                qc.h(i)
        for i in range(n_qubits):
            qc.measure(i, i)

        return qc

    @staticmethod
    def measurable_eigenvalues(pauli_string):
        """
        Build the eigenvalues vector (size = `2**n_qubits`) for the measurable
        version of a given `PauliString`.

        Parameters
        ----------
        pauli_string : PauliString

        Returns
        -------
        np.array<int>
            The eigenvalues.
        """

        # Activity 3.2
        # TODO: Verify if the use np.kron would be useful.
        eigenvalues = np.zeros((2**len(pauli_string),), dtype=np.int)
        # Eigenvalues of X, Y and Z are 1 and -1
        # Eigenvalues of I are 1 and 1
        EIG_Z = np.array([+1,-1], dtype=np.int)
        EIG_I = np.array([+1,+1], dtype=np.int)

        for i in range(2**len(pauli_string)):
            s = f'0{len(pauli_string)}b'
            bitstring = f'{i:{s}}'
            pauli_eigenvalues = np.zeros(len(pauli_string))

            for j, xyz in enumerate(pauli_string.x_bits + pauli_string.z_bits):
                # Measurable Pauli Z
                if xyz:
                    pauli_eigenvalues[j] = EIG_Z[int(bitstring[-(j+1)])]
                # Measurable Pauli I
                else:
                    pauli_eigenvalues[j] = EIG_I[int(bitstring[-(j+1)])]
            eigenvalues[i] = np.prod(pauli_eigenvalues)

        return eigenvalues


class BasicEvaluator(Evaluator):
    """
    The `BasicEvaluator` should build 1 quantum circuit and 1 interpreter for
    each `PauliString`. The interpreter should be a one-dimensional array of
    size `2**number of qubits`. It does not exploit the fact that commuting
    `PauliStrings` can be evaluated from a common circuit.
    """

    @staticmethod
    def prepare_measurement_circuits_and_interpreters(lcps):
        """
        For each `PauliString` in the LCPS, this method build a measurement
        `QuantumCircuit` and provide the associated interpreter. This
        interpreter allows to compute :math:`h<P> = \sum_i T_i N_i/N_{tot}`
        for each `PauliString`.

        Parameters
        ----------
        lcps : LinearCombinaisonPauliString
            The LCPS to be evaluated, being set.

        Returns
        -------
        list<qiskit.QuantumCircuit>, list<np.array>
        """

        circuits = list()
        interpreters = list()

        # Activity 3.2.
        for pauli_string, coef in zip(lcps.pauli_strings, lcps.coefs):
            circuit, interpreter = BasicEvaluator.pauli_string_circuit_and_interpreter(coef, pauli_string)
            circuits.append(circuit)
            interpreters.append(interpreter)

        return circuits, interpreters

    @staticmethod
    def pauli_string_circuit_and_interpreter(coef, pauli_string):
        """
        This method builds a measurement `QuantumCircuit` for a `PauliString`
        and provide the associated interpreter. The interpreter includes the
        associated coefficient for convenience.

        Parameters
        ----------
        coef : complex or float
            The coefficient associated to the `pauli_string`.
        pauli_string : PauliString
            `PauliString` to be measured and interpreted.

        Returns
        -------
        qiskit.QuantumCircuit, np.array
            The `QuantumCircuit` to be used to measure in the basis given by
            the `PauliString` given with the interpreter to interpret to
            result of the eventual eval_circuit.
        """

        circuit = Evaluator.pauli_string_based_measurement(pauli_string)
        interpreter = coef * Evaluator.measurable_eigenvalues(pauli_string)

        return circuit, interpreter


class BitwiseCommutingCliqueEvaluator(Evaluator):
    """
    The `BitwiseCommutingCliqueEvaluator` should build 1 quantum circuit and
    1 interpreter for each clique of `PauliStrings`. The interpreter should be
    a two-dimensional array of size `(number of cliques, 2**number of qubits)`.
    It does exploit the fact that commuting PauliStrings can be evaluated from
    a common circuit.
    """

    @staticmethod
    def prepare_measurement_circuits_and_interpreters(lcps):
        """
        Divide the LCPS into bitwise commuting cliques. For each `PauliString`
        clique in the LCPS, this method builds a measurement `QuantumCircuit`
        and provides the associated interpreter. This interpreter allows to
        compute :math:`\sum_i h_i <P_i> = \sum_j T_j N_j/N_{tot}` for each
        `PauliString` clique.

        Parameters
        ----------
        lcps : LinearCombinaisonPauliString
            The LCPS to be evaluated, being set.

        Returns
        -------
        list<qiskit.QuantumCircuit>, list<np.array>
            The `QuantumCircuit` to be used to measure in the basis given by
            the `PauliString` clique given with the interpreter to interpret
            to result of the eventual `eval_circuit`.
        """

        cliques = lcps.divide_in_bitwise_commuting_cliques()

        circuits = list()
        interpreters = list()

        for clique in cliques:
            circuit, interpreter = BitwiseCommutingCliqueEvaluator.bitwise_clique_circuit_and_interpreter(clique)
            circuits.append(circuit)
            interpreters.append(interpreter)

        return circuits, interpreters

    @staticmethod
    def bitwise_clique_circuit_and_interpreter(clique):

        """
        This method builds a measurement `QuantumCircuit` for a `PauliString`
        clique and provides the associated interpreter. The interpreter
        includes the associated coefficients for convenience.

        Parameters
        ----------
        clique : LinearCombinaisonPauliString
            A LCPS where all `PauliString` bitwise commute with another.

        Returns
        -------
        qiskit.QuantumCircuit, np.array
            The `QuantumCircuit` to be used to measure in the basis given by
            the `PauliString` clique given with the interpreter to interpret
            to result of the eventual `eval_circuit`.
        """

        interpreter = circuit = None

        #################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after activity 3.2)
        # Hint : the next method does the work for 1 PauliString + coef
        #################################################################

        interpreter = np.zeros((len(clique), 2**clique.n_qubits))

        raise NotImplementedError()

        return circuit, interpreter
