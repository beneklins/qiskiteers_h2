"""
evaluator.py - Evaluate LinearCombinaisonPauliString on wave-function (quantum circuit)

Copyright 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>
This file has been modified by <Your,Name> during the
QSciTech-QuantumBC virtual workshop on gate-based quantum computing.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from qiskit import QuantumCircuit, execute
import time
import numpy as np
from pauli_string import PauliString


class Evaluator(object):
    def __init__(self, varform, backend, execute_opts={}, measure_filter=None, record=None):
        """
        An Evaluator allows to transform a LCPS into a callable function. The LCPS is not set at the initialization.
        The evaluator will build the QuantumCircuit necessary to estimate the expected value of the LCPS. Upon using 
        the 'eval' method, it will execute these circuits and interpret the results to return an estimate of the
        expected value of the LCPS.

        Args:
            varform (qiskit.QuantumCircuit): A paramatrized QuantumCircuit.
            backend (qiskit.backend): A qiskit backend. Could be a simulator are an actual quantum computer.
            execute_opts (dict, optional): Optional arguments to be passed to the qiskit.execute function.
                                           Defaults to {}.

            measure_filter (qiskit.ignis...MeasureFilter, optional): A measure filter that can be used on the result 
                of an execution to mitigate readout errors. Defaults to None.
            
            record (object, optional): And object that could be called on each evaluation to record the results. 
                Defaults to None.
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
        Set the LCPS to be evaluated. Further LCPS can be later provided still using the same Evaluator.
        This sets the value of the attribute 'n_qubits'.
        The measurement circuits and the interpreters are generated right away with 
        'prepare_measurement_circuits_and_interpreters' (defined at the subclass level).

        Args:
            lcps (LinearCombinaisonPauliString): The LCPS to be evaluated.
        """

        self.n_qubits = lcps.n_qubits
        self.measurement_circuits, self.interpreters = self.prepare_measurement_circuits_and_interpreters(lcps)

    def eval(self, params):
        """
        Evaluate an estimate of the expectation value of the set LCPS.

        Args:
            params (list or np.array): Parameter values at which the expectation value should be evaluated.
                Will be fed to the 'varform' paramatrized QuantumCircuit.

        Returns:
            float: The value of the estimated expectation value of the LCPS.
        """

        t0 = time.time()
        eval_circuits = self.prepare_eval_circuits(params)
        counts_arrays = list()

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.2)
        # A. For each eval_circuits :
        #   1. Execute the eval_circuits on the backend
        #   2. Extract the result from the job
        #   2. (Optional) Apply error mitigation with the measure_filter
        #   3. Assemble the counts into an array with counts2array()
        #   4. Compute the exp_value of the PauliString
        # B. Combine all the results into the output value (e.i. the energy)
        # (Optional) record the result with the record object
        # (Optional) monitor the time of execution
        ################################################################################################################

        output = self.interpret_counts(counts_arrays)
        eval_time = time.time()-t0

        raise NotImplementedError()

        return output

    def prepare_eval_circuits(self, params):
        """
        Assign parameter values to the variational circuit (varfom) to set the wave function.
        Combine varform circuit with each of the measurement circuits.

        Args:
            params (list or np.array): Params to be assigned to the 'varform' QuantumCircuit.

        Returns:
            list<QuantumCircuit>: All the QuantumCircuit necessary to the evaluation of the LCPS.
        """

        eval_circuits = list()

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.2)
        ################################################################################################################

        raise NotImplementedError()

        return eval_circuits

    def counts2array(self, counts):
        """
        Transform a counts dict into an array.

        Args:
            counts (dict): The counts dict as return by qiskit.result.Result.get_counts().

        Returns:
            np.array<int>: Counts vector sorted in the usual way :
            0...00, 0...01, 0...10, ..., 1...11
        """

        array = np.zeros((2**self.n_qubits,), dtype=int)
        for state, count in counts.items():
            i = int(state,base=2)
            array[i] = count

        return array

    def interpret_count_arrays(self, counts_arrays):
        """
        Interprets all the counts_arrays resulting from the execution of all the eval_circuits.
        This computes the sum_i h_i <P_i> .

        Args:
            counts_arrays (list<np.array>): counts_arrays resulting from the execution of all the eval_circuits.

        Returns:
            float: sum of all the interpreted values of counts_arrays. Mathematical return sum_i h_i <P_i>.
        """

        cumulative = 0
        for interpreter, counts_array in zip(self.interpreters, counts_arrays):
            cumulative += self.interpret_count_array(interpreter,counts_array)
            
        return np.real(cumulative)

    @staticmethod
    def interpret_count_array(interpreter, counts_array):
        """
        Interprets the counts_array resulting from the execution of one eval_circuit.
        This computes the h_i <P_i> either for one PauliString or a Clique.

        Args:
            interpreter (np.array): Array of the Eigenvalues of the measurable PauliStrings associated with
                the circuit that returned the 'counts_array'.
            counts_array (np.array): count_arrays resulting from the execution of the eval_circuit associated 
                with the 'interpreter'.

        Returns:
            float: the interpreted values for the PauliStrings associated with the eval_circuit that gave counts_arrays.
                Mathematical return h_i <P_i> 
        """

        value = None

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.2)
        ################################################################################################################
        
        raise NotImplementedError()

        return value

    @staticmethod
    def pauli_string_based_measurement(pauli_string):
        """
        Build a QuantumCircuit that measures the qubits in the basis given by a PauliString.

        Args:
            pauli_string (PauliString): 

        Returns:
            qiskit.QuantumCircuit: A quantum circuit starting with rotation of the qubit following the PauliString and
                                   finishing with the measurement of all the qubits.
        """

        n_qubits = len(pauli_string)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.2)
        ################################################################################################################
        
        raise NotImplementedError()

        return qc

    @staticmethod
    def measurable_eigenvalues(pauli_string):
        """
        Build the eigenvalues vector (size = 2**n_qubits) for the measurable version of a given PauliString.

        Args:
            pauli_string ([type]): [description]

        Returns:
            [type]: [description]
        """

        eigenvalues = None

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.2)
        # Hint : starts with
        eigenvalues = np.ones((1,), dtype=np.int)
        # and use np.kron to build the vector
        # Eigenvalues of X, Y and Z are 1 and -1
        # Eigenvalues of I are 1 and 1
        EIG_Z = np.array([1,-1], dtype=np.int)
        EIG_I = np.array([1,1], dtype=np.int)
        ################################################################################################################
        
        raise NotImplementedError()

        return eigenvalues


class BasicEvaluator(Evaluator):
    """
    The BasicEvaluator should build 1 quantum circuit and 1 interpreter for each PauliString.
    The interpreter should be 1d array of size 2**number of qubits.
    It does not exploit the fact that commuting PauliStrings can be evaluated from a common circuit.
    """
    
    @staticmethod
    def prepare_measurement_circuits_and_interpreters(lcps):
        """
        For each PauliString in the LCPS, this method build a measurement QuantumCircuit and provide the associated
        interpreter. This interpreter allow to compute h<P> = sum_i T_i N_i/N_tot for each PauliString.

        Args:
            lcps (LinearCombinaisonPauliString): The LCPS to be evaluated, being set.

        Returns:
            list<qiskit.QuantumCircuit>, list<np.array>: [description]
        """
        
        circuits = list()
        interpreters = list()

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.2)
        # Hint : the next method does the work for 1 PauliString + coef
        ################################################################################################################
        
        raise NotImplementedError()

        return circuits, interpreters

    @staticmethod
    def pauli_string_circuit_and_interpreter(coef, pauli_string):
        """
        This method builds a measurement QuantumCircuit for a PauliString and provide the associated interpreter.
        The interpreter includes the associated coef for convenience.

        Args:
            coef (complex or float): The coef associated to the 'pauli_string'.
            pauli_string (PauliString): PauliString to be measured and interpreted.

        Returns:
            qiskit.QuantumCircuit, np.array: The QuantumCircuit to be used to measure in the basis given by the 
                PauliString given with the interpreter to interpret to result of the eventual eval_circuit.
        """

        circuit = Evaluator.pauli_string_based_measurement(pauli_string)
        interpreter = coef * Evaluator.measurable_eigenvalues(pauli_string)

        return circuit, interpreter


class BitwiseCommutingCliqueEvaluator(Evaluator):
    """
    The BitwiseCommutingCliqueEvaluator should build 1 quantum circuit and 1 interpreter for each clique of PauliStrings.
    The interpreter should be 2d array of size (number of cliques ,2**number of qubits).
    It does exploit the fact that commuting PauliStrings can be evaluated from a common circuit.
    """

    @staticmethod
    def prepare_measurement_circuits_and_interpreters(lcps):
        """
        Divide the LCPS into bitwise commuting cliques.
        For each PauliString clique in the LCPS, this method builds a measurement QuantumCircuit and provides the
        associated interpreter. This interpreter allows to compute sum_i h_i<P_i> = sum_j T_j N_j/N_tot for each
        PauliString clique.

        Args:
            lcps (LinearCombinaisonPauliString): The LCPS to be evaluated, being set.

        Returns:
            list<qiskit.QuantumCircuit>, list<np.array>: The QuantumCircuit to be used to measure in the basis given by
                the PauliString clique given with the interpreter to interpret to result of the evantual eval_circuit.
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
        This method builds a measurement QuantumCircuit for a PauliString clique and provides the associated interpreter.
        The interpreter includes the associated coefs for conveniance

        Args:
            clique (LinearCombinaisonPauliString): A LCPS where all PauliString bitwise commute with on an other.

        Returns:
            qiskit.QuantumCircuit, np.array: The QuantumCircuit to be used to measure in the basis given by the 
                PauliString clique given with the interpreter to interpret to result of the evantual eval_circuit.
        """

        interpreter = circuit = None

        ################################################################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after activity 3.2)
        # Hint : the next method does the work for 1 PauliString + coef
        ################################################################################################################

        interpreter = np.zeros((len(clique), 2**clique.n_qubits))
        
        raise NotImplementedError()

        return circuit, interpreter

