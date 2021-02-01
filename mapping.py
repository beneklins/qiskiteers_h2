"""
mapping.py - Map a Hamiltonian to a LinearCombinaisonPauliString

Copyright © 2021 Brett Henderson <brettrhenderson25@gmail.com>,
                 Igor Benek-Lins <physics@ibeneklins.com>,
                 Melvin Mathews <mel.matt007@gmail.com>,
Copyright © 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>

This file is part of “The Three Qiskiteers H₂ ground state finder” (T3QH2).
For the licence, see the LICENCE file.
"""

from pauli_string import PauliString, LinearCombinaisonPauliString
import numpy as np


class Mapping(object):

    def fermionic_hamiltonian_to_linear_combinaison_pauli_string(self, fermionic_hamiltonian):
        """
        Do the mapping of a FermionicHamiltonian. First generates the LCPS representation of the creation/annihilation
        operators for the specific mapping. Uses the 'to_linear_combinaison_pauli_string' of the FermionicHamiltonian
        to generate the complete LCPS.

        Args:
            fermionic_hamiltonian (FermionicHamiltonian): A FermionicHamiltonian that provided a 
                'to_linear_combinaison_pauli_string' method.

        Returns:
            LinearCombinaisonPauliString: The LCPS reprenseting the FermionicHamiltonian
        """

        aps, ams = self.fermionic_operator_linear_combinaison_pauli_string(fermionic_hamiltonian.number_of_orbitals())
        pslc = fermionic_hamiltonian.to_linear_combinaison_pauli_string(aps, ams)
        return pslc


class JordanWigner(Mapping):
    def __init__(self):
        """
        The Jordan-Wigner mapping
        """

        self.name = 'jordan-wigner'

    def fermionic_operator_linear_combinaison_pauli_string(self, n_qubits):
        """
        Build the LCPS reprensetations for the creation/annihilation operator for each qubit following 
        Jordan-Wigner mapping.

        Args:
            n_qubits (int): The number of orbitals to be mapped to the same number of qubits.

        Returns:
            list<LinearCombinaisonPauliString>, list<LinearCombinaisonPauliString>: Lists of the creation/annihilation
                operators for each orbital in the form of LinearCombinaisonPauliString.
        """

        aps = list()
        ams = list()

        # Activity 3.1.
        # The creation and annihilation operators are given by two
        # Pauli strings:
        # a_i  = 1/2 (real(P_i)) + i imag(P_i))
        # a_i+ = 1/2 (real(P_i)) - i imag(P_i))
        for i in range(n_qubits):

            coefficients_crea = (0.5, -0.5j)
            coefficients_anih = (0.5, +0.5j)
            real_p_x = np.zeros(n_qubits, dtype=bool)
            imag_p_x = np.zeros(n_qubits, dtype=bool)
            real_p_z = np.zeros(n_qubits, dtype=bool)
            imag_p_z = np.zeros(n_qubits, dtype=bool)

            # From the slides.
            real_p_z[:i] = 1
            real_p_x[i] = 1
            imag_p_z[:i+1] = 1
            imag_p_x[i] = 1

            # In the ZX representation.
            real_string = PauliString(real_p_z, real_p_x)
            imag_string = PauliString(imag_p_z, imag_p_x)
            strings = (real_string, imag_string)

            aps.append(LinearCombinaisonPauliString(
                coefficients_crea, strings))
            ams.append(LinearCombinaisonPauliString(
                coefficients_anih, strings))

        return aps, ams


class Parity(Mapping):
    def __init__(self):
        """
        The Parity mapping
        """

        self.name = 'parity'

    def fermionic_operator_linear_combinaison_pauli_string(self, n_qubits):
        """
        Build the LCPS reprensetations for the creation/annihilation operator for each qubit following 
        Parity mapping.

        Args:
            n_qubits (int): The number of orbtials to be mapped to the same number of qubits.

        Returns:
            list<LinearCombinaisonPauliString>, list<LinearCombinaisonPauliString>: Lists of the creation/annihilation
                operators for each orbital in the form of LinearCombinaisonPauliString
        """

        aps = list()
        ams = list()
        
        ################################################################################################################
        # YOUR CODE HERE
        # OPTIONAL
        ################################################################################################################

        raise NotImplementedError()

        return aps, ams
