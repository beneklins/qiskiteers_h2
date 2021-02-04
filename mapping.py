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
        Do the mapping of a `FermionicHamiltonian`. First generates the LCPS
        representation of the creation/annihilation operators for the specific
        mapping. Uses the `to_linear_combinaison_pauli_string` of the
        `FermionicHamiltonian` to generate the complete LCPS.

        Parameters
        ----------
        fermionic_hamiltonian : FermionicHamiltonian
            A `FermionicHamiltonian` that provided
            `to_linear_combinaison_pauli_string` method.

        Returns
        -------
        LinearCombinaisonPauliString
            The LCPS representing the `FermionicHamiltonian`.
        """

        aps, ams = self.fermionic_operator_linear_combinaison_pauli_string(
            fermionic_hamiltonian.number_of_orbitals())
        lcps = fermionic_hamiltonian.to_linear_combinaison_pauli_string(
            aps, ams)

        return lcps


class JordanWigner(Mapping):
    def __init__(self):
        """
        The Jordan-Wigner mapping.
        """

        self.name = 'jordan-wigner'

    def fermionic_operator_linear_combinaison_pauli_string(self, n_qubits):
        """
        Build the LCPS representations for the creation/annihilation operator
        for each qubit following Jordan-Wigner mapping.

        Parameters
        ----------
        n_qubits : int
            The number of orbitals to be mapped to the same number of qubits.

        Returns
        -------
        list<LinearCombinaisonPauliString>, list<LinearCombinaisonPauliString>
            Lists of the creation/annihilation operators for each orbital in
            the form of `LinearCombinaisonPauliString`.
        """

        aps = list()
        ams = list()

        # Activity 3.1.
        for i in range(n_qubits):
            pauli_string_x_bits = np.zeros(n_qubits, dtype=bool)
            pauli_string_r_z_bits = np.zeros(n_qubits, dtype=bool)
            pauli_string_i_z_bits = np.zeros(n_qubits, dtype=bool)
            pauli_string_r_z_bits[:i] = 1
            pauli_string_x_bits[i] = 1
            pauli_string_i_z_bits[:i+1] = 1

            aps.append(
                0.5 * PauliString(pauli_string_r_z_bits, pauli_string_x_bits)
                + -0.5j * PauliString(pauli_string_i_z_bits,
                                      pauli_string_x_bits))
            ams.append(
                0.5 * PauliString(pauli_string_r_z_bits, pauli_string_x_bits)
                + 0.5j * PauliString(pauli_string_i_z_bits,
                                     pauli_string_x_bits))

        return aps, ams


class Parity(Mapping):
    def __init__(self):
        """
        The parity mapping.
        """

        self.name = 'parity'

    def fermionic_operator_linear_combinaison_pauli_string(self, n_qubits):
        """
        Build the LCPS representations for the creation/annihilation operator
        for each qubit following the parity mapping.

        Parameters
        ----------
        n_qubits : int
            The number of orbitals to be mapped to the same number of qubits.

        Returns
        -------
        list<LinearCombinaisonPauliString>, list<LinearCombinaisonPauliString>
            Lists of the creation/annihilation operators for each orbital in
            the form of `LinearCombinaisonPauliString`.
        """

        aps = list()
        ams = list()

        #################################################################
        # YOUR CODE HERE
        # OPTIONAL
        #################################################################

        raise NotImplementedError()

        return aps, ams
