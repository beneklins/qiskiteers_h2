"""
hamiltonian.py - Define Hamiltonian

Copyright © 2021 Brett Henderson <brettrhenderson25@gmail.com>,
                 Igor Benek-Lins <physics@ibeneklins.com>,
                 Melvin Mathews <mel.matt007@gmail.com>,
Copyright © 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>

This file is part of “The Three Qiskiteers H₂ ground state finder” (T3QH2).
For the licence, see the LICENCE file.
"""

import numpy as np
from pauli_string import PauliString, LinearCombinaisonPauliString

class FermionicHamiltonian(object):

    def __str__(self):
        """
        String representation of FermionicHamiltonian.

        Returns:
            str: Description of FermionicHamiltonian.
        """

        out = f'Fermionic Hamiltonian'
        out += f'\nNumber of orbitals : {self.number_of_orbitals():d}'
        out += f'\nIncluding spin : {str(self.with_spin)}'
        return out

    def number_of_orbitals(self):
        """
        Number of orbitals in the state basis.

        Returns:
            int: The number of orbitals in the state basis.
        """

        return self.integrals.shape[0]

    def include_spin(self, order='group_spin'):
        """
        Transforms a spinless FermionicHamiltonian to include spin.
        The transformation doubles the number of orbitals in the basis following the input order.
        Does nothing if the spin is already included (with_spin is True).

        Args:
            order (str, optional): Controls the order of the basis state. Defaults to 'group_spin'.
                With order as 'group_orbital', the integrals will alternate between spin up and down (g_up,g_down,...).
                With order as 'group_spin', the integrals will gather same spin together (g_up,...,g_down,...).

        Raises:
            ValueError: If the order parameter is not one of 'group_spin' or 'group_orbital'.

        Returns:
            FermionicHamiltonian: Including the spin.
        """        

        if self.with_spin:
            print('already with spin')
            return self

        if order == 'group_spin':
            new_integrals = np.kron(self.spin_tensor, self.integrals)
        elif order == 'group_orbital':
            new_integrals = np.kron(self.integrals, self.spin_tensor)
        else:
            raise ValueError("Order should be 'group_spin' or 'group_orbital'.")
        
        return self.__class__(new_integrals, with_spin=True)

    def get_integrals(self, cut_zeros=True, threshold=1e-9):
        """
        Returns the integral tensor with an optional threshold for values close to 0.

        Args:
            cut_zeros (bool, optional): If True, all integral values smaller than 'threshold' will be set to 0.
                                        Defaults to True.
            threshold (float, optional): Value of the threshold. Defaults to 1e-9.

        Returns:
            np.ndarray: The integral tensor.
        """        

        integrals = self.integrals.copy()
        integrals[np.abs(integrals) < threshold] = 0

        return integrals


class OneBodyFermionicHamiltonian(FermionicHamiltonian):
    spin_tensor = np.eye(2)

    def __init__(self, integrals, with_spin=False):
        """
        A FermionicHamiltonian representing a one body term in the form of $sum_i h_{ij} a_i^\dagger a_j$.

        Args:
            integrals (np.ndarray): Square tensor (n*n) containing the integral values.
            with_spin (bool, optional): Does the integral tensor include the spin? Defaults to False.
                Should be False if the integrals are for orbital part only.
                Should be True if the spin is already included in the integrals.

        Raises:
            ValueError: When the dimension of the 'integrals' parameter is not 2.
        """        

        if not(integrals.ndim == 2):
            raise ValueError('Integral tensor should be ndim == 2 for a one-body hamiltonian')

        self.integrals = integrals
        self.with_spin = with_spin

    def change_basis(self, transform):
        """
        Transforms the integrals tensor (n*n) into a new basis.

        Args:
            transform (np.ndarray): Square tensor (n*n) defining the basis change.

        Returns:
            OneBodyFermionicHamiltonian: Transformed Hamiltonian.
        """

        # Activity 2.2.
        new_integrals = np.einsum(
            'mi, nj, mn -> ij',
            transform, transform, self.integrals)

        return OneBodyFermionicHamiltonian(new_integrals, self.with_spin)

    def to_linear_combinaison_pauli_string(self, aps, ams):
        """
        Generates a qubit operator reprensentation (LinearCombinaisonPauliString) of the OneBodyFermionicHamiltonian
        given some creation/annihilation operators.

        Args:
            aps (list<LinearCombinaisonPauliString>): List of the creation operators for each orbital in the form of
                                                    LinearCombinaisonPauliString.
            ams (list<LinearCombinaisonPauliString>): List of the annihilation operators for each orbital in the form of
                                                    LinearCombinaisonPauliString.

        Returns:
            LinearCombinaisonPauliString: Qubit operator reprensentation of the OneBodyFermionicHamiltonian.
        """        

        n_orbs = self.number_of_orbitals()

        # Since each creation/annihilation operator consists of 2 PauliString for each orbital
        # and we compute ap * am, there will be (2*n_orbs)**2 Coefs and PauliStrings.
        new_coefs = np.zeros(((2*n_orbs)**2,), dtype=np.complex)
        new_pauli_strings = np.zeros(((2*n_orbs)**2,), dtype=PauliString)

        lcps = None

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.1)
        # lcps =
        ################################################################################################################

        raise NotImplementedError()

        return lcps


class TwoBodyFermionicHamiltonian(FermionicHamiltonian):
    spin_tensor = np.kron(np.eye(2)[:, None, None, :], np.eye(2)[None, :, :, None])  # physicist notation

    def __init__(self, integrals, with_spin=False):
        """
        A FermionicHamiltonian representing a two body term in the form of
        $sum_i h_{ijkl} a_i^\dagger a_j^\dagger a_k a_l$.

        Args:
            integrals (np.ndarray): Square tensor (n*n) containing the integral values.
            with_spin (bool, optional): Does the integral tensor include the spin? Defaults to False.
                Should be False if the integrals are for orbital part only.
                Should be True if the spin is already included in the integrals.

        Raises:
            ValueError: When the dimension of the 'integrals' parameter is not 4.
        """  

        if not(integrals.ndim == 4):
            raise ValueError('Integral tensor should be ndim == 4 for a two-body hamiltonian')
            
        self.integrals = integrals
        self.with_spin = with_spin

    def change_basis(self, transform):
        """
        Transforms the integrals tensor (n*n*n*n) into a new basis.

        Args:
            transform (np.ndarray): Square tensor (n*n) defining the basis change.

        Returns:
            TwoBodyFermionicHamiltonian: Transformed Hamiltonian.
        """

        # Activity 2.2.
        new_integrals = np.einsum(
            'mi, nj, ok, pl, mnop -> ijkl',
            transform, transform, transform, transform, self.integrals)

        return TwoBodyFermionicHamiltonian(new_integrals, self.with_spin)

    def to_linear_combinaison_pauli_string(self, aps, ams):
        """
        Generates a qubit operator reprensentation (LinearCombinaisonPauliString) of the TwoBodyFermionicHamiltonian
        given some creation/annihilation operators.

        Args:
            aps (list<LinearCombinaisonPauliString>): List of the creation operators for each orbital in the form of
                                                    LinearCombinaisonPauliString.
            ams (list<LinearCombinaisonPauliString>): List of the annihilation operators for each orbital in the form of
                                                    LinearCombinaisonPauliString.

        Returns:
            LinearCombinaisonPauliString: Qubit operator reprensentation of the TwoBodyFermionicHamiltonian.
        """     

        n_orbs = self.number_of_orbitals()
        # Since each creation/annihilation operator consist of 2 PauliString for each orbital
        # and we compute ap * ap * am * am there will be (2*n_orbs)**4 Coefs and PauliStrings
        new_coefs = np.zeros(((2*n_orbs)**4,), dtype=np.complex)
        new_pauli_strings = np.zeros(((2*n_orbs)**4,), dtype=PauliString)

        lcps = None

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.1)
        # lcps =
        ################################################################################################################

        raise NotImplementedError()

        return lcps
        

class MolecularFermionicHamiltonian(FermionicHamiltonian):
    def __init__(self, one_body, two_body, with_spin=False):
        """
        A composite FermionicHamiltonian made of 1 OneBodyFermionicHamiltonian and 1 TwoBodyFermionicHamiltonian.

        Args:
            one_body (OneBodyFermionicHamiltonian): A fermionic Hamiltonian representing a one body term.
            two_body (TwoBodyFermionicHamiltonian): A fermionic Hamiltonian representing a two body term.
            with_spin (bool, optional): Does the integral tensor include the spin? Defaults to False.
                Should be False if the integrals are for orbital part only.
                Should be True if the spin is already included in the integrals.
        """

        if one_body.number_of_orbitals() != two_body.number_of_orbitals():
            raise()

        self.one_body = one_body
        self.two_body = two_body
        self.with_spin = with_spin
    
    @classmethod
    def from_integrals(cls, h1, h2, with_spin=False):
        """
        Generates a MolecularFermionicHamiltonian describing a Molecule from h1 and h2 integral tensors.

        Args:
            h1 (np.ndarray(n,n)): One Body integral tensor
            h2 (np.ndarray(n,n,n,n)): Two Body integral tensor
            with_spin (bool, optional): Does the integral tensor include the spin? Defaults to False.
                Should be False if the integrals are for orbital part only.
                Should be True if the spin is already included in the integrals.

        Returns:
            MolecularFermionicHamiltonian: The Hamiltonian describing the molecule including one OneBody and one
            TwoBody terms.
        """

        one_body = OneBodyFermionicHamiltonian(h1, with_spin)
        two_body = TwoBodyFermionicHamiltonian(h2, with_spin)

        return cls(one_body, two_body, with_spin)

    @classmethod
    def from_pyscf_mol(cls, mol):
        """
        Generates a MolecularFermionicHamiltonian describing a molecule from a pyscf Molecule representation.

        Args:
            mol (pyscf.gto.mole.Mole): Molecule object used to compute different integrals.

        Returns:
            MolecularFermionicHamiltonian: The Hamiltonian describing the Molecule including one OneBody and one
            TwoBody terms.
        """

        # Diagonalisation of the overlap matrix S.
        overlap_matrix = mol.intor('int1e_ovlp')
        # Construct an orthonormal basis
        overlap_matrix_eigval, overlap_matrix_eigvec = (
            np.linalg.eigh(overlap_matrix))
        # Reverse the ordering of eigenvalues.
        overlap_matrix_eig_order = np.argsort(overlap_matrix_eigval)
        overlap_matrix_eigvec = overlap_matrix_eigvec[:, overlap_matrix_eig_order]
        # Linear transformation that takes us from the atomic (ao) to the
        # orthonormal (oo) basis.
        ao2oo = overlap_matrix_eigvec/np.sqrt(overlap_matrix_eigval[None, :])

        # Build h1 and h2 in atomic orbital (AO) basis.
        h1_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        h2_ao = mol.intor('int2e')
        # Transform h1 and h2 into the orthonormal orbital (OO) basis.
        h1_oo = np.einsum(
            'mi, nj, mn -> ij',
            ao2oo, ao2oo, h1_ao)
        h2_oo = np.einsum(
            'mi, nj, ok, pl, mnop -> ijkl',
            ao2oo, ao2oo, ao2oo, ao2oo, h2_ao)

        # Linear transformation that takes us from the orthonormal (oo)
        # to the molecular (mo) basis.
        h1_oo_eigval, h1_oo_eigvec = np.linalg.eig(h1_oo)
        # Reverse the ordering of eigenvalues.
        # TODO: Check if we indeed need to reverse the ordering here.
        h1_oo_eig_order = np.argsort(h1_oo_eigval)
        oo2mo = h1_oo_eigvec[:, h1_oo_eig_order]

        # Transform h1 and h2 into the molecular orbital (MO) basis.
        h1_mo = np.einsum(
            'mi, nj, mn -> ij',
            oo2mo, oo2mo, h1_oo)
        h2_mo = np.einsum(
            'mi, nj, ok, pl, mnop -> ijkl',
            oo2mo, oo2mo, oo2mo, oo2mo, h2_ao)

        # Build the one and two body Hamiltonians
        one_body = OneBodyFermionicHamiltonian(h1_mo)
        two_body = TwoBodyFermionicHamiltonian(h2_mo)

        return cls(one_body, two_body)

    def number_of_orbitals(self):
        """
        Number of orbitals in the state basis.

        Returns:
            int: The number of orbitals in the state basis.
        """ 

        return self.one_body.integrals.shape[0]

    def change_basis(self, transform):
        """
        Transforms the integrals tensors for both sub Hamiltonian.
        See FermionicHamiltonian.change_basis.

        Args:
            transform (np.ndarray): Square tensor (n*n) defining the basis change.

        Returns:
            MolecularFermionicHamiltonian: Transformed Hamiltonian.
        """

        new_one_body = self.one_body.change_basis(transform)
        new_two_body = self.two_body.change_basis(transform)

        return MolecularFermionicHamiltonian(new_one_body, new_two_body, self.with_spin)

    def include_spin(self, order='group_spin'):
        """
        Transforms a spinless FermionicHamiltonian to inlude spin for both sub Hamiltonians.
        See FermionicHamiltonian.include_spin.

        Args:
            order (str, optional): Controls the order of the basis state. Defaults to 'group_spin'.
                With order as 'group_orbital', the integrals will alternate between spin up and down (g_up,g_down,...).
                With order as 'group_spin', the integrals will gather same spin together (g_up,...,g_down,...).

        Raises:
            ValueError: If the order parameter is not one of 'group_spin' or 'group_orbital'.

        Returns:
            FermionicHamiltonian: Including the spin.
        """  

        if self.with_spin:
            print('already with spin')
            return self

        new_one_body = self.one_body.include_spin()
        new_two_body = self.two_body.include_spin()

        return MolecularFermionicHamiltonian(new_one_body, new_two_body, with_spin=True)

    def get_integrals(self, **vargs):
        """
        Return the integral tensors for both sub Hamiltonians with an optional threshold for values close to 0.

        Args:
            cut_zeros (bool, optional): If True, all integral values small than threshold will be set to 0.
                                        Defaults to True.
            threshold (float, optional): Value of the threshold. Defaults to 1e-9.

        Returns:
            np.ndarray, np.ndarray: The integral tensors.
        """ 

        integrals_one = self.one_body.get_integrals(**vargs)
        integrals_two = self.two_body.get_integrals(**vargs)

        return integrals_one, integrals_two

    def to_linear_combinaison_pauli_string(self, aps, ams):
        """
        Generates a qubit operator representation (LinearCombinaisonPauliString) of the MolecularFermionicHamiltonian
        given some creation/annihilation operators.

        Args:
            aps (list<LinearCombinaisonPauliString>): List of the creation operators for each orbital in the form of
                                                    LinearCombinaisonPauliString.
            ams (list<LinearCombinaisonPauliString>): List of the annihilation operators for each orbital in the form of
                                                    LinearCombinaisonPauliString.

        Returns:
            LinearCombinaisonPauliString: Qubit operator reprensentation of the MolecularFermionicHamiltonian.
        """     

        out = None

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.1)
        ################################################################################################################

        raise NotImplementedError()

        return out


