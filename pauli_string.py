"""
pauli_string.py - Define PauliString and LinearCombinaisonPauliString

Copyright © 2021 Brett Henderson <brettrhenderson25@gmail.com>,
                 Igor Benek-Lins <physics@ibeneklins.com>,
                 Melvin Mathews <mel.matt007@gmail.com>,
Copyright © 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>

This file is part of “The Three Qiskiteers H₂ ground state finder” (T3QH2).
For the licence, see the LICENCE file.
"""

import numpy as np


class PauliString(object):

    def __init__(self, z_bits, x_bits):
        """
        Describe a Pauli string as 2 arrays of booleans. The `PauliString` represents `(-1j)**(z_bits*x_bits) Z**z_bits X**x_bits`.

        Parameters
        ----------
        z_bits : np.ndarray<bool>
            True where a Z Pauli is applied.
        x_bits : np.ndarray<bool>
            True where a X Pauli is applied.

        Raises
        ------
        ValueError
            If the number of `z_bits` and `x_bits` are different.
        """

        if len(z_bits) != len(x_bits):
            raise ValueError('z_bits and x_bits must have the same number of elements')

        self.z_bits = z_bits
        self.x_bits = x_bits

    def __str__(self):
        """
        String representation of the `PauliString`.

        Returns
        -------
        str
            String of I, Z, X and Y.
        """

        pauli_labels = 'IZXY'
        pauli_choices = (self.z_bits + 2*self.x_bits).astype(int)
        out = ''
        for i in reversed(pauli_choices):
            out += pauli_labels[i]
        return out

    def __len__(self):
        """
        Number of Pauli in the `PauliString`. It is also the number of qubits.

        Returns
        -------
        int
            Length of the `PauliString`, also number of qubits.
        """

        return len(self.z_bits)

    def __mul__(self, other):
        """
        Allow the use of `*` with other `PauliString` or with a numeric
        coefficient.

        Parameters
        ----------
        other : PauliString/float
            Will compute the product of Pauli strings if `PauliString` or
            computer a linear combination of Pauli strings if `float`.

        Returns
        -------
        PauliString/LinearCombinaisonPauliString
            `PauliString` when other is a PauliString or
            `LinearCombinaisonPauliString` when other is numeric
        """

        if isinstance(other, PauliString):
            return self.mul_pauli_string(other)
        else:
            return self.mul_coef(other)

    def __rmul__(self, other):
        """
        Same as `__mul__`. Allow the use of `*` with a preceding numeric
        coefficient. Example: `1/2 * PauliString`.


        Parameters
        ----------
        other : PauliString/float
            Will compute the product of Pauli strings if `PauliString` or
            computer a linear combination of Pauli strings if `float`.

        Returns
        -------
        PauliString/LinearCombinaisonPauliString
            `PauliString` when other is a PauliString or
            `LinearCombinaisonPauliString` when other is numeric
        """

        return self.__mul__(other)

    @classmethod
    def from_zx_bits(cls, zx_bits):
        """
        Construct a `PauliString` from a single `array<bool>` of length `2n`.

        Parameters
        ----------
        zx_bits : np.array<bool>
            An array of booleans. First `n` bits specify the :math:`Z`s.
            Second half specify the :math:`X`s.

        Returns
        -------
        PauliString
            The Pauli string specified by `zx_bits`.
        """

        z_bits = x_bits = None

        # Activity 3.1.
        z_bits = zx_bits[:len(zx_bits)//2]
        x_bits = zx_bits[len(zx_bits)//2:]

        return cls(z_bits, x_bits)

    @classmethod
    def from_str(cls, pauli_str):
        """
        Construct a PauliString from a string (as returned by `__str__`).

        Parameters
        ----------
        pauli_str : str
            String of length `n` made of :math:`I`, :math:`X`, :math:`Y` and
            :math:`Z`.

        Returns
        -------
        PauliString
            The Pauli string specified by `pauli_str`.
        """

        # Activity 3.1.
        z_bits, x_bits = [], []
        for pauli in pauli_str[::-1]:
            if pauli.upper() == 'X':
                z_bits.append(False)
                x_bits.append(True)
            elif pauli.upper() == 'Y':
                z_bits.append(True)
                x_bits.append(True)
            elif pauli.upper() == 'Z':
                z_bits.append(True)
                x_bits.append(False)
            elif pauli.upper() == 'I':
                z_bits.append(False)
                x_bits.append(False)
            else:
                raise ValueError('Pauli string must only have characters X, Y, Z and I')

        z_bits = np.array(z_bits, dtype=bool)
        x_bits = np.array(x_bits, dtype=bool)

        return cls(z_bits, x_bits)

    def to_zx_bits(self):
        """
        Return the `zx_bits` representation of the `PauliString`. Useful to
        compare `PauliString`s together.

        Returns
        -------
        np.array<bool>
            `zx_bits` representation of the `PauliString` of length `2n`.
        """

        # Activity 3.1.
        zx_bits = np.concatenate((self.z_bits, self.x_bits))

        return zx_bits

    def to_xz_bits(self):
        """
        Return the `xz_bits` representation of the `PauliString`. Useful to
        check commutativity.

        Returns
        -------
        np.array<bool>
            `xz_bits` representation of the `PauliString` of length `2n`.
        """

        # Activity 3.1.
        xz_bits = np.concatenate((self.x_bits, self.z_bits))

        return xz_bits

    def mul_pauli_string(self, other):
        """
        Product with an `other` Pauli string.

        Parameters
        ----------
        other : PauliString
            An other PauliString.

        Raises
        ------
        ValueError: If the other `PauliString` is not of the same length.

        Returns
        -------
        PauliString : complex
            The resulting `PauliString` and the product phase.
        """

        if len(self) != len(other):
            raise ValueError('PauliString must be of the same length')

        # Activity 3.1 using prime numbers!
        # pauli_str = self.__str__()
        # pauli_str_len = len(pauli_str)
        # other_str = other.__str__()
        # pauli_str_array = np.array(list(pauli_str), dtype='U1')
        # other_str_array = np.array(list(other_str), dtype='U1')
        # #
        # dict_spin_matrices = dict(I=1, X=2, Y=3, Z=5)
        # pauli_array = np.array([dict_spin_matrices[i] for i in pauli_str_array])
        # other_array = np.array([dict_spin_matrices[i] for i in other_str_array])
        # #
        # dict_spin_matrices_prod = dict({
        #     1: 'I', 2: 'X', 3: 'Y', 5: 'Z', 6: 'iZ', 15: 'iX', 10: 'iiiY',
        #     4: 'I', 9: 'I', 25: 'I',
        # })
        # extra_phase_array = np.empty(pauli_str_len, dtype='U2')
        # for idx in range(pauli_str_len):
        #     # We are acting on the left.
        #     left_operator, right_operator = other_array[idx], pauli_array[idx]
        #     extra_phase = ''
        #     # Do not add an extra phase if the identity is present.
        #     if 1 not in (left_operator, right_operator):
        #         if left_operator > right_operator:
        #             extra_phase = 'ii' # = -1
        #     extra_phase_array[idx] = extra_phase
        # # We are acting on the left.
        # result = other_array * pauli_array
        # result_str = ''.join([dict_spin_matrices_prod[i] for i in result])
        # result_str = result_str + ''.join(extra_phase_array)
        # winding_number = 0
        # resulting_pauli_str = ''
        # for i in result_str:
        #     if i == 'i':
        #         winding_number += 1
        #     else:
        #         resulting_pauli_str += i
        # phase = (-1j)**winding_number

        # new_pauli_array = np.array(list(resulting_pauli_str), dtype='U1')
        # new_y_bits = (new_pauli_array == 'Y')
        # new_x_bits = (new_pauli_array == 'X') + new_y_bits
        # new_z_bits = (new_pauli_array == 'Z') + new_y_bits

        new_z_bits = new_x_bits = phase = None

        # Activity 3.1.
        # xor will do mod 2 addition to the bitstring.
        new_z_bits = np.logical_xor(self.z_bits, other.z_bits)
        new_x_bits = np.logical_xor(self.x_bits, other.x_bits)
        # 2z2 · x1 + z1 · x1 + z2 · x2 - z3 · x3
        w = (2 * np.sum(np.logical_and(other.z_bits, self.x_bits))
             + np.sum(np.logical_and(self.z_bits, self.x_bits))
             + np.sum(np.logical_and(other.z_bits, other.x_bits))
             - np.sum(np.logical_and(new_z_bits, new_x_bits))) % 4
        phase = (-1j)**w

        return self.__class__(new_z_bits, new_x_bits), phase

    def mul_coef(self, coef):
        """
        Build a LCPS from a `PauliString` (`self`) and a number (`coef`).

        Parameters
        ----------
        coef : int/float/complex
            A numeric coefficient.

        Returns
        -------
        LinearCombinaisonPauliString
            A LCPS with only one `PauliString` and coefficient.
        """

        # Activity 3.1.
        coefs = np.array([coef], dtype=complex)
        pauli_strings = np.array([self], dtype=PauliString)

        return LinearCombinaisonPauliString(coefs, pauli_strings)

    def ids(self):
        """
        Position of identity in the `PauliString`.

        Returns
        -------
        np.array<bool>
            True where both `z_bits` and `x_bits` are `False`.
        """

        # Activity 3.1.
        ids = (self.x_bits | self.z_bits) == False

        return ids

    def copy(self):
        """
        Build a copy of the `PauliString`.

        Returns
        -------
        PauliString
            A copy.
        """

        return PauliString(self.z_bits.copy(), self.x_bits.copy())

    def to_matrix(self):
        """
        Build the matrix representation of the `PauliString` using the
        Kronecker product.

        Returns
        -------
        np.array<complex>
            A :math:`2^n` side square matrix.
        """

        I_MAT = np.array([[1, 0],[0, 1]])
        X_MAT = np.array([[0, 1],[1, 0]])
        Y_MAT = np.array([[0, -1j],[1j, 0]])
        Z_MAT = np.array([[1, 0],[0, -1]])

        # Activity 3.1 (optional).
        paulis = [I_MAT, Z_MAT, X_MAT, Y_MAT]
        matrix = np.ones((1,1),dtype = np.complex)

        for pauli in reversed((self.z_bits + 2*self.x_bits).astype(int)):
            matrix = np.kron(matrix, paulis[pauli])

        return matrix


class LinearCombinaisonPauliString(object):
    def __init__(self,coefs,pauli_strings):
        """
        Describes a linear combination of Pauli Strings.

        Parameters
        ----------
        coefs : np.array
            Coefficients multiplying the respective `PauliStrings`.
        pauli_strings : np.array<PauliString>
            PauliStrings.

        Raises
        ------
        ValueError
            If the number of coefficients is different from the number of
            `PauliStrings`.
        ValueError
            If all `PauliStrings` are not of the same length.
        """

        if len(coefs) != len(pauli_strings):
            raise ValueError('Must provide a equal number of coefs and PauliString')

        n_qubits = len(pauli_strings[0])
        for pauli in pauli_strings:
            if len(pauli) != n_qubits:
                raise ValueError('All PauliString must be of same length')

        self.n_terms = len(pauli_strings)
        self.n_qubits = len(pauli_strings[0])

        self.coefs = np.array(coefs, dtype=complex)
        self.pauli_strings = np.array(pauli_strings, dtype=PauliString)

    def __str__(self):
        """
        String representation of the `LinearCombinaisonPauliString`.

        Returns
        -------
        str
            Descriptive string.
        """

        out = f'{self.n_terms:d} pauli strings for {self.n_qubits:d} qubits (Real, Imaginary)'
        for coef, pauli in zip(self.coefs,self.pauli_strings):
            out += '\n' + f'{str(pauli)} ({np.real(coef):+.5f},{np.imag(coef):+.5f})'
        return out

    def __getitem__(self, key):
        """
        Return a subset of the `LinearCombinaisonPauliString` array-like.

        Parameters
        ----------
        key : int or slice
            Elements to be returned.

        Returns
        -------
        LinearCombinaisonPauliString
            LCPS with the element specified in key.
        """

        if isinstance(key,slice):
            new_coefs = np.array(self.coefs[key])
            new_pauli_strings = self.pauli_strings[key]
        else:
            if isinstance(key,int):
                key = [key]
            new_coefs = self.coefs[key]
            new_pauli_strings = self.pauli_strings[key]

        return self.__class__(new_coefs,new_pauli_strings)

    def __len__(self):
        """
        Number of `PauliStrings` in the LCPS.

        Returns
        -------
        int
            Number of `PauliStrings`/coefficients.
        """

        return len(self.pauli_strings)

    def __add__(self,other):
        """
        Allow the use of `+` to add two LCPS together.

        Parameters
        ----------
        other : LinearCombinaisonPauliString
            An other LCPS.

        Returns
        -------
        LinearCombinaisonPauliString
            New LCPS of length `len(self) + len(other)`.
        """

        return self.add_pauli_string_linear_combinaison(other)

    def __mul__(self, other):
        """
        Allow the use of `*` with other LCPS or numeric values.

        Parameters
        ----------
        other : LinearCombinaisonPauliString
            An other LCPS

        Returns
        -------
        LinearCombinaisonPauliString/LinearCombinaisonPauliString
            New LCPS of length `len(self) * len(other)` or a new LCPS of
            same length with modified coefficients.
        """

        if isinstance(other,LinearCombinaisonPauliString):
            return self.mul_linear_combinaison_pauli_string(other)
        else:
            return self.mul_coef(other)

    def __rmul__(self, other):
        """
        Same as `__mul__`. Allow the use of `*` with a preceding numeric
        coefficient. Example: `1/2 * PauliString`.

        Parameters
        ----------
        other : LinearCombinaisonPauliString
            An other LCPS.

        Returns
        -------
        LinearCombinaisonPauliString/LinearCombinaisonPauliString
            New LCPS of length `len(self) * len(other)` or a new LCPS of
            same length with modified coefficients.
        """

        return self.__mul__(other)

    def add_pauli_string_linear_combinaison(self, other):
        """
        Adding with an other LCPS. Merging the coefficients and
        `PauliStrings` arrays.

        Parameters
        ----------
        other : LinearCombinaisonPauliString
            An other LCPS.

        Raises
        ------
        ValueError
            If `other` is not an LCPS.
        ValueError
            If the other LCPS has not the same number of qubits.

        Returns
        -------
        LinearCombinaisonPauliString
            New LCPS of length `len(self) + len(other)`.
        """

        if not isinstance(other,LinearCombinaisonPauliString):
            raise ValueError('Can only add with an other LCPS')

        if self.n_qubits != other.n_qubits:
            raise ValueError('Can only add with LCPS of identical number of qubits')

        # Activity 3.1.
        new_coefs = np.concatenate((self.coefs, other.coefs))
        new_pauli_strings = np.concatenate((self.pauli_strings, other.pauli_strings))

        return self.__class__(new_coefs, new_pauli_strings)

    def mul_linear_combinaison_pauli_string(self, other):
        """
        Multiply with an other LCPS.

        Parameters
        ----------
        other : LinearCombinaisonPauliString
            An other LCPS.

        Raises
        ------
        ValueError
            If `other` is not an LCPS.
        ValueError
            If the other LCPS has not the same number of qubits.

        Returns
        -------
        LinearCombinaisonPauliString
            New LCPS of length `len(self) * len(other)`.
        """

        if not isinstance(other, LinearCombinaisonPauliString):
            raise ValueError()

        if self.n_qubits != other.n_qubits:
            raise ValueError('Can only add with LCPS of identical number of qubits')

        new_coefs = np.zeros((len(self)*len(other),), dtype=np.complex)
        new_pauli_strings = np.zeros((len(self)*len(other),), dtype=PauliString)

        # Activity 3.1.
        for i in range(self.n_terms):
            for j in range(other.n_terms):
                new_pauli_strings[i * other.n_terms + j], phase = self.pauli_strings[i] * other.pauli_strings[j]
                new_coefs[i * other.n_terms + j] = self.coefs[i] * other.coefs[j] * phase

        return self.__class__(new_coefs, new_pauli_strings)

    def mul_coef(self,other):
        """
        Multiply the LCPS by a numeric coefficient or an array of the
        same length.

        Parameters
        ----------
        other : float/complex/np.array
            One numeric factor or one factor per `PauliString`.

        Raises
        ------
        ValueError
            If `other` is `np.array`, it should be of the same length as
            the LCPS.

        Returns
        -------
        LinearCombinaisonPauliString
            New LCPS properly multiplied by the coefficients.
        """

        # Activity 3.1.
        if isinstance(other, np.ndarray):
            if len(other) != self.__len__():
                raise ValueError("Array length does not equal length of LCPS!")
        new_coefs = self.coefs * other
        new_pauli_strings = self.pauli_strings

        return self.__class__(new_coefs, new_pauli_strings)

    def to_zx_bits(self):
        """
        Build an array that contains all the `zx_bits` for each `PauliString`.

        Returns
        -------
        np.array<bool>
            A two-dimensional array of booleans where each line is the
            `zx_bits` of a `PauliString`.
        """

        # Activity 3.1.
        zx_bits = np.zeros((len(self), 2*self.n_qubits), dtype=np.bool)
        for i, ps in enumerate(self.pauli_strings):
            zx_bits[i, :] = ps.to_zx_bits()

        return zx_bits

    def to_xz_bits(self):
        """
        Build an array that contains all the `xz_bits` for each `PauliString`.

        Returns
        -------
        np.array<bool>
            A two-dimensional array of booleans where each line is the
            `xz_bits` of a `PauliString`.
        """

        xz_bits = np.zeros((len(self), 2*self.n_qubits), dtype=np.bool)

        # Activity 3.1.
        for i, ps in enumerate(self.pauli_strings):
            xz_bits[i, :] = ps.to_xz_bits()

        return xz_bits

    def ids(self):
        """
        Build an array that identifies the position of all the :math:`I`
        for each PauliString.

        Returns
        -------
        np.array<bool>
            A two-dimensional array of booleans where each line is the
            `xz_bits` of a `PauliString`.
        """

        ids = np.zeros((len(self), self.n_qubits), dtype=np.bool)

        # Activity 3.1.
        for i, ps in enumerate(self.pauli_strings):
            ids[i, :] = ps.ids()

        return ids

    def combine(self):
        """
        Finds unique `PauliStrings` in the LCPS and combines the
        coefficients of identical `PauliStrings`. Reduces the length of
        the LCPS.

        Returns
        -------
        LinearCombinaisonPauliString
            LCPS with combined coefficients.
        """

        # Activity 3.1.
        old_pauli_strings = self.to_zx_bits()
        new_pauli_strings_zx = np.unique(old_pauli_strings, axis=0)
        new_pauli_strings = np.array(
            [PauliString.from_zx_bits(zx)
             for zx in new_pauli_strings_zx], dtype=PauliString)
        new_coefs = np.array(
            [sum(self.coefs[(old_pauli_strings==u).all(axis=1)])
             for u in new_pauli_strings_zx])

        return self.__class__(new_coefs, new_pauli_strings)

    def apply_threshold(self, threshold=1e-6):
        """
        Remove `PauliStrings` with coefficients smaller then threshold.

        Parameters
        ----------
        threshold : float, optional, default=1e-6
            `PauliStrings` with coefficients smaller than `threshold`
            will be removed.

        Returns
        -------
        LinearCombinaisonPauliString
            LCPS without coefficients smaller then threshold.
        """

        # Activity 3.1.
        new_lcps = self[np.abs(self.coefs) >= threshold]

        return new_lcps

    def divide_in_bitwise_commuting_cliques(self):
        """
        Find bitwise commuting cliques in the LCPS.

        Returns
        -------
        list<LinearCombinaisonPauliString>
            List of LCPS where all elements of one LCPS bitwise commute
            with each other.
        """

        cliques = list()

        #################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.2)
        # This one can be hard to implement
        # Use to_zx_bits
        # Transform all I into Z and look for unique PauliStrings
        #################################################################

        raise NotImplementedError()

        return cliques

    def sort(self):
        """
        Sort the `PauliStrings` by order of the `zx_bits`.

        Returns
        -------
        LinearCombinaisonPauliString
            Sorted.
        """

        # Activity 3.1.
        order = np.argsort(
            [''.join((1*np.flip(bitstring)).astype(str))
             for bitstring in self.to_zx_bits()])
        #order = np.argsort([str(ps) for ps in self.pauli_strings])

        new_coefs = self.coefs[order]
        new_pauli_strings = self.pauli_strings[order]

        return self.__class__(new_coefs, new_pauli_strings)

    def to_matrix(self):
        """
        Build the total matrix representation of the LCPS.

        Returns
        -------
        np.array<complex>
            A :math:`2**n` side square matrix.
        """

        size = 2**self.n_qubits
        matrix = np.zeros((size, size), dtype=np.complex)

        # Activity 3.1.
        for i in range(self.n_terms):
            matrix += self.coefs[i]*self.pauli_strings[i].to_matrix()

        return matrix
