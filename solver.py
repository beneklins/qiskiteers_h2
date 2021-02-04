"""
solver.py - Solvers for LinearCombinaisonPauliString

Copyright © 2021 Brett Henderson <brettrhenderson25@gmail.com>,
                 Igor Benek-Lins <physics@ibeneklins.com>,
                 Melvin Mathews <mel.matt007@gmail.com>,
Copyright © 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>

This file is part of “The Three Qiskiteers H₂ ground state finder” (T3QH2).
For the licence, see the LICENCE file.
"""

import numpy as np
import time


class LCPSSolver(object):
    pass


class ExactSolver(LCPSSolver):
    def __init__(self):
        """
        Exact solver to compute the expectation value of a given operator
        in the form of a `LinearCombinaisonPauliString`.
        """

        self.last_eig_value = None
        self.last_eig_vector = None

    def eig(self, lcps):
        """
        Convert LCPS into a matrix and return sorted eigenvalues and
        eigenvectors.

        Parameters
        ----------
        lcps : LinearCombinaisonPauliString
        	The LCPS to be solved.

        Returns
        -------
        np.array, np.array
        	Eigenvalues and eigenvectors sorted with respect to the
        	eigenvalues.
        """

        # Activity 3.2 (optional).
        hamiltonian_matrix_repr = lcps.to_matrix()
        # Order the eigenvalues in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix_repr)
        eigenorder = np.argsort(eigenvalues)
        eig_values = eigenvalues[eigenorder]
        eig_vectors = eigenvectors[:, eigenorder]

        return eig_values, eig_vectors

    def lowest_eig_value(self, lcps):
        """
        Return lowest eigenvalue and the associated eigenvector.

        Parameters
        ----------
        lcps : LinearCombinaisonPauliString
        	The LCPS to be solved.

        Returns
        -------
        float, np.array
        	The lowest eigenvalue and the associated eigenvector.
        """

        # Activity 3.2 (optional).
        eigenvalues, eigenvectors = self.eig(lcps)
        # # Order the eigenvalues in ascending order.
        # eigenorder = np.argsort(eigenvalues)
        # eigenvalues = eigenvalues[eigenorder]
        # eigenvectors = eigenvectors[:, eigenorder]
        eig_value = eigenvalues[0]
        eig_vector = eigenvectors[:, 0]

        return eig_value, eig_vector


class VQESolver(LCPSSolver):

    def __init__(self, evaluator, minimizer, start_params, name='vqe_solver'):
        """
        Solver based on the VQE algorithm to estimate the lowest
        expectation value of a given operator in the form of a
        `LinearCombinaisonPauliString`.

        Parameters
        ----------
        evaluator : Evaluator
        	The Evaluator that allows to transform a LCPS into a function.
        minimizer : lambda
        	Function that accepts 2 inputs: a function and a starting
        	parameter list/array.
        start_params : list/np.array
        	The starting parameter to the minimization process.
        name : str, optional, default='vqe_solver'
        	A name to the solver. Useful when testing multiple solver in
        	parallel.
        """

        self.evaluator = evaluator
        self.minimizer = minimizer

        self.start_params = start_params

        self.name = name

        self.last_minimization_duration = 0
        self.last_result = None
        self.last_opt_value = None
        self.last_opt_params = None

    def lowest_eig_value(self, lcps):
        """
        Return lowest expectation value and associated parameters the
        minimization could find.

        Parameters
        ----------
        lcps : LinearCombinaisonPauliString
        	The LCPS to be solved.

        Returns
        -------
        float, np.array
        	The lowest eigenvalue and the associated parameters.
        """

        t0 = time.time()

        # Activity 3.2: Brett.
        self.evaluator.set_linear_combinaison_pauli_string(lcps)
        result = self.minimizer(self.evaluator.eval, self.start_params)
        opt_value, opt_params = result.fun, result.x
        self.last_minimization_duration = time.time()-t0

        return opt_value, opt_params
