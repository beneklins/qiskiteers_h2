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
        """Exact solver to compute the expectation value of a given operator in the form of a
        LinearCombinaisonPauliString
        """
        
        self.last_eig_value = None
        self.last_eig_vector = None

    def eig(self, lcps):
        """
        Convert LCPS into a matrix and return sorted eigenvalues and eigenvectors.

        Args:
            lcps (LinearCombinaisonPauliString): The LCPS to be solved.

        Returns:
            np.array, np.array : Eigenvalues and eigenvectors sorted with respect to the eigenvalues.
        """

        eig_values = eig_vectors = None

        ################################################################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after activity 3.2)
        # Hint : np.linalg.eigh
        ################################################################################################################

        hamiltonian_matrix = lcps.to_matrix()
        
        # eigh returns eigenvalues in ascending order, with corresponding eigenvectors
        eig_values, eig_vectors = np.linalg.eigh(hamiltonian_matrix)

        return eig_values, eig_vectors

    def lowest_eig_value(self, lcps):
        """
        Return lowest eigenvalue and associated eigenvector.

        Args:
            lcps (LinearCombinaisonPauliString): The LCPS to be solved.

        Returns:
            float, np.array : The lowest eigenvalue and the associated eigenvector.
        """

        eig_value = eig_vector = None

        ################################################################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after activity 3.2)
        ################################################################################################################
        eig_values, eig_vectors = self.eig(lcps)
        eig_value, eig_vector = eig_values[0], eig_vectors[:,0]

        return eig_value, eig_vector


class VQESolver(LCPSSolver):

    def __init__(self, evaluator, minimizer, start_params, name='vqe_solver'):
        """
        Solver based on the VQE algorithm to estimate the lowest expectation value of a given operator in the form of a
        LinearCombinaisonPauliString

        Args:
            evaluator (Evaluator): The Evaluator that allows to transform a LCPS into a function.
            minimizer (lambda): Function that accepts 2 inputs : a function and a starting parameter list/array.
            start_params (list or np.array): The starting parameter to the minimization process.
            name (str, optional): A name to the solver. Useful when testing multiple solver in parallel.
                Defaults to 'vqe_solver'.
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
        Return lowest expectation value and associated parameters the minimization could find.

        Args:
            lcps (LinearCombinaisonPauliString): The LCPS to be solved.

        Returns:
            float, np.array : The lowest eigenvalue and the associated parameters.
        """

        t0 = time.time()

        opt_value = opt_params = None

        ################################################################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after activity 3.2)
        ################################################################################################################

        self.evaluator.set_linear_combinaison_pauli_string(lcps)
        result = self.minimizer(self.evaluator.eval, self.start_params)
        opt_value, opt_params = result.fun, result.x

        self.last_minimization_duration = time.time()-t0
        
        return opt_value, opt_params

