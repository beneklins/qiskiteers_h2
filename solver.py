"""
solver.py - Solvers for LinearCombinaisonPauliString

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

        eig_values, eig_vectors = None

        ################################################################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after activity 3.2)
        # Hint : np.linalg.eigh
        ################################################################################################################

        raise NotImplementedError()

        return eig_values, eig_vectors

    def lowest_eig_value(self, lcps):
        """
        Return lowest eigenvalue and associated eigenvector.

        Args:
            lcps (LinearCombinaisonPauliString): The LCPS to be solved.

        Returns:
            float, np.array : The lowest eigenvalue and the associated eigenvector.
        """

        eig_value, eig_vector = None

        ################################################################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after activity 3.2)
        ################################################################################################################

        raise NotImplementedError()

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

        opt_value, opt_params = None

        ################################################################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after activity 3.2)
        ################################################################################################################

        raise NotImplementedError()

        self.last_minimization_duration = time.time()-t0
        
        return opt_value, opt_params

