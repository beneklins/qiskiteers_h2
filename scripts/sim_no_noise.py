import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter
from evaluator import Evaluator
from pauli_string import PauliString
from evaluator import BasicEvaluator
from qiskit import Aer, execute
from hamiltonian import MolecularFermionicHamiltonian
from mapping import JordanWigner
from scipy.optimize import minimize
from solver import VQESolver

# create variational circuit
varform_4qubits_1param = QuantumCircuit(4)
a = Parameter('a')
varform_4qubits_1param.ry(a, 1)
varform_4qubits_1param.x(0)
varform_4qubits_1param.cx(1, 0)
varform_4qubits_1param.cx(0, 2)
varform_4qubits_1param.cx(1, 3)

# H2 at ~ the equilibrium bond length (0.735 Angstrom)
with open('Integrals_sto-3g_H2_d_0.7350_no_spin.npz', 'rb') as f:
    out = np.load(f)
    h1_load_no_spin = out['h1']
    h2_load_no_spin = out['h2']
molecular_hamiltonian = MolecularFermionicHamiltonian.from_integrals(h1_load_no_spin, h2_load_no_spin).include_spin()

# map to the quantum computer
mapping = JordanWigner()
lcps_h2 = mapping.fermionic_hamiltonian_to_linear_combinaison_pauli_string(
    molecular_hamiltonian).combine().apply_threshold().sort()

# Run the Simulator with no noise a bunch of times
qasm_simulator = Aer.get_backend('qasm_simulator')

# Minimizer for the Variational Circuit parameter
minimizer = lambda fct, start_param_values: minimize(
    fct,
    start_param_values,
    method='SLSQP',
    options={'maxiter': 5, 'eps': 1e-1, 'ftol': 1e-4})

def get_energies(N, shots):
    execute_opts = {'shots' : shots}
    evaluator = BasicEvaluator(varform_4qubits_1param, qasm_simulator, execute_opts=execute_opts, measure_filter=None)
    vqe_solver = VQESolver(evaluator, minimizer, [0,], name='vqe_solver')

    electronic_energies = np.zeros(N)
    for i in range(N):
        electronic_energy, _ = vqe_solver.lowest_eig_value(lcps_h2)
        electronic_energies[i] = electronic_energy
        print(f"Mean electronic energy after {i+1} simulations: {electronic_energies[:i+1].mean()}                ", end='\r')
    return electronic_energies
    print("Done!                                                                                                              \n ")

    # print("Electronic Energies:")
    # print(electronic_energies)
    # print(f"Mean Electronic Energy Calculated from Noiseless Simulator, 1024 shots, {N} Times: {electronic_energies.mean()} Eh")
    # print(f"Standard Deviation : {electronic_energies.std()} Eh")

if __name__ == "__main__":
    N = 50
    exact_e = -1.8572750302023788
    for shots in [100, 500, 1024, 4096, 8192]:
        print(f'Running with {shots} shots...')
        electronic_energies = get_energies(N, shots)
        with open(f'sim_no_noise_{shots}_shots.npz', 'wb') as f:
            np.savez(f, distance=0.735,
                     electronic_energies=electronic_energies,
                     exact_electronic_energy=exact_e,
                     shots=shots,
                     number_of_sims=N,
                     mean_energy=electronic_energies.mean(),
                     std_energy=electronic_energies.std(),
                     error=np.abs((electronic_energies.mean() - exact_e) / exact_e))