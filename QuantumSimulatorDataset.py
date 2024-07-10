import numpy as np
import pylatexenc
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info import Operator
from qiskit.visualization import circuit_drawer
import torch.nn.functional as F
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

# returns two random numbers within a range defined by n in a circular mannar
def choose_two_random_qubits(n):
  a = np.random.randint(0, n)
  b = (a+1) % n
  return a,b

def random_conservation_unitary():
    # Initialize a 4x4 complex matrix
    unitary_matrix = np.zeros((4, 4), dtype=complex)
    # Generate a random 2x2 unitary matrix
    inside = random_unitary(2)
    # Insert the 2x2 unitary matrix in the center of the 4x4 matrix
    unitary_matrix[1:3, 1:3] = inside
    # Set random phases on the diagonal elements
    unitary_matrix[0, 0] = np.exp(-1j * np.random.rand() * 2 * np.pi)
    unitary_matrix[3, 3] = np.exp(-1j * np.random.rand() * 2 * np.pi)
    return Operator(unitary_matrix)

def save_circuit_params(params, file_name):
  with open(directory_path + file_name, 'wb') as file:
    pickle.dump(params, file)

def generate_circuit_params(circuit_length = None, num_qbits = None, file_name = None):
  #length is a number, operators is a list of qiskit Operator, qbits are list of tuples
  if file_name is not None:
    with open(file_name, 'rb') as f:
      params = pickle.load(f)
  else:
    params = {'len': circuit_length, 'num_qbits': num_qbits, 'operators': [], 'qbits': []}
    for i in range(circuit_length):
      params['operators'].append(random_conservation_unitary())
      params['qbits'].append(choose_two_random_qubits(num_qbits))
  return params

def generate_circuit_from_params(params):
    n = params['num_qbits']
    num_gates = params['len']

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(n, 2 * n)

    # Create uniform superposition
    circuit.h(range(n))
    circuit.barrier()

    # this makes the gpu implementation really slow, but i can’t think of any other way to run qiskit with batched initial state
    # possible solution:
    # initial_state = params.get('initial_state', None)
    # if initial_state:
    #    circuit.append(initial_state, range(n))

    # Measure the initial superposition state
    circuit.measure(range(n), range(n, 2 * n))
    circuit.barrier()

    # Apply gates from params['operators'] to specified qubits
    for i in range(num_gates):
        operator = params['operators'][i]
        qubits = params['qbits'][i]
        circuit.append(operator, qubits)

    # Final measurement to map quantum state to classical bits
    circuit.measure(range(n), range(n))

    # Draw the circuit for visualization (optional)
    #circuit.draw(output='mpl')
    circuit.draw(output='text', filename='temp.txt', scale=0.5)
    img = circuit_drawer(circuit, output='mpl')
    img.savefig('circuit_diagram.png')
    return circuit

def get_simulator_circuit(circuit):
  simulator = Aer.get_backend('aer_simulator_statevector')
  #simulator.set options(device=’GPU’)
  compiled_circuit = transpile(circuit, simulator)
  return compiled_circuit, simulator

def run_batch(circuit, compiled_circuit, simulator, batch_size):
  job = simulator.run(compiled_circuit, shots=batch_size, memory=True)
  #job = execute(circuit, simulator, shots = batch size)
  result = job.result().get_memory(circuit)
  results = np.array([list(sample) for sample in result], dtype=float)
  n = results.shape[1]//2
  initial_state = results[:, :n]
  final_state = results[:, n:]
  return initial_state, final_state

class QuantumSimulationDataset(Dataset):
  def __init__(self, parameters, batch_size):
    self.batch = batch_size
    self.circuit = generate_circuit_from_params(parameters)
    self.compiled_circuit, self.simulator = get_simulator_circuit(self.circuit)

  def __len__(self):
    return np.iinfo(np.int32).max

  def __getitem__(self, _):
    initial_states, final_states = run_batch(self.circuit, self.compiled_circuit, self.simulator, self.batch)
    return torch.Tensor(initial_states), torch.Tensor(final_states)

# this allows for fast matrix−matrix products for small systems
def generate_dense_unitary_circuit_from_params(params):
  n = params['num_qbits']
  num_gates = params['len']
  circuit = QuantumCircuit(n, 0)
  for i in range(num_gates):
    operator = params['operators'][i]
    qubits = params['qbits'][i]
    circuit.append(operator, qubits)
  circuit.draw(output='text', filename='temp.txt', scale=0.5)
  op = Operator(circuit)
  return torch.tensor(op.to_matrix(), dtype=torch.cfloat)

def get_final_state_vector(batched_initial_state, op_mat):
  sampled_final_state = torch.matmul(op_mat.unsqueeze(0),batched_initial_state.unsqueeze(2))
  probs = sampled_final_state * sampled_final_state.conj()
  return torch.real(probs)

def sample_from_state_vector(probs, num_qbits, num_samples):
  decimal = torch.multinomial(probs.squeeze(-1), num_samples, replacement=True)
  mask = 2 ** torch.arange(num_qbits).to(decimal.device, decimal.dtype)
  return decimal.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bitstring_from_initial_state(initial_state, num_qbits):
  decimal = torch.argmax(initial_state, dim = -2)
  mask = 2 ** torch.arange(num_qbits).to(decimal.device, decimal.dtype)
  return decimal.unsqueeze(-1).bitwise and(mask).ne(0).float().squeeze(1)

def decimal_from_initial_state(initial_state, num_qbits):
  mask = 2 ** torch.arange(num_qbits).to(initial_state.device, torch.int64)
  return torch.sum( (mask * initial_state) , -1)

@torch.no_grad()
def run_batch_dense_torch(unitary_circuit, num_qbits, batch_size, num_samples, device, inverse_density = 3):
  bits = torch.zeros((batch_size, num_qbits), dtype=torch.bool, device=device)
  x = np.arange(num_qbits)
  rng = np.random.default_rng()
  perms = rng.permuted(np.tile(x, batch_size).reshape(batch_size, x.size), axis=1)[:, :num_qbits//inverse_density]
  bits[np.arange(batch_size).reshape(batch_size, 1), perms] = True
  decimal = decimal_from_initial_state(bits, num_qbits)
  one_hot = F.one_hot(decimal, num_classes=2**num_qbits)
  final_state = get_final_state_vector(one_hot.cfloat(), unitary_circuit)
  sample = sample_from_state_vector(final_state, num_qbits, num_samples)
  return bits.float(), sample

class QuantumSimulationDatasetFast(Dataset):
  def __init__(self, parameters, batch_size, num_final_states_per_initial_state, device, inverse_density):
    self.device = device
    self.batch_size = batch_size // num_final_states_per_initial_state
    self.num_samples = num_final_states_per_initial_state
    self.num_qubits = parameters['num_qbits']
    self.mat = generate_dense_unitary_circuit_from_params(parameters).to(device)#.to sparse()
    self.inverse_density = inverse_density

  def __len__(self):
    return np.iinfo(np.int32).max

  def __getitem__(self, _):
    initial_states, final_states = run_batch_dense_torch(self.mat, self.num_qubits, self.batch_size, self.num_samples, self.device, self.inverse_density)
    initial_states = initial_states.unsqueeze(1).expand_as(final_states).clone()
    return initial_states.flatten(end_dim=-2), final_states.flatten(end_dim=-2)

if __name__ == '__main__':
  params = generate_circuit_params(12, 12)
  params['qbits'] = [(0,1),(2,3),(4,5),(6,7), (8,9), (10,11), (1,2),(3,4),(5,6),(7,8),(9,10), (11,0)]
  # (0,1),(2,3),(4,5),(6,7), (8,9), (10,11), (1,2),(3,4),(5,6),(7,8),(9,10), (11,0)]"
  # (0,1),(2,3),(4,5),(6,7), (8,9), (10,11), (1,2),(3,4),(5,6),(7,8),(9,10), (11,0),
  # (0,1),(2,3),(4,5),(6,7), (8,9), (10,11), (1,2),(3,4),(5,6),(7,8),(9,10), (11,0)]

  #save_circuit_params(params, 'dense_small.param')

  params = generate_circuit_params(file_name = None)

  # If GPU available, use below line
  dataset = QuantumSimulationDatasetFast(params, 64,4, device = 'cuda', inverse_density=3)
  #dataset = QuantumSimulationDataset(params, 64)


  for i, (initial, final) in enumerate(dataset):
    print(i, initial[0], final[0])


