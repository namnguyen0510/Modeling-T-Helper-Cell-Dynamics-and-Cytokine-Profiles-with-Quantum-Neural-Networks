import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd 
import torch
from torch import nn
from torch import nn, optim
from torch.autograd import Variable
from utils import *
import os


# Set random seeds for reproducibility 2,42
seed = 99
np.random.seed(seed)
torch.manual_seed(seed)

#exp_name = 'Case study 2 \n Cytokine Profile by Cluster'
#exp_name = 'Case study 2 \n T Cell Dynamics by Cluster'
exp_name = 'Case study 3 \n T Cell Dynamics by Patient Group'


model_dirc = '/Users/namnguyen/Documents/0. GAIA-QTech/3. Research/0. Th1_Th2/code_thelpers/model_weights/4-TPE-MPE'
w_path = [os.path.join(model_dirc, x) for x in sorted(os.listdir(model_dirc)) if '.DS_Store' not in x]
print(w_path)

shoots = 1
# Number of qubits and classical bits
num_qubits = 11
# Define the device
qml.drawer.use_style('black_white')
dev = qml.device("default.qubit", wires=num_qubits, shots=shoots)
parameters = np.random.uniform(-2*np.pi,2*np.pi,size = (11,3))
@qml.qnode(dev)
def quantum_circuit(parameters):
    # Step 1: State Initialization (for simplicity, we assume |x(t)> is |00>)
    for i in range(3):
        qml.Hadamard(wires=i)
    # UNITARY TRANSFORMATION OF A+T
    qml.Rot(parameters[0,0],parameters[0,1],parameters[0,2], wires = 0)   #w0
    qml.Rot(parameters[1,0],parameters[1,1],parameters[1,2], wires = 1)   #w1
    qml.Rot(parameters[2,0],parameters[2,1],parameters[2,2], wires = 2)   #w2
    qml.CZ(wires = [0,1])
    qml.CZ(wires = [1,2])
    qml.CZ(wires = [2,0])
    # CONTROL SIGNAL TO CREATE TH1 + TH2 PRODUCTION
    qml.Toffoli(wires = [0,1,3])
    qml.Toffoli(wires = [0,2,4])
    # UNITARY TRANSFORMATION TO PRESENT TH1 VS TH2 DYNAMICS
    qml.Rot(parameters[3,0],parameters[3,1],parameters[3,2], wires = 3)   #w3
    qml.Rot(parameters[4,0],parameters[4,1],parameters[4,2], wires = 4)   #w4
    qml.CZ(wires = [3,4])
    # CONTROL SIGNAL TO CREATE CYTOKINES SIGNALS
    ## FROM TH1
    qml.CNOT(wires = [3,5])
    qml.CNOT(wires = [3,7])
    ## FROM TH2
    qml.CNOT(wires = [4,6])
    qml.CNOT(wires = [4,8])
    # UNITARY TRANSFORMATION TO PRESENT TH1'S CYTOKINES DYNAMICS: IFN-GAMMA AND IL-2
    qml.Rot(parameters[5,0],parameters[5,1],parameters[5,2], wires = 5)   #w5
    qml.Rot(parameters[6,0],parameters[6,1],parameters[6,2], wires = 7)   #w6
    # UNITARY TRANSFORMATION TO PRESENT THW'S CYTOKINES DYNAMICS: IL-10 AND IL-4
    qml.Rot(parameters[7,0],parameters[7,1],parameters[7,2], wires = 6)   #w7
    qml.Rot(parameters[8,0],parameters[8,1],parameters[8,2], wires = 8)   #w8
    ## FEEDBACK PRESENT THE CROSS-INHIBITION EFFECT OF IFN-GAMMA TO TH2 PRODUCTION
    qml.CNOT(wires = [5,1])
    ## FEEDBACK PRESENT THE CROSS-INHIBITION EFFECT OF IL-10 TO TH1 PRODUCTION
    qml.CNOT(wires = [6,2])
    qml.Toffoli(wires = [0,1,9])
    qml.Toffoli(wires = [0,2,10])
    # UNITARY TRANSFORMATION TO PRESENT THW'S CYTOKINES DYNAMICS: IL-10 AND IL-4
    qml.Rot(parameters[9,0],parameters[9,1],parameters[9,2], wires = 9)   #w9
    qml.Rot(parameters[10,0],parameters[10,1],parameters[10,2], wires = 10)  #w10
    qml.CZ(wires = [5,7])
    qml.CZ(wires = [6,8])
    qml.CZ(wires = [8,9])
    qml.CZ(wires = [9,10])
    qml.CZ(wires = [10,6])
    return qml.sample(wires=range(num_qubits))
# Draw the circuit
'''
fig, ax = qml.draw_mpl(quantum_circuit, style = 'rcParams')(parameters)
plt.savefig('quantum_circuit.jpg', dpi = 600)
plt.show()
'''
# Convert the quantum circuit to a Torch model
class QuantumTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = np.random.uniform(-2*np.pi,2*np.pi,size = (11,3))
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32, requires_grad = True))

    def forward(self):
        return quantum_circuit(self.weights)

WEIGHTS = []
for model_path in w_path:
    print(model_path)
    # Instantiate the model
    model = QuantumTorchModel()
    model.load_state_dict(torch.load(model_path))
    # Print model parameters
    print("Initial model parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter '{name}':")
        print(param)
        WEIGHTS.append(param.detach().numpy())

print(WEIGHTS)

from sklearn.decomposition import PCA

# Create a new figure for plotting
plt.figure(figsize = (4,4))
for i in range(len(WEIGHTS)):
    data = WEIGHTS[i]
    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    # Extract the first two principal components
    x_pca = pca_result[:, 0]
    y_pca = pca_result[:, 1]

    # Create a scatter plot
    plt.scatter(x_pca, y_pca, color = plt.cm.Set1(i), marker='o')

# Set labels for the axes
plt.xlabel('Principal Component 1', fontweight = 'bold')
plt.ylabel('Principal Component 2', fontweight = 'bold')
plt.title(f'{exp_name}', fontweight = 'bold')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{exp_name}.jpg', dpi = 600)
# Show the plot
plt.show()