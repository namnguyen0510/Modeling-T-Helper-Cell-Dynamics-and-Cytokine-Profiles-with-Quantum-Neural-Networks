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

# Set random seeds for reproducibility 2,42
seed = 44
np.random.seed(seed)
torch.manual_seed(seed)

exp_name = '5.2_' 
group_name = 'MPE'
data = pd.read_csv('clinical_datasets/5. zeng.csv')
data = data[data['Group'] == group_name]
y_true = data.iloc[:,3].to_numpy().astype('float')
y_true = MinMaxNormalization(y_true)
print(data)
print(y_true)

#br



shoots = 365
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

# Instantiate the model
model = QuantumTorchModel()
# Print model parameters
print("Initial model parameters:")
for name, param in model.named_parameters():
    print(f"Parameter '{name}':")
    print(param)


# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay = 1e-3)
criterion = nn.MSELoss()

# Training loop
epochs = 1000
best_loss = 10e6
for epoch in range(epochs):
    optimizer.zero_grad()  # Zero gradients
    # SAMPLING
    samples = model()
    # DATA CALIBRATION
    samples = 2*samples-1
    # DATA READOUT FROM QUANTUM CIRCUIT
    Th1 = samples[:,3]
    Th2 = samples[:,4]
    Igamma = samples[:,5]
    I10 = samples[:,6]
    I2 = samples[:,7] + samples[:,9]
    I4 = samples[:,8] + samples[:,10]
    # DYNAMICS EXTRACTION
    results = [Th1, Th2, Igamma, I2, I10, I4]
    dynamics = [np.cumsum(x) for x in results]
    dynamics = np.array(dynamics)
    # DYNAMICS CALIBRATION (Non-Negative)
    min_val = dynamics[2:,:].min()
    dynamics[2:,:] = dynamics[2:,:]+np.abs(min_val)
    min_val = dynamics[:2,:].min()
    dynamics[:2,:] = dynamics[:2,:]+np.abs(min_val)
    # DYNAMICS EXTRACTION
    Th1 = dynamics[0]
    Th2 = dynamics[1]
    Igamma = dynamics[2]
    I2 = dynamics[3]
    I10 = dynamics[4]
    I4 = dynamics[5]
    # PREDICTION
    y_pred = np.array([Igamma[-1], I2[-1], I10[-1], I4[-1]]).astype('float')
    y_pred = MinMaxNormalization(y_pred)
    loss = criterion(torch.tensor(y_pred, requires_grad = True), torch.tensor(y_true))
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    # Print progress
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
    if loss < best_loss:
        #best_Th1 = MinMaxNormalization(dynamics[0])
        #best_Th2 = MinMaxNormalization(dynamics[1])
        best_Th1 = dynamics[0]
        best_Th2 = dynamics[1]
        best_Igamma = MinMaxNormalization(dynamics[2])
        best_I2 = MinMaxNormalization(dynamics[3])
        best_I10 = MinMaxNormalization(dynamics[4])
        best_I4 = MinMaxNormalization(dynamics[5])
        best_loss = loss
        out_df = pd.DataFrame([])
        out_df['Th1'] =  best_Th1
        out_df['Th2'] = best_Th2
        out_df['Igamma'] = best_Igamma
        out_df['I2'] = best_I2
        out_df['I10'] = best_I10
        out_df['I4'] = best_I4
        out_df.to_csv('{}_best_result.csv'.format(exp_name), index = False)
        # Save the model
        torch.save(model.state_dict(), '{}_best_model.pth'.format(exp_name))
print(best_loss)

Th1 = best_Th1
Th2 = best_Th2
Igamma = best_Igamma
I2 = best_I2
I10 = best_I10
I4 = best_I4


# VISUALIZATION
# Plot results in subplots
t = np.linspace(0, shoots, len(Th1))
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0,0].plot(t, Th1, label='Th1', color = plt.cm.Set1(0))
axs[0,0].plot(t, Th2, label='Th2', color = plt.cm.Set1(1))
axs[0,0].set_xlabel('Time [day]', fontweight = 'bold', fontsize = 14)
axs[0,0].set_ylabel('Level [pg/ml]', fontweight = 'bold', fontsize = 14)
axs[0,0].set_title('Cell Dynamics', fontweight = 'bold', fontsize = 14)
axs[0,0].legend(loc = 'upper left')
axs[0,0].grid(True)


axs[1,0].plot(t, Igamma, label=r'IFN-$\gamma$', color = plt.cm.Set2(0))
axs[1,0].plot(t, I2, label='IL-2', color = plt.cm.Set2(1))
axs[1,0].plot(t, I10, label='IL-10', color = plt.cm.Set2(2))
axs[1,0].plot(t, I4, label='IL-4', color = plt.cm.Set2(3))

axs[1,0].plot(t[-1], y_true[0], marker = '*', markersize = 20, color = plt.cm.Set2(0))
axs[1,0].plot(t[-1], y_true[1], marker = '*', markersize = 20, color = plt.cm.Set2(1))
axs[1,0].plot(t[-1], y_true[2], marker = '*', markersize = 20, color = plt.cm.Set2(2))
axs[1,0].plot(t[-1], y_true[3], marker = '*', markersize = 20, color = plt.cm.Set2(3))
axs[1,0].set_xlabel('Time [day]', fontweight = 'bold', fontsize = 14)
axs[1,0].set_ylabel('Normalized Level [pg/ml]', fontweight = 'bold', fontsize = 14)
axs[1,0].set_title('Cytokine Dynamics', fontweight = 'bold', fontsize = 14)
axs[1,0].legend(loc = 'upper left')
axs[1,0].grid(True)


axs[0,1].plot(t, Th1/Th2, label='Th1/Th2 Ratio', color = plt.cm.Set1(0))
'''
axs[0,1].axhline(y = ratio.iloc[0,1], ls = '--', label = 'Q1', color = plt.cm.Set2(2))
axs[0,1].axhline(y = ratio.iloc[0,2], ls = '--', label = 'Q2', color = plt.cm.Set2(1))
axs[0,1].axhline(y = ratio.iloc[0,3], ls = '--', label = 'Q3', color = plt.cm.Set2(0))
'''
axs[0,1].set_xlabel('Time [day]', fontweight = 'bold', fontsize = 14)
axs[0,1].set_ylabel('Th1/Th2 Ratio', fontweight = 'bold', fontsize = 14)
axs[0,1].set_title('T Cell Ratio', fontweight = 'bold', fontsize = 14)
axs[0,1].legend(loc = 'upper right')
axs[0,1].grid(True)



axs[0,2].scatter(I2, Igamma, label= r'[1] IL-2 v.s [2] IFN-$\gamma$', color = plt.cm.Set1(0), s = 2)
axs[0,2].scatter(I4, I10, label='[1] IL-4 v.s [2] IL-10',color =  plt.cm.Set1(1), s = 2)
axs[0,2].set_xlabel('Cytokine [1]', fontweight = 'bold', fontsize = 14)
axs[0,2].set_ylabel('Cytokine [2]', fontweight = 'bold', fontsize = 14)
axs[0,2].set_title('Intra-cellular Correlation', fontweight = 'bold', fontsize = 14)
axs[0,2].legend(loc = 'upper left')
axs[0,2].grid(True)



axs[1,1].scatter(Igamma, I4, label= r'[1] IFN-$\gamma$ v.s [2] IL-4',  color = plt.cm.Set2(0), s = 2)
axs[1,1].scatter(Igamma, I10, label= r'[1] IFN-$\gamma$ v.s [2] IL-10',  color = plt.cm.Set2(1), s = 2)
axs[1,1].set_xlabel('Cytokine [1]', fontweight = 'bold', fontsize = 14)
axs[1,1].set_ylabel('Cytokine [2]', fontweight = 'bold', fontsize = 14)
axs[1,1].set_title('Inter-cellular Correlation', fontweight = 'bold', fontsize = 14)
axs[1,1].legend(loc = 'upper left')
axs[1,1].grid(True)

axs[1,2].scatter(I2, I4, label='[1] IL-2 v.s [2] IL-4', color = plt.cm.Set2(0), s = 2)
axs[1,2].scatter(I2, I10, label='[1] IL-2 v.s [2] IL-10', color = plt.cm.Set2(1), s = 2)
axs[1,2].set_xlabel('Cytokine [1]', fontweight = 'bold', fontsize = 14)
axs[1,2].set_ylabel('Cytokine [2]', fontweight = 'bold', fontsize = 14)
axs[1,2].set_title('Inter-cellular Correlation', fontweight = 'bold', fontsize = 14)
axs[1,2].legend(loc = 'upper left')
axs[1,2].grid(True)

for i in range(2):
    for j in range(3):
        axs[i,j].tick_params(axis='both', which='major', labelsize=14)
        axs[i,j].tick_params(axis='both', which='minor', labelsize=14)

plt.tight_layout()
plt.savefig('{}_best_result.jpg'.format(exp_name), dpi = 600)
plt.show()
