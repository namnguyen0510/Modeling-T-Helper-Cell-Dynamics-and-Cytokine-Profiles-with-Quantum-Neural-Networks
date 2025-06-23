import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



name = 30
# Number of qubits and classical bits
num_qubits = 11
# Define the device
qml.drawer.use_style('black_white')
dev = qml.device("default.qubit", wires=num_qubits, shots=name)

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

fig, ax = qml.draw_mpl(quantum_circuit, style = 'rcParams')(parameters)
plt.savefig('quantum_circuit.jpg', dpi = 600)
plt.show()


samples = quantum_circuit(parameters)
print(samples)
samples = 2*samples-1


Th1 = samples[:,3]
Th2 = samples[:,4]
Igamma = samples[:,5]
I10 = samples[:,6]
I2 = samples[:,7] + samples[:,9]
I4 = samples[:,8] + samples[:,10]

results = [Th1, Th2, Igamma, I10, I2, I4]
dynamics = [np.cumsum(x) for x in results]
dynamics = np.array(dynamics)
#
min_val = dynamics[2:,:].min()
dynamics[2:,:] = dynamics[2:,:]+np.abs(min_val)
min_val = dynamics[:2,:].min()
dynamics[:2,:] = dynamics[:2,:]+np.abs(min_val)
#
Th1 = dynamics[0]
Th2 = dynamics[1]

Igamma = dynamics[2]
I10 = dynamics[3]
I2 = dynamics[4]
I4 = dynamics[5]



t = np.linspace(0,name, len(Th1))



# Plot results in subplots
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
#axs[0,0].plot(t, Th1, label='Th1', color = plt.cm.Set1(0))
#axs[0,0].plot(t, Th2, label='Th2', color = plt.cm.Set1(1))
axs[1,0].plot(t, I2, label='IL-2', color = plt.cm.Set2(0))
axs[1,0].plot(t, Igamma, label=r'IFN-$\gamma$', color = plt.cm.Set2(1))
axs[1,0].plot(t, I4, label='IL-4', color = plt.cm.Set2(2))
axs[1,0].plot(t, I10, label='IL-10', color = plt.cm.Set2(3))
axs[1,0].set_xlabel('Time')
axs[1,0].set_ylabel('Concentration')
axs[1,0].set_title('Cytokine Dynamics')
axs[1,0].legend()
axs[1,0].grid(True)


axs[0,1].scatter(I2, Igamma, label= r'IL-2 v.s IFN-$\gamma$', color = plt.cm.Set1(0), s = 2)
axs[0,1].scatter(I4, I10, label='IL-4 v.s IL-10',color =  plt.cm.Set1(1), s = 2)
axs[0,1].set_xlabel('Cytokine Concentration 1')
axs[0,1].set_ylabel('Cytokine Concentration 2')
axs[0,1].set_title('Intra-cellular Corr.')
axs[0,1].legend()
axs[0,1].grid(True)

axs[0,0].plot(t, Th1, label='Th1', color = plt.cm.Set1(0))
axs[0,0].plot(t, Th2, label='Th2', color = plt.cm.Set1(1))
axs[0,0].set_xlabel('Time')
axs[0,0].set_ylabel('Concentration')
axs[0,0].set_title('Cell Dynamics')
axs[0,0].legend()
axs[0,0].grid(True)



axs[1,1].scatter(Igamma, I4, label= r'IFN-$\gamma$ v.s IL-4',  color = plt.cm.Set2(0), s = 2)
axs[1,1].scatter(Igamma, I10, label= r'IFN-$\gamma$ v.s IL-10',  color = plt.cm.Set2(1), s = 2)
axs[1,1].set_xlabel('Cytokine Concentration 1')
axs[1,1].set_ylabel('Cytokine Concentration 2')
axs[1,1].set_title('Inter-cellular Corr.')
axs[1,1].legend()
axs[1,1].grid(True)


axs[1,2].scatter(I2, I4, label='IL-2 v.s IL-4', color = plt.cm.Set2(0), s = 2)
axs[1,2].scatter(I2, I10, label='IL-2 v.s IL-10', color = plt.cm.Set2(1), s = 2)
axs[1,2].set_xlabel('Cytokine Concentration 1')
axs[1,2].set_ylabel('Cytokine Concentration 2')
axs[1,2].set_title('Inter-cellular Corr.')
axs[1,2].legend()
axs[1,2].grid(True)




plt.tight_layout()
plt.savefig('Concentration_model.jpg', dpi = 600)
plt.show()
