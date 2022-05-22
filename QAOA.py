from pennylane import numpy as np
import pennylane as qml
import math
from tqdm import tqdm
from utilities import *
import collections

def qaoa(G:Graph, shots:int=1000, n_layers:int=1, const=0, sample_method:str='max'):
    '''
    standard qaoa for max cut
    --------------------------
    G : Graph 

    shots : number of circuit shots

    n_layers : number of QAOA layers

    const : constant in max cut objective function

    sample_method : 'max' return the bitstring with largest cut value


    Return cut value and solution
    '''
    n_wires = G.n_v
    edges = G.e
    # subgraph with no edges, any partition is optimal
    if edges == []:
        return const ,format(0,"0{}b".format(n_wires))[::-1]

    # during optimization phase we use default number of shots
    # dev is for optimization phase
    dev = qml.device('lightning.qubit', wires=n_wires,shots=shots)

    # cost Hamiltonian
    coeffs = []
    obs = []
    for edge in edges:
        coeffs.append(0.5 * edge[2])
        obs.append( qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]) )
    H_C = qml.Hamiltonian(coeffs, obs)

    # mixer Hamiltonian
    coeffs = []
    obs = []
    for i in range(n_wires):
        coeffs.append(1.)
        obs.append(qml.PauliX(i))
    H_B = qml.Hamiltonian(coeffs, obs)

    def qaoa_layer(gamma,beta):
        qml.templates.subroutines.ApproxTimeEvolution(H_C,gamma,1)
        qml.templates.subroutines.ApproxTimeEvolution(H_B,beta,1)

    def _circuit(n_layers=1):

        @qml.qnode(dev)
        def circuit(params):
            # apply Hadamards to get the n qubit |+> state
            for wire in range(n_wires):
                qml.Hadamard(wire)
            # p instances of unitary operators

            qml.layer(qaoa_layer, n_layers, params[0], params[1])
            return qml.expval(H_C)

            # during the optimization phase we are evaluating a term
            # in the objective using expval
        return circuit

    # initializing parameters
    init_params = np.ones((2, n_layers))
    opt = qml.GradientDescentOptimizer()
    #opt = qml.QNGOptimizer(0.01)
    params = init_params
    steps = 20
    for _ in range(steps):
        params = opt.step(_circuit(n_layers=n_layers), params)
    

    # measure in computational basis, 0-2^n-1 represent each basis
    def comp_basis_measurement(wires):
        n_wires = len(wires)
        return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)

    dev2 = qml.device('lightning.qubit', wires=n_wires, shots=1000)
    @qml.qnode(dev2)
    def circuit2(params, n_layers=1):
        # apply Hadamards to get the n qubit |+> state
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        qml.layer(qaoa_layer, n_layers, params[0], params[1])

        return [qml.sample(qml.PauliZ(i)) for i in range(n_wires)]

    #
    # sample after optimization to find the most frequent bitstring    
    bit_strings = []
    samples = circuit2(params, n_layers=n_layers)
    
    # samples is (n_samples, n_wires) ndarray
    for x in samples.T:
        bitlist = [str(int((i + 1)/2)) for i in x]
        bitstring = ''.join(bitlist)
        bit_strings.append(bitstring)

    counts = collections.Counter(bit_strings)
    # use the bitstring with min obj
    if sample_method == 'max':
        sol = None
        obj = 10e6
        for bitstring in counts.keys():
            #print(bitstring)
            obj_temp = 0
            for edge in edges:
                obj_temp += 0.5 * edge[2] * (2*( bitstring[ edge[0] ]==bitstring[ edge[1] ] )-1)
            if obj_temp < obj:
                obj = obj_temp
                sol = bitstring
        return const - obj, sol

    most_freq_bit_string = counts.most_common(1)
    sol = most_freq_bit_string[0][0]
    obj = 0
    for edge in edges:
        obj += 0.5 * edge[2] * (2*( sol[ edge[0] ]==sol[ edge[1] ] )-1)
    return const - obj, sol

