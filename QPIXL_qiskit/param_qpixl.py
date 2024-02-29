from helper import *
from qiskit import QuantumCircuit

def param_qpixl(a):
    """Takes in a parametervector of desired size of image (power of two) and returns a parameterized circuit.

    Args:
        a (ParameterVector): parametervector object of appropiate size

    Returns:
        QuantumCIrcuit: Parameterized QuantumCircuit to encode images of parametervector size
    """
    
    n = len(a)
    k = ilog2(n)

    # Construct parameterized QPIXL circuit
    circuit = QuantumCircuit(k + 1)
    # Hadamard register
    circuit.h(range(k))
    # Compressed uniformly controlled rotation register
    ctrl, pc, i = 0, 0, 0
    while i < (2**k):
        # Reset the parity check
        pc = int(0)

        # Add RY gate
        circuit.ry(a[i], k)

        # Loop over sequence of consecutive zero angles
        if i == ((2**k) - 1):
            ctrl=0
        else:
            ctrl = grayCode(i) ^ grayCode(i+1)
            # print('ctrl gray', ctrl)
            ctrl = k - countr_zero(ctrl, n_bits=k+1) - 1
            # print('ctrl post op', ctrl)

        # Update parity check
        pc ^= (2**ctrl)
        # print('parity   ',pc)
        i += 1
                    
            
        for j in range(k):
            if (pc >> j)  &  1:
                # print('ctrl applied   ',j)
                circuit.cnot(j, k)
    return circuit.reverse_bits()

def encode_image(img):
    """Encodes an image into parameters for parameterized qpixl 

    Args:
        img (np.array): flattened image vector

    Returns:
        np.array: parameters for circuit
    """
    img = pad_0(img)
    img = convertToAngles(img)
    return grayPermutation(sfwht(img*2))

