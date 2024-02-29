from helper_qml import *
import pennylane as qml
from pennylane import numpy as np

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
    # Hadamard register
    for i in range(1,k+1):
        qml.Hadamard(i)
    # Compressed uniformly controlled rotation register
    ctrl, pc, i = 0, 0, 0
    while i < (2**k):
        # Reset the parity check
        pc = int(0)

        # Add RY gate
        qml.RY(a[i], 0)

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
                qml.CNOT([k, j])

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

