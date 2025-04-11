from helper import *
from qiskit import QuantumCircuit


def cFRQI(a, compression):
    """    Takes a standard image in a numpy array (so that the matrix looks like
    the image you want if you picture the pixels) and returns the QPIXL
    compressed FRQI circuit. The compression ratio determines
    how many gates will be filtered and then cancelled out. Made into code from this paper:
    https://www.nature.com/articles/s41598-022-11024-y

    Args:
        a (np.array): numpy array of image, must be flattened and padded with zeros up to a power of two
        compression (float): number between 0 an 100, where 0 is no compression and 100 is no image

    Returns:
        QuantumCircuit: qiskit circuit that prepared the encoded image
    """
    a = convertToAngles(a) # convert grayscale to angles
    a = preprocess_image(a) # need to flatten the transpose for easier decoding, 
                            # only really necessary if you want to recover an image.
                            # for classification tasks etc. transpose isn't required.
    n = len(a)
    k = ilog2(n)

    a = 2*a 
    a = sfwht(a)
    a = grayPermutation(a) 
    a_sort_ind = np.argsort(np.abs(a))

    # set smallest absolute values of a to zero according to compression param
    cutoff = int((compression / 100.0) * n)
    for it in a_sort_ind[:cutoff]:
        a[it] = 0
    # print(a)
    # Construct FRQI circuit
    circuit = QuantumCircuit(k + 1)
    # Hadamard register
    circuit.h(range(k))
    # Compressed uniformly controlled rotation register
    ctrl, pc, i = 0, 0, 0
    while i < (2**k):
        # Reset the parity check
        pc = int(0)

        # Add RY gate
        if a[i] != 0:
            circuit.ry(a[i], k)

        # Loop over sequence of consecutive zero angles to 
        # cancel out CNOTS (or rather, to not include them)
        if i == ((2**k) - 1):
            ctrl=0
        else:
            ctrl = grayCode(i) ^ grayCode(i+1)
            ctrl = k - countr_zero(ctrl, n_bits=k+1) - 1

        # Update parity check
        pc ^= (2**ctrl)
        i += 1
        
        while i < (2**k) and a[i] == 0:
            # Compute control qubit
            if i == ((2**k) - 1):
                ctrl=0
            else:
                ctrl = grayCode(i) ^ grayCode(i+1)
                ctrl = k - countr_zero(ctrl, n_bits=k+1) - 1

            # Update parity check
            pc ^= (2**ctrl)
            i += 1
                        
        for j in range(k):
            if (pc >> j)  &  1:
                circuit.cx(j, k)
    return circuit.reverse_bits()


def cFRQIangs(a, compression, pre_pattern=None,post_pattern=None):
    """    Takes a standard image in a numpy array (so that the matrix looks like
    the image you want if you picture the pixels) and returns the QPIXL
    compressed FRQI circuit. The compression ratio determines
    how many gates will be filtered and then cancelled out. Made into code from this paper:
    https://www.nature.com/articles/s41598-022-11024-y

    Args:
        a (np.array): numpy array of image, must be flattened and padded with zeros up to a power of two
        compression (float): number between 0 an 100, where 0 is no compression and 100 is no image

    Returns:
        QuantumCircuit: qiskit circuit that prepared the encoded image
    """
    a = convertToAngles(a) # convert grayscale to angles
    a = preprocess_image(a) # need to flatten the transpose for easier decoding, 
                            # only really necessary if you want to recover an image.
                            # for classification tasks etc. transpose isn't required.
    n = len(a)
    k = ilog2(n)

    a = 2*a 
    a = sfwht(a)
    a = grayPermutation(a) 
    a_sort_ind = np.argsort(np.abs(a))

    # set smallest absolute values of a to zero according to compression param
    cutoff = int((compression / 100.0) * n)
    for it in a_sort_ind[:cutoff]:
        a[it] = 0
    # print(a)
    # Construct FRQI circuit
    circuit = QuantumCircuit(k + 2)
    # Hadamard register
    circuit.h(range(2,k+2))
    circuit.x(0)
    # Compressed uniformly controlled rotation register
    ctrl, pc, i = 0, 0, 0
    while i < (2**k):
        # Reset the parity check
        pc = int(0)

        # Add RY gate
        if a[i] != 0:
            if pre_pattern is None:
            # circuit.ry(a[i],0)
            # circuit.rx(-a[i]/2,0)
            # circuit.unitary(np.array([[np.cos(a[-i]), -1j*np.sin(a[-i])],
            #                          [-1j*np.sin(a[-i]), np.cos(a[-i])]]), [0], label='ry')
                circuit.cry(a[i],0,1)
            # circuit.unitary(np.array([[np.cos(-a[-i]), -1j*np.sin(-a[-i])],
            #                          [-1j*np.sin(-a[-i]), np.cos(-a[-i])]]), [0], label='ry')
            else:
                pre_pattern(circuit)
                circuit.cry(a[i],0,1)
                post_pattern(circuit)
            
        # Loop over sequence of consecutive zero angles to 
        # cancel out CNOTS (or rather, to not include them)
        if i == ((2**k) - 1):
            ctrl=0
        else:
            ctrl = grayCode(i) ^ grayCode(i+1)
            ctrl = k - countr_zero(ctrl, n_bits=k+1) - 1

        # Update parity check
        pc ^= (2**ctrl)
        i += 1
        
        while i < (2**k) and a[i] == 0:
            # Compute control qubit
            if i == ((2**k) - 1):
                ctrl=0
            else:
                ctrl = grayCode(i) ^ grayCode(i+1)
                ctrl = k - countr_zero(ctrl, n_bits=k+1) - 1

            # Update parity check
            pc ^= (2**ctrl)
            i += 1
                        
        for j in range(k):
            if (pc >> j)  &  1:
                circuit.cx(k-j+1, 1)
    return circuit

def decodeAngQPIXL(state, qc, length ,max_pixel_val=255, min_pixel_val=0):
    """Automatically decodes qpixl output statevector

    Args:
        state (statevector array): statevector from simulator - beware of bit ordering
        qc (qiskit circuit): the circuit used for the state generation
        max_pixel_val (int, optional): normalization value. Defaults to 255.
    Returns:
        np.array: your image, flat
    """
    decoded_data = []
    datum=0
    to_trace = list(range(length))
    to_trace.pop(length-datum-1)
    test = decodeQPIXL(partial_trace(state, [qc.qubits.index(qubit) for qubit in [qc.qubits[qub] for qub in to_trace]]).probabilities())
    return convertToGrayscale(np.array([test[permute_bits(i,len(qc.qubits)-length,datum)] for i in range(len(test))]),max_pixel_val,min_pixel_val)
