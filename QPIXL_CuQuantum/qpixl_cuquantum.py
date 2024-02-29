from helper import *
import cudaq

def permutation(j,perm,total_data_qubits):
    j = (j-perm)%total_data_qubits
    return j

def cFRQI(data, compression):
    """    Takes a list of data array as a numpy array (so that the matrix looks like
    the image you want if you picture the pixels) and returns the parallel QPIXL circuit. 
    The compression ratio determines how many gates will removed when the rotations are small 
    (as a percentage, so keep it small). Made into code from this paper:
    https://www.nature.com/articles/s41598-022-11024-y and expanded upon with this parallelization idea. 

    Args:
        a ([np.array, np.array,...]): list or array of numpy arrays of image, must be flattened and padded with zeros up to a power of two
        compression (float): number between 0 an 100, where 0 is no compression and 100 is no image

    Returns:
        QuantumCircuit: qiskit circuit that prepared the encoded image
    """
    kernel = cudaq.make_kernel()
    
    for ind,a in enumerate(data): #convert 'a' point into 
        data[ind] = convertToAngles(data[ind]) # convert grayscale to angles
        data[ind] = preprocess_image(data[ind]) # need to flatten the transpose for easier decoding, 
                                # only really necessary if you want to recover an image.
                                # for classification tasks etc. transpose isn't required.
        n = len(data[ind])
        k = ilog2(n)

        data[ind] = 2*data[ind]
        data[ind] = sfwht(data[ind])
        data[ind] = grayPermutation(data[ind]) 
        a_sort_ind = np.argsort(np.abs(data[ind]))

        # set smallest absolute values of a to zero according to compression param
        cutoff = int((compression / 100.0) * n)
        for it in a_sort_ind[:cutoff]:
            data[ind][it] = 0
    qubits = kernel.qalloc(k + len(data))
    # Construct FRQI circuit
    # Hadamard register
    for qubit in range(k):
        kernel.h(qubits[qubit])
    # Compressed uniformly controlled rotation register
    ctrl, pc, i = 0, 0, 0
    while i < (2**k):
        # Reset the parity check
        pc = int(0)
        # Add RY gate
        for ind, arr in enumerate(data):
            if arr[i] != 0:
                kernel.ry(arr[i], qubits[k+ind])
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
        pc_list = [] # For parallel implementation, loop over various data components
        for ind,a in enumerate(data):
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
            pc_list.append(pc)
        for j in range(k):
            for ind in range(len(data)): # For parallel implementation
                pc = pc_list[ind]
                if (pc >> j)  &  1:
                    kernel.cx(qubits[permutation(j,ind,k)], qubits[k+ind])
                
    return kernel#.reverse_bits()
# c = cFRQI(np.random.random((3,2**3)),0)