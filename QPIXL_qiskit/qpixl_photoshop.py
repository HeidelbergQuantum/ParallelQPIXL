from helper import *
from qpixl import *

def one_image_photoshop(backend, img,shape,comp=10, state_to_prob = np.real):
    """An example class for how you might do some 'quantum photshopping' with QPIXL

    Args:
        backend (qiskit quantum backend): A backend to simualte things, must output statevector or full probabilities
        img (np.array): array with image, can be flat or not
        shape (tuple): tuple with shape of image
        comp (int, optional): compress image, by what percentage. Defaults to 10.
        state_to_prob (function): how to cast your statevector to real values. Defaults to np.real.

    Returns:
        image: processed image
    """
    test = pad_0(img)
    test = convertToAngles(test)
    qc = cFRQI(test,10)
    ### INSERT DESIRED GATES HERE
    for i in range(1):
        qc.cnot(i,i+10)
    #################
    job = backend.run(qc)
    sv = np.real(job.result().get_statevector())
    img = decodeQPIXL(sv, state_to_prob = state_to_prob)
    img = reconstruct_img(img, shape)
    return img


def two_image_comb(backend, img1,img2,shape,comp=10,state_to_prob = np.abs):
    """An example class for how you might do some 'quantum photshopping' with QPIXL
    this can combine two images! You should change the gates in the playzone in the middle

    Args:
        backend (qiskit quantum backend): A backend to simualte things, must output statevector or full probabilities
        img1 (np.array): array with image 1, can be flat or not
        img2 (np.array): array with image 1, can be flat or not
        shape (tuple): tuple with shape of image
        comp (int, optional): compress image, by what percentage. Defaults to 10.
        state_to_prob (function): how to cast your statevector to real values. Defaults to np.real.

    Returns:
        image: processed image
    """

    img1 = convertToAngles(pad_0(img1))
    img2 = convertToAngles(pad_0(img2))
    qc1 = cFRQI(img1,comp)
    qc2 = cFRQI(img2,comp)
    big_qc = QuantumCircuit(qc1.width()+qc2.width())
    big_qc = big_qc.compose(qc1, qubits=list(range(qc1.width())))
    big_qc = big_qc.compose(qc2, qubits=list(range(qc1.width(),qc1.width()*2)))
    ### INSERT DESIRED GATES HERE
    big_qc.x(range(11,22))
    for i in range(11):
        big_qc.cnot(i, i+qc1.width())
        # Example of CNOT between two images
    #########################
    job = backend.run(big_qc)
    sv = np.real(job.result().get_statevector())
    img = decodeQPIXL(sv, state_to_prob = state_to_prob)#Image 1 is the one that is recovered
    img = reconstruct_img(img, shape)
    return img
