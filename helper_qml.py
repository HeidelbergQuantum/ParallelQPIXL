# qip/helper.py

from pennylane import numpy as np
import torch
def sfwht(a):
    """Fast walsh hadamard transform with scaling

    Args:
        a (flat array): array with values to be transformed

    Returns:
        input array: array of same type as input, inplace transform
    """
    n = len(a)
    k = ilog2(n)
    j = 1
    while j < n:
        for i in range(n):
            if i & j == 0:
                j1 = i + j
                x = a[i]
                y = a[j1]
                a[i], a[j1] = (x + y) / 2, (x - y) / 2
        j *= 2
    return a            

def isfwht(a):
    """Inverse of the walsh hadamard transform

    Args:
        a (array): array of values

    Returns:
        array: array with inverse transformed applied, inplace
    """
    n = len(a)
    k = ilog2(n)
    j=1
    while j< n:
        for i in range(n):
            if (i&j) == 0:
                j1=i+j 
                x=a[i]
                y=a[j1]
                a[i],a[j1]=(x+y),(x-y)
        j*=2 
    return a            

def ispow2(x):
    """am I a power of two

    Args:
        x (int): number

    Returns:
        Bool: is it a power of two? The answer
    """
    return not (x&x-1)

def nextpow2(x):
    """Returns next power of two, or identity if x is a power of two

    Args:
        x (int): number to check

    Returns:
        int: next power of two (or x if x is a power of two)
    """
    x-=1
    x|=x>>1
    x|=x>>2
    x|=x>>4
    x|=x>>8
    x|=x>>16
    x|=x>>32
    x+=1
    return x 

def ilog2(x):
    """Integer log 2"""
    return int(np.log2(x))

def grayCode(x):
    """Gray code permutation of x, to change indices"""
    return x^(x>>1)

def grayPermutation(a):
    """Gray permutes an array"""
    b = np.zeros(len(a))
    for i in range(len(a)):
        b[i] = a[grayCode(i)]
    return b

def invGrayPermutation(a):
    """inverse gray permutes an array"""
    b = np.zeros(len(a))
    for i in range(len(a)):
        b[grayCode(i)] = a[i]
    return b

def convertToAngles(a):
    """Converts image to angles"""
    scal = np.pi/(a.max()*2)
    a = a *scal
    return a

def convertToGrayscale(a,maxval=1):
    """Converts encoded postprocessed statevector back to grayscale, normalized to maxval"""
    scal = 2*maxval/np.pi 
    a = a * scal
    return a

def countr_zero(n,n_bits=8):
    """Returns the number of consecutive 0 bits 
    in the value of x, starting from the 
    least significant bit ("right")."""
    if n == 0:
        return n_bits
    count = 0
    while n & 1 == 0:
        count += 1
        n >>= 1
    return count

def preprocess_image(img):
    """Program requires flattened transpose of image array, this returns exactly that"""
    return img.T.flatten()

def readpgm(name):
    """Reads pgm P2 files"""
    with open(name) as f:
        lines = f.readlines()
    # This ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
    # here,it makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 
    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
        
    return (np.array(data[3:]),(data[1],data[0]),data[2])

def pad_0(img):
    """Pads array with 0s to next power of two

    Args:
        img (numpy array): image, can be wide

    Returns:
        padded image: flattened image with appropiate padding for quantum algorithm
    """
    # img = np.array(img)
    img.flatten()
    # return np.pad(img,(0,nextpow2(len(img))-len(img)))
    return torch.nn.functional.pad(img, (0,nextpow2(len(img))-len(img)), mode='constant', value=0)

def decodeQPIXL(state,max_pixel_val=255, state_to_prob = np.abs):
    """Automatically decodes qpixl output statevector

    Args:
        state (statevector array): statevector from simulator - beware of bit ordering
        max_pixel_val (int, optional): normalization value. Defaults to 255.
        state_to_prob (function): If you made some transforms, your image 
                                    may be complex, how would you 
                                    like to make the vector real?
    Returns:
        np.array: your image, flat
    """
    state_to_prob(state)
    pv = np.zeros(len(state)//2)
    for i in range(0,len(state),2):
        pv[i//2]=np.arctan2(state[i+1],state[i])
    return convertToGrayscale(pv,max_pixel_val)

def reconstruct_img(pic_vec, shape: tuple):
    """reconstruct image from decoded statevector

    Args:
        pic_vec (np.array): your decoded statevector
        shape (tuple): shape that you want the image back in

    Returns:
        np.array: array of correct image size, ready to show! May need to be transposed.
    """
    ldm = shape[0]
    holder = np.zeros(shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            holder[row,col]=pic_vec[row + col * ldm]
    return holder

class examples():
    def __init__(self) -> None:
        """SImple holder class with some example images
        """
        self.space= np.array([[0,0,0,0,1,1,1,0],
                                [0,0,0,1,1,0,0,0],
                                [1,0,1,1,1,1,1,0],
                                [0,1,1,0,1,1,0,1],
                                [0,0,1,1,1,1,0,1],
                                [0,0,1,1,1,1,0,0],
                                [0,0,1,1,1,1,0,1],
                                [0,1,1,0,1,1,0,1],
                                [1,0,1,1,1,1,1,0],
                                [0,0,0,1,1,0,0,0],
                                [0,0,0,0,1,1,1,0],])
        self.invader = np.array([[0,0,0,0,1,1,1,1],
                                 [0,1,1,1,1,1,0,0],
                                 [0,1,0,0,1,1,1,1],
                                 [0,1,0,1,1,1,0,0],
                                 [1,1,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,0,0],
                                 [1,1,0,0,1,1,1,1],
                                 [0,1,0,1,1,1,0,0],
                                 [0,1,1,1,1,1,1,1],
                                 [0,1,1,1,1,1,0,0],
                                 [0,0,0,0,1,1,1,1],])