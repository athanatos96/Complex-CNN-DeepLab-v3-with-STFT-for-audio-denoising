import torch
import torch.functional as F

from .constant import *
from .utils import istft, realimag
#from pypesq import pesq


class PESQ:
    def __init__(self):
        self.pesq = pesq_metric

    def __call__(self, output, bd):
        return self.pesq(output, bd)


def pesq_metric(y_hat, bd):
    #print("pesq_metric")
    # PESQ
    with torch.no_grad():
        #print("pesq_metric 1a")
        y_hat = y_hat.cpu().numpy()
        #print("pesq_metric 2a")
        y = bd['y'].cpu().numpy()  # target signal
        #print("pesq_metric 3a")
        sum = 0
        #print("pesq_metric 4a")
        
        #print("y_hat: ", y_hat.shape)
        #print("y: ", y.shape)
        #print("SAMPLE_RATE: ", SAMPLE_RATE)
        #print("Inside pesq")
        for i in range(len(y)):
            #print("pesq_metric 5a")
            sum += pesq2(y[i, 0], y_hat[i, 0], SAMPLE_RATE)
            
            #print("pesq_metric 6a")
        sum /= len(y)
        #print("pesq_metric 7a")
        
        #print("End pesq_metric")
        return torch.tensor(sum)


from pesq import pesq, PesqError

def pesq2(ref, deg, fs):
    #print("Inside pesq2")
    #print('ref:',ref.shape)
    #print('deg:',deg.shape)
    #print('fs:',fs)
    # Calculate the PESQ score
    score = pesq(fs, ref, deg, mode="wb", on_error = PesqError.RETURN_VALUES)
    """
    Args:
        ref: numpy 1D array, reference audio signal 
        deg: numpy 1D array, degraded audio signal
        fs:  integer, sampling rate
        mode: 'wb' (wide-band) or 'nb' (narrow-band)
        on_error: error-handling behavior, it could be PesqError.RETURN_VALUES or PesqError.RAISE_EXCEPTION by default
    Returns:
        pesq_score: float, P.862.2 Prediction (MOS-LQO)
    """
    #print("After pesq2")
    # Return the PESQ score
    return score



# Added from https://github.com/vBaiCai/python-pesq/blob/master/pypesq/__init__.py
'''
Added becouse the original call to the package was not working
'''
'''
import warnings
import numpy as np
from pesq_core import _pesq
from math import fabs
EPSILON = 1e-6

def pesq2(ref, deg, fs=16000, normalize=False):
    print("Inside pesq")
    print(type(ref))
    print(ref.shape)
    print(type(deg))
    print(deg.shape)

'''    
'''
    params:
        ref: ref signal,
        deg: deg signal, 
        fs: sample rate,
    '''
'''
    ref = np.array(ref, copy=True)
    deg = np.array(deg, copy=True)
    print("pesq 1")
    if normalize:
        ref = ref/np.max(np.abs(ref)) if np.abs(ref) > EPSILON else ref 
        deg = deg/np.max(np.abs(deg)) if np.abs(deg) > EPSILON else deg
    print("pesq 2")
    max_sample = np.max(np.abs(np.array([ref, deg])))
    print("max_sample: ", max_sample)
    if max_sample > 1:
        c = 1 / max_sample
        print("c: ", c)
        ref = ref * c
        deg = deg * c
    print("pesq 3")
    if ref.ndim != 1 or deg.ndim != 1:
        raise ValueError("signals must be 1-D array ")

    if fs not in [16000, 8000]:
        raise ValueError("sample rate must be 16000 or 8000")

    if fabs(ref.shape[0] - deg.shape[0]) > fs / 4:
        raise ValueError("ref and deg signals should be in same length.")

    if np.count_nonzero(ref==0) == ref.size:
        raise ValueError("ref is all zeros, processing error! ")

    if np.count_nonzero(deg==0) == deg.size:
        raise ValueError("deg is all zeros, pesq score is nan! ")
    print("pesq 4")
    if ref.dtype != np.int16:
        ref *= 32767
        ref = ref.astype(np.int16)
    print("pesq 5")
    if deg.dtype != np.int16:
        deg *= 32767
        deg = deg.astype(np.int16)
    print("pesq 6")
    
    
    print("Inside pesq2")
    print('ref:',ref.shape)
    print('deg:',deg.shape)
    print('fs:',fs)
    # Calculate the PESQ score
    score = pesq(fs, ref, deg, )
    print("After pesq2")
    # Return the PESQ score
    
    
    
    try:
        score = _pesq(ref, deg, fs)
    except:
        print("pesq except")
        warnings.warn('Processing Error! return NaN')
        score = np.NAN
    print("pesq 7")
    return score
'''