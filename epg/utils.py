import numpy as np


def rounder(a):
    ra = 0
    R = int(round(np.real(a)*100))/100
    I = int(round(np.imag(a)*100))/100
    if np.absolute(R) > 5*np.finfo(float).eps:
        ra += R
    if np.absolute(I) > 5*np.finfo(float).eps:
        ra += I*1j
    return ra

