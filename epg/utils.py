import numpy as np


class GlobalValues:
    def __init__(self):
        self.eps = np.finfo(float).eps
        self.gamma_hz = 42577478.518
        self.gamma_pi = 2 * np.pi * self.gamma_hz


def rounder(a):
    ra = 0
    R = np.array(np.array(np.real(a)*100).round(0)/100, dtype=int)
    I = np.array(np.array(np.imag(a)*100).round(0)/100, dtype=int)
    if np.absolute(R) > 5*np.finfo(float).eps:
        ra += R
    if np.absolute(I) > 5*np.finfo(float).eps:
        ra += I*1j
    return ra

