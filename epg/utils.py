import numpy as np


class GlobalValues:
    def __init__(self):
        self.eps = np.finfo(float).eps
        self.gamma_hz = 42577478.518
        self.gamma_pi = 2 * np.pi * self.gamma_hz
        self.grad_shift_raster_factor: int = 100


def round_complex(a):
    ra = 0
    R = np.array(np.array(np.real(a)*100).round(0)/100, dtype=int)
    I = np.array(np.array(np.imag(a)*100).round(0)/100, dtype=int)
    if np.absolute(R) > 5*np.finfo(float).eps:
        ra += R
    if np.absolute(I) > 5*np.finfo(float).eps:
        ra += I*1j
    return ra


def shift_from_amplitude_duration(amp_mT: float, duration_us: float, dim_mm):
    phi = GlobalValues().gamma_hz * amp_mT * 1e-3 * dim_mm * 1e-3 * duration_us * 1e-6
    return phi


if __name__ == '__main__':
    grad = 20.0     # mT/m
    duration = 1050  # us
    slice_thickness = 0.7   # mm
    shift = shift_from_amplitude_duration(grad, duration, slice_thickness)

