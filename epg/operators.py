import math as m
import cmath as cm
import numpy as np


# Important methods for EPG operations
# RF operator (T)
def rf_rotation(phi, alpha):
    alpha = m.radians(alpha)
    phi = m.radians(phi)
    R = [[(m.cos(alpha / 2)) ** 2, cm.exp(2 * 1j * phi) * (m.sin(alpha / 2) ** 2),
          -1j * cm.exp(1j * phi) * m.sin(alpha)],
         [cm.exp(-2 * 1j * phi) * m.sin(alpha / 2) ** 2, m.cos(alpha / 2) ** 2, 1j * cm.exp(-1j * phi) * m.sin(alpha)],
         [-1j * 0.5 * cm.exp(-1j * phi) * m.sin(alpha), 1j * 0.5 * cm.exp(1j * phi) * m.sin(alpha), m.cos(alpha)]]
    return np.matrix(R)


# Relaxation operator (E)
def relaxation(tau, t1, t2, omega):
    if t1 != 0 and t2 != 0:
        e1 = m.exp(-tau / t1)
        e2 = m.exp(-tau / t2)
        relax_mat = np.matrix([[e2, 0, 0], [0, e2, 0], [0, 0, e1]])
        omega_new = relax_mat * omega
        omega_new[:, 0] = omega_new[:, 0] + np.matrix([[0], [0], [1 - e1]])
    else:
        omega_new = omega

    return omega_new


# Gradient operator (S)
def grad_shift(dk, omega):
    dk = round(dk)
    n = np.shape(omega)[1]
    if dk == 0:
        omega_new = omega
    else:
        if n > 1:
            f = np.hstack((np.fliplr(omega[0, :]), omega[1, 1:]))
            if dk < 0:
                # Negative shift (F- to the right, F+ to the left)
                f = np.hstack((np.zeros((1, np.absolute(dk))), f))
                z = np.hstack((omega[2, :], np.zeros((1, np.absolute(dk)))))
                fp = np.hstack((np.fliplr(f[0, 0:n]), np.zeros((1, np.absolute(dk)))))
                fm = f[0, n - 1:]
                fm[0, 0] = np.conj(fm[0, 0])
            else:
                # Positive shift (F+ to the right, F- to the left)
                f = np.hstack((f, np.zeros((1, np.absolute(dk)))))
                z = np.hstack((omega[2, :], np.zeros((1, np.absolute(dk)))))
                fp = np.fliplr(f[0, 0:n + dk])
                fm = np.hstack((f[0, n + dk - 1:], np.zeros((1, np.absolute(dk)))))
                fp[0, 0] = np.conj(fp[0, 0])

        else:
            # n = 1:  This happens if pulse sequence starts with nonzero transverse components
            #         and no RF pulse at t = 0 -- that is, the gradient happens first
            if dk > 0:
                fp = np.hstack((np.zeros((1, np.absolute(dk))), np.matrix(omega[0, 0])))
                fm = np.zeros((1, np.absolute(dk) + 1))
                z = np.hstack((np.matrix(omega[2, 0]), np.zeros((1, np.absolute(dk)))))
            else:
                fp = np.zeros((1, np.absolute(dk) + 1))
                fm = np.hstack((np.zeros((1, np.absolute(dk))), np.matrix(omega[1, 0])))
                z = np.hstack((np.matrix(omega[2, 0]), np.zeros((1, np.absolute(dk)))))

        omega_new = np.vstack((fp, fm, z))
    return omega_new
