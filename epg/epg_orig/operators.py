import numpy as np


# Important methods for EPG operations
# RF operator (T)
def rf_rotation(flip_angle: float, phase: float):
    """

    Parameters
    ----------
    flip_angle: rf flip angle in deg
    phase: rf phase in deg

    Returns rotation matrix
    -------

    """
    flip_angle = np.radians(flip_angle)
    phase = np.radians(phase)
    r = [
        [
            (np.cos(flip_angle / 2)) ** 2,
            np.exp(2 * 1j * phase) * (np.sin(flip_angle / 2) ** 2),
            -1j * np.exp(1j * phase) * np.sin(flip_angle)
        ],
        [
            np.exp(-2 * 1j * phase) * np.sin(flip_angle / 2) ** 2,
            np.cos(flip_angle / 2) ** 2,
            1j * np.exp(-1j * phase) * np.sin(flip_angle)
        ],
        [
            -1j * 0.5 * np.exp(-1j * phase) * np.sin(flip_angle),
            1j * 0.5 * np.exp(1j * phase) * np.sin(flip_angle),
            np.cos(flip_angle)
        ]
    ]
    return np.matrix(r)


# Relaxation operator (E)
def relaxation(tau: float, t1: float, t2: float, omega):
    """

    Parameters
    ----------
    tau: time for relaxation [s]
    t1: t1 param [s]
    t2: t2 param [s]
    omega:

    Returns
    -------

    """
    if t1 != 0 and t2 != 0:
        e1 = np.exp(-tau / t1)
        e2 = np.exp(-tau / t2)
        relax_mat = np.array([[e2, 0, 0], [0, e2, 0], [0, 0, e1]])
        omega_new = relax_mat * omega
        omega_new[:, 0] = omega_new[:, 0] + np.array([[0], [0], [1 - e1]])
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
