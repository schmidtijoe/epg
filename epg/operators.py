import numpy as np
import logging
import utils
logModule = logging.getLogger(__name__)


# Important methods for EPG operations
# RF operator (T)
def rf_rotation(flip_angle: float, phase: float):
    """

    Parameters
    ----------
    flip_angle: rf flip angle in deg
    phase: rf phase in deg

    Caution: we check for radians vs degrees, this is only a hacky comparison if angle is bigger than pi.
    Hence, for small flip angles (<pi in degrees) we assume this is radiant
    Returns rotation matrix
    -------

    """
    if flip_angle > 3 * np.pi / 2:
        flip_angle = np.radians(flip_angle)
    if phase > 3 * np.pi / 2:
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
    return np.array(r)


# Relaxation operator (E)
def relaxation(tau: float, t1: float, t2: float, omega: np.ndarray):
    """

    Parameters
    ----------
    tau: time for relaxation [s]
    t1: t1 param [s]
    t2: t2 param [s]

    Returns
    -------

    """
    e1 = np.exp(-tau / t1)
    e2 = np.exp(-tau / t2)
    rel_mat = np.array([
        [e2, 0, 0],
        [0, e2, 0],
        [0, 0, e1]
    ])
    tmp_result = np.dot(rel_mat, omega)
    tmp_result[:, 0] += np.array([0, 0, 1 - e1])
    return tmp_result


# Gradient operator (S)
def grad_shift(delta_k: float, omega: np.ndarray):
    """
    Parameters
    ----------
    delta_k
    omega

    Returns
    -------

    """
    # no shifting if small gradient values (0.0)
    if np.abs(delta_k) < utils.GlobalValues().eps:
        return omega

    # we always round up to next grid value: want to calculate whole state shifts.
    dk = int(np.round(delta_k))

    # omega has dim [fn+, fn-, zn]
    # however, when shifting we can walk from l- to k+ via shifts. hence we can not loose objects by shifting

    # only first two axis are shifted
    def pos_shift(init_state, delta_shift):
        tmp_result = np.zeros_like(init_state, dtype=complex)
        # plus states to the right -> conjugate the filled in from minus states
        tmp_result[0, :delta_shift] = np.conjugate(init_state[1, delta_shift:0:-1])
        tmp_result[0, delta_shift:] = init_state[0, :-delta_shift]
        # neg states to the left -> conjugate the filled in plus states
        tmp_result[1, :-delta_shift] = init_state[1, delta_shift:]
        # fill z states
        tmp_result[2] = init_state[2]
        return tmp_result

    def neg_shift(init_state, delta_shift):
        delta_shift = np.abs(delta_shift).astype(int)
        tmp_result = np.zeros_like(init_state, dtype=complex)
        # plus states to left
        tmp_result[0, :-delta_shift] = init_state[0, delta_shift:]
        # neg states to right, fill in from 0th conjugate
        tmp_result[1, :delta_shift] = np.conjugate(init_state[0, delta_shift:0:-1])
        tmp_result[1, delta_shift:] = init_state[1, :-delta_shift]
        # fill z_states
        tmp_result[2] = init_state[2]
        return tmp_result

    if dk >= 0:
        tmp_state = pos_shift(omega, dk)
    else:
        tmp_state = neg_shift(omega, dk)

    return tmp_state


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    q_state_matrix = np.zeros([3, 5], dtype=complex)  # ease in with 5 states preallocated
    # fill first 2
    q_state_matrix[:, 0] = np.array([0.3, 0.3, 0.7])
    q_state_matrix[:, 1] = np.array([0.1 + 0.2j, 0.1 - 0.2j, 0.1])
    logModule.info(f"start q matrix \n{q_state_matrix}")
    res = grad_shift(1.0, q_state_matrix)
    logModule.info(f"result: \n{res}")

    logModule.info(f"interpolate noninteger shift {1.6}, start: \n{res}")
    logModule.info(f"shift {1.0}: \n{grad_shift(1, res)}")
    logModule.info(f"shift {2.0}: \n{grad_shift(2, res)}")
    res = grad_shift(1.6, res)
    logModule.info(f"result: \n{res}")

    r = rf_rotation(90, 90)
    logModule.info(f"rotation(90,90): \n{r}")

    logModule.info(f"applied to state matrix")
    result = np.dot(r, res)
    logModule.info(f"result: \n{result}")

