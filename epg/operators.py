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
def relaxation(tau: float, t1: float, t2: float):
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
    e1 = np.exp(-tau / t1)
    e2 = np.exp(-tau / t2)
    rel_mat = np.array([
        [e2, 0, 0],
        [0, e2, 0],
        [0, 0, e1]
    ])
    return rel_mat


# Gradient operator (S)
def grad_shift(delta_k: float, omega: np.ndarray):
    # no shifting if small gradient values (0.0)
    if delta_k < utils.GlobalValues().eps:
        return omega

    # we always round up to next integer value: want to calculate whole state shifts.
    # if gradient moment doesnt allow for whole state shift we take the next highest and
    # extrapolate the state afterwards
    dk = np.ceil(delta_k)
    # calculate how we overestimate the state, this fraction of the state we are using later
    interpolate_fraction = delta_k / dk
    dk = int(dk)    # cast

    # define shift
    def shift_right(arr_ax, int_shift):
        tmp = np.empty_like(arr_ax, dtype=complex)
        tmp[:int_shift] = 0.0
        tmp[int_shift:] = arr_ax[:-int_shift]
        return tmp

    def shift_left(arr_ax, int_shift):
        tmp = np.empty_like(arr_ax, dtype=complex)
        tmp[-int_shift:] = 0.0
        tmp[:-int_shift] = arr_ax[int_shift:]
        return tmp

    # only first two axis are shifted
    tmp_result = np.empty_like(omega, dtype=complex)
    if dk >= 0:
        # plus states to the right, neg states to the left
        tmp_result[0] = shift_right(omega[0], dk)
        tmp_result[1] = shift_left(omega[1], dk)
        # fill zeroth state of plus
        tmp_result[0, 0] = np.conjugate(tmp_result[1, 0])
    else:
        # plus states to left, neg states to right
        tmp_result[0] = shift_left(omega[0], np.abs(dk))
        tmp_result[1] = shift_right(omega[1], np.abs(dk))
        # fill zeroth state of minus
        tmp_result[1, 0] = np.conjugate(tmp_result[0, 0])
    tmp_result[2] = omega[2]

    # we calculated the state.
    # if delta_k / dk = 1 -> delta k is whole state shift. nothing needed
    if np.abs(interpolate_fraction - 1.0) < utils.GlobalValues().eps:
        return tmp_result

    # else we need to interpolate the states
    interp_result = omega + interpolate_fraction * np.subtract(tmp_result, omega)
    return interp_result


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

