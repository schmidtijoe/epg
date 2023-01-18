import numpy as np
import logging

logModule = logging.getLogger(__name__)


def mag_real_2_comp(mag: np.ndarray):
    if mag.__len__() != 3:
        err = f"magnetization vector given is not 3d. shape: {mag.shape}"
        logModule.error(err)
        raise AttributeError(err)

    comp_t = np.array([
        [1.0, 1.0j, 0.0], [1.0, -1.0j, 0.0], [0.0, 0.0, 1.0]
    ])
    return np.squeeze(np.dot(comp_t, mag))


def mag_comp_2_real(mag: np.ndarray):
    if mag.__len__() != 3:
        err = f"magnetization vector given is not 3d. shape: {mag.shape}"
        logModule.error(err)
        raise AttributeError(err)

    comp_t = np.array([
        [0.5, 0.5, 0.0], [-0.5j, 0.5j, 0.0], [0.0, 0.0, 1.0]
    ])
    res = np.squeeze(np.dot(comp_t, mag))
    for val in res:
        if np.imag(val) > np.finfo(float).eps:
            err = "leftover imaginary part, suggests faulty calculation"
            logModule.error(err)
    return np.real(res)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    m_0 = np.array([0.4, 0.3, 0.2])
    logModule.info(f"Input vector: {m_0}")
    logModule.info(f"real 2 comp")
    m_res = mag_real_2_comp(m_0)
    logModule.info(f"result: {m_res}")

    m_0 = np.array([0.5 + 0.5j, 0.5 - 0.5j, 0.2])
    logModule.info(f"Input vector: {m_0}")
    logModule.info(f"comp 2 real")
    m_res = mag_comp_2_real(m_0)
    logModule.info(f"result: {m_res}")

