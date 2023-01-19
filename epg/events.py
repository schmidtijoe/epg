import numpy as np
import logging
import operators as op
import utils

logModule = logging.getLogger(__name__)


class Event:
    def __init__(self, duration: float):
        self.duration: float = duration             # ms
        self._check_type()

    def _check_type(self):
        t = self.__class__.__name__
        d_types = {
            "Event": "uninitialized Event",
            "RF": "rf",
            "Grad": "grad",
            "Relaxation": "delay",
            "Echo": "echo"
        }
        if d_types.get(t) is None:
            err = f"object class type is not supported: {t}, none of: {d_types.keys()}"
            logModule.error(err)
            raise AttributeError(err)
        return d_types.get(t)

    def get_type(self):
        return self._check_type()

    def _operator(self, q_mat: np.ndarray):
        # this is the method we need to set in the subclass
        return np.zeros(3)

    def _process(self, q_mat: np.ndarray):
        # this is the method we need to set to use the operator in the subclass
        return np.dot(self._operator(q_mat), q_mat)

    def process(self, q_mat: np.ndarray):
        # call method to act on q state matrix
        return self._process(q_mat)


class RF(Event):
    def __init__(self, flip_angle: float, phase: float, duration: float):
        super().__init__(duration=duration)
        logModule.debug(f"create RF object: flip_angle {flip_angle:.1f}, phase {phase:.1f} \n"
                       f"Ensure posting values in degrees!")
        self.flip_angle = flip_angle
        self.phase = phase
        self._check_values()

        # convert to radians
        self.flip_angle, self.phase = np.radians([self.flip_angle, self.phase])

    def _check_values(self):
        # catch bad values
        while np.abs(self.flip_angle) > 360.0:
            self.flip_angle -= np.sign(self.flip_angle) * 360.0
        while np.abs(self.phase) > 360.0:
            self.phase -= np.sign(self.phase) * 360.0

    def _operator(self, fn_mat: np.ndarray):
        # get rf matrix rotation
        return op.rf_rotation(self.flip_angle, self.phase)

    def _process(self, q_mat: np.ndarray):
        # use rf rotation from left to matrix
        return np.dot(self._operator(q_mat), q_mat)


class Grad(Event):
    def __init__(self, moment: float, duration: float = 1.0):
        super().__init__(duration=duration)
        self.duration: float = duration  # ms
        self.moment: float = moment  # 1/m
        self.amplitude: float = moment / utils.GlobalValues().gamma_hz / duration * 1e6  # T/m

    @classmethod
    def create_rect_grad(cls, amplitude: float, duration: float, moment: float = 0.0):
        # check timing provided in ms
        if duration < 1.0:
            duration *= 1e3
            if duration < 1.0:
                err = f"provide timing in ms"
                logModule.error(err)
                raise ValueError(err)
        # if moment provided were good
        if moment > utils.GlobalValues().eps:
            return cls(moment=moment, duration=duration)
        # if no amplitude provided raise error
        if amplitude < utils.GlobalValues().eps:
            err = f"provide gradient moment or amplitude"
            logModule.error(err)
            raise ValueError(err)
        moment = utils.GlobalValues().gamma_hz * amplitude * duration * 1e-3
        return cls(moment=moment, duration=duration)

    def _operator(self, q_mat: np.ndarray):
        return op.grad_shift(self.moment, q_mat)

    def _process(self, q_mat: np.ndarray):
        # shift operator already acts on q_mat, no dot product
        return self._operator(q_mat)


class Relaxation(Event):
    def __init__(self, delay: float, t1: float, t2: float):
        super().__init__(duration=delay)
        # provide same order of magnitude either all s or all ms
        self.t1 = t1
        self.t2 = t2

        self._check_values()

    def _check_values(self):
        if self.t1 < utils.GlobalValues().eps or self.t2 < utils.GlobalValues().eps:
            err = f"provide nonzero t1 / t2! provided: {1e3*self.t1:.3f} ms / {1e3*self.t2:.3f} ms"
            logModule.error(err)
            raise ValueError(err)

    def _operator(self, q_mat: np.ndarray):
        return op.relaxation(self.duration, self.t1, self.t2)

    def _process(self, q_mat: np.ndarray):
        return np.dot(self._operator(q_mat), q_mat)


class Echo(Event):
    def __init__(self, time: float, echo_num: int):
        super().__init__(duration=0.0)
        self.echo_num = echo_num
        self.time = time

    def _operator(self, q_mat: np.ndarray):
        return q_mat

    def _process(self, q_mat: np.ndarray):
        return q_mat


if __name__ == '__main__':
    g = Grad(1)
    omega = np.array([[0], [0], [1]])
    res = g.process(omega)
    logModule.info(f"result: {res}")
