import numpy as np
import logging
import operators as op
import utils

logModule = logging.getLogger(__name__)


class Event:
    def __init__(self, duration: float):
        self.duration: float = duration  # ms
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
    def __init__(self, flip_angle: float, phase: float, duration: float, rephase_offset: float = 0.0):
        super().__init__(duration=duration)
        logModule.debug(f"create RF object: flip_angle {flip_angle:.1f}, phase {phase:.1f} \n"
                        f"Ensure posting values in degrees!")
        self.flip_angle = flip_angle
        self.phase = phase
        self._check_values()
        # want to simulate for phase offsets created by the pulse
        # (most noticeably insufficient rephasing of excitation)
        self.rephase_offset: float = rephase_offset

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
        result = np.dot(self._operator(q_mat), q_mat)
        if self.rephase_offset > utils.GlobalValues().eps:
            # we use the gradient shift operator to shift the phase states
            result = Grad().create_rf_phase_shift(phase_shift=self.rephase_offset).process(result)
        return result


class Grad(Event):
    def __init__(self, shift: float = 0.0, duration: float = 1.0, moment: float = 0.0, amplitude: float = 0.0):
        super().__init__(duration=duration)
        self.duration: float = duration  # ms
        self.phase_shift: float = shift
        self.moment, self.amplitude = self._set_moment_amplitude(duration=duration, moment=moment, amplitude=amplitude)

    @classmethod
    def create_rect_grad(cls, duration: float,
                         amplitude: float = 0.0, moment: float = 0.0,
                         voxel_dim_extend: float = 1.0):
        """

        Parameters
        ----------
        amplitude: gradient amplitude in T/m (optional, either this or moment)
        duration: gradient duration in ms
        moment: gradient moment in 1/m (optional, either this or amplitude)
        voxel_dim_extend: if no moment is supplied the moment / dephasing effect is calculated across
                            the voxel dim in mm, default - 1.0

        Returns grad event
        -------

        """
        moment, amplitude = cls()._set_moment_amplitude(duration=duration, moment=moment, amplitude=amplitude)
        phase_shift = moment * voxel_dim_extend * 1e-3
        return cls(shift=phase_shift, duration=duration, moment=moment, amplitude=amplitude)

    @classmethod
    def create_rf_phase_shift(cls, phase_shift: float = 0.0):
        # if we set rf phase offsets
        grad = cls(shift=phase_shift)
        # set duration 0 after checkup
        grad.duration = 0.0
        return grad

    @staticmethod
    def _set_moment_amplitude(duration: float, moment: float = 0.0, amplitude: float = 0.0):
        # check timing provided in ms (crusher etc might be as little as a few hundred us, set 100us as threshold here
        if duration < 0.1:
            duration *= 1e3
            if duration < 1.0:
                err = f"provide timing in ms"
                logModule.error(err)
                raise ValueError(err)
        if duration < utils.GlobalValues().eps:
            err = f"provide duration"
            logModule.error(err)
            raise ValueError(err)
        # if moment provided were good to calculate the amplitude
        if moment > utils.GlobalValues().eps:
            amplitude = np.divide(moment, utils.GlobalValues().gamma_hz * 1e-3 * duration)
        elif amplitude < utils.GlobalValues().eps:
            # no moment, no amplitude provided log in debug
            deb = f"provide gradient moment or amplitude"
            logModule.debug(deb)
        else:
            # calculate moment
            moment = utils.GlobalValues().gamma_hz * amplitude * duration * 1e-3  # cast to sec
        return moment, amplitude

    @staticmethod
    def calculate_min_duration(shift: float = 0.0, voxel_dim_extend: float = 1.0):
        # to reach shift we need a certain grad moment
        moment = shift / voxel_dim_extend * 1e3
        # set amplitude to system max: 39 mT/m
        amplitude = 0.039
        # return duration in ms
        return moment / amplitude / utils.GlobalValues().gamma_hz * 1e3

    def _operator(self, q_mat: np.ndarray):
        return op.grad_shift(self.phase_shift, q_mat)

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
            err = f"provide nonzero t1 / t2! provided: {1e3 * self.t1:.3f} ms / {1e3 * self.t2:.3f} ms"
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
