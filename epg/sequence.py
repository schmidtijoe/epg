import events as evts
import options as opt
import typing
import logging

logModule = logging.getLogger(__name__)


class Sequence:
    def __init__(
            self,
            params: opt.SeqConfig = opt.SeqConfig()):
        self.params = params
        self._events = []
        self._timing = []

    def add_event(self, event: typing.Union[evts.RF, evts.Grad, evts.Relaxation, evts.Echo], start_time: float):
        self._events.append(event)
        self._timing.append(start_time)

    def get_num_events(self):
        return self._events.__len__()

    def get_event(self, idx: int):
        return self._events[idx], self._timing[idx]

    @classmethod
    def create_semc(cls, params: opt.SeqConfig = opt.SeqConfig()):
        """Constructs SEMC sequence (repeated RF excitation with constant interval TR)

        Inputs
        ------
        params : options.SeqConfig
            sequence parameters
        Returns
        -------
        seq : Sequence
            Constructed sequence

        """
        logModule.debug("__ set values __")
        # set some constants
        esp = params.ESP  # echo spacing in ms
        grad_duration = 1.0  # grad duration in ms
        rf_duration = 2.0
        t1 = 1500.0  # t1 in ms
        t2 = 35.0  # t2 in ms
        exc_fa = params.excitationFA
        exc_phase = params.excitationRfPhase
        ref_fa = params.refocusingFA[0]
        ref_phase = params.refocusingRfPhase[0]

        logModule.debug("__ build sequence __")
        seq = cls(params=params)
        logModule.debug("excitation")
        # excitation
        seq.add_event(evts.RF(flip_angle=exc_fa, phase=exc_phase, duration=rf_duration), start_time=0.0)
        # delay
        seq.add_event(evts.Relaxation(delay=esp / 2, t1=t1, t2=t2), start_time=rf_duration / 2)
        # grad crushing
        crusher_moment = 1.0
        seq.add_event(evts.Grad(moment=crusher_moment, duration=grad_duration),
                      start_time=esp / 2 - grad_duration - rf_duration / 2)

        logModule.debug("refocus")
        # refocusing
        seq.add_event(evts.RF(flip_angle=ref_fa, phase=ref_phase, duration=rf_duration),
                      start_time=esp / 2 - rf_duration / 2)
        # grad crushing
        seq.add_event(evts.Grad(moment=crusher_moment, duration=grad_duration), start_time=esp / 2 + rf_duration / 2)
        # delay
        seq.add_event(evts.Relaxation(delay=esp / 2, t1=t1, t2=t2), start_time=esp / 2)

        # first echo
        seq.add_event(evts.Echo(time=esp, echo_num=1), start_time=esp)

        # iterate
        for k in range(params.ETL - 1):
            logModule.debug(f"echo: {k + 2}")

            # delay
            seq.add_event(evts.Relaxation(delay=esp / 2, t1=t1, t2=t2), start_time=esp * (k + 1))
            # grad crushing
            seq.add_event(evts.Grad(
                moment=crusher_moment, duration=grad_duration),
                start_time=(k + 1.5) * esp - grad_duration - rf_duration / 2
            )

            # refocusing
            seq.add_event(evts.RF(
                flip_angle=ref_fa, phase=ref_phase, duration=rf_duration),
                start_time=(k + 1.5) * esp - rf_duration / 2
            )
            # grad crushing
            seq.add_event(evts.Grad(
                moment=crusher_moment, duration=grad_duration),
                start_time=(k + 1.5) * esp + rf_duration / 2
            )
            # delay
            seq.add_event(evts.Relaxation(delay=esp / 2, t1=t1, t2=t2), start_time=(k + 1.5) * esp)

            # first echo
            seq.add_event(evts.Echo(time=esp, echo_num=1), start_time=(k + 2) * esp)

        return seq
