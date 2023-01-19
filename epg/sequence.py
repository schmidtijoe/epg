import events as evts
import options as opt
import typing
import logging

logModule = logging.getLogger(__name__)


class Sequence:
    def __init__(
            self,
            params: opt.SeqParamsSEMC = opt.SeqParamsSEMC()):
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


# -- sequence definitions --
def create_semc(params: opt.SeqParamsSEMC = opt.SeqParamsSEMC()):
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
    logModule.debug("__ build sequence __")
    seq = Sequence(params=params)
    logModule.debug("excitation")
    # excitation
    exc_rf =evts.RF(
        flip_angle=params.excitationFA,
        phase=params.excitationRfPhase,
        duration=params.excitationDuration,
        rephase_offset=params.excitationRephaseOffset
    )
    seq.add_event(exc_rf, start_time=-params.excitationDuration / 2)

    # delay
    delay = evts.Relaxation(delay=params.ESP / 2, t1=params.T1, t2=params.T2)
    seq.add_event(delay, start_time=0.0)

    # loop through refocusing
    logModule.debug("refocus etl")
    for loop_idx in range(params.ETL):
        # grad crushing
        crusher = evts.Grad().create_rect_grad(
                amplitude=1e-3*params.refocusingGradCrusher,    # cast to T/m
                duration=params.refocusingGradCrushDuration
        )
        seq.add_event(
            crusher,
            start_time=(0.5+loop_idx) * params.ESP -
                       params.refocusingGradCrushDuration - params.refocusingDuration / 2
        )

        # refocusing
        ref_rf = evts.RF(
                flip_angle=params.refocusingFA[loop_idx],
                phase=params.refocusingRfPhase[loop_idx],
                duration=params.refocusingDuration
        )
        seq.add_event(ref_rf, start_time=(loop_idx+0.5)*params.ESP - params.refocusingDuration / 2)

        # grad crushing
        seq.add_event(crusher, start_time=(loop_idx+0.5)*params.ESP + params.refocusingDuration / 2)

        # delay
        seq.add_event(delay, start_time=(loop_idx+0.5)*params.ESP)

        # echo
        logModule.debug(f"echo: {loop_idx+1}")
        echo = evts.Echo(time=(loop_idx+1)*params.ESP, echo_num=loop_idx+1)
        seq.add_event(echo, start_time=(loop_idx+1)*params.ESP)

        # delay
        seq.add_event(delay, start_time=params.ESP * (loop_idx + 1))
    return seq
