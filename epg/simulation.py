import numpy as np
import logging
import sequence
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpc

logModule = logging.getLogger(__name__)


class EPG:
    class TempState:
        def __init__(self, matrix: np.ndarray, start_time: float = 0.0, duration: float = 0.0, desc: str = ""):
            self.matrix: np.ndarray = matrix
            self.start_time: float = start_time
            self.desc: str = desc
            self.end_time: float = start_time + duration

        def get_state_size(self):
            return self.matrix.shape[-1]

    def __init__(self, seq: sequence.Sequence):
        etl = seq.params.ETL
        self.seq = seq
        # need the matrix to be 3 x n  to multiply operators from the left
        self.q_matrix = np.zeros([3, etl ** 2], dtype=complex)  # we initialize the fn states with a size of etl**2
        # need checks to see if this is enough
        # the advantage is preallocate space and no need to concat and stack the matrices

        # starting with only z magnetization
        self.q_matrix[:, 0] = np.array([0, 0, 1])
        self.q_history: list = []
        self.history_len: int = 0

        self._add_to_history(time=0.0, duration=0.0, desc="init")

    def _check_q_size(self):
        # we just look if the last columns is different from 0 then we can suspect that
        # the matrix is filled all the way and more space would've been needed
        if np.array(np.abs(self.q_matrix[:, -1]) > np.finfo(float).eps).any():
            err = f"allocated q matrix completely filled"
            logModule.error(err)
            # handle
            logModule.debug(f"adjusting size + 5")
            self.q_matrix = np.concatenate([self.q_matrix, np.zeros([3, 5])], axis=-1)

    def _add_to_history(self, desc: str, time: float, duration: float):
        size = self.get_q_size()
        tmp_state = self.TempState(matrix=self.q_matrix[:, :size], start_time=time, duration=duration, desc=desc)
        self.q_history.append(tmp_state)
        self.history_len += 1

    def get_q_size(self) -> int:
        size = self.q_matrix.shape[-1] - 1
        while not np.array(np.abs(self.q_matrix[:, size]) > np.finfo(float).eps).any():
            size -= 1
        return size + 1

    def get_history(self) -> list:
        return self.q_history

    def get_history_state(self, idx) -> TempState:
        return self.get_history()[idx]

    def simulate(self):
        logModule.info("__ start sim __")
        # q matrix is initialized with only z magnetization
        for idx_event in tqdm.trange(self.seq.get_num_events(), desc="process events"):
            event, timing = self.seq.get_event(idx_event)
            event_type = event.get_type()
            logModule.debug(f"event type: {event_type}")
            # process event
            logModule.debug(f"process")
            self.q_matrix = event.process(self.q_matrix)
            logModule.debug(f"add to history")
            self._add_to_history(desc=f"{event_type}", time=timing, duration=event.duration)
        logModule.info("__ finished! __")

    def plot_epg(self):
        plt.style.use('ggplot')
        rf_color = '#5c25ba'
        grad_color = '#25baa6'
        state_spread_size = self.get_q_size()
        num_samples = 2000
        time_array = np.linspace(-5.0, np.max(self.get_history_state(-1).end_time), num_samples)
        # rf and grad
        rf_array = np.full_like(time_array, np.nan)
        grad_array = np.full_like(time_array, np.nan)
        echo_array = np.full_like(time_array, np.nan)

        # plotting
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1])
        ax_epg = fig.add_subplot(gs[0])
        ax_seq = fig.add_subplot(gs[1])
        ax_epg.set_ylabel(f"$F_n, Z_n$")
        ax_epg.set_xticklabels([])
        # epgs
        # set up arrays needed to plot
        # get color mapping
        vmax = -1
        for state_idx in range(self.history_len):
            state = self.get_history_state(state_idx)
            if state.desc == "echo":
                val = np.abs(state.matrix[0, 0])
                m = np.max(val)
                if m > vmax:
                    vmax = m
        norm = mpc.Normalize(vmin=0.0, vmax=1.2*vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)

        # keep some indexes
        start_idx = 0
        last_idx = 0
        # save state
        last_state = self.get_history_state(0)
        first_rf_idx = 0
        first_rf = False
        last_rf_idx = 0
        for idx_q in np.arange(1, self.history_len):
            event, time = self.seq.get_event(idx_q - 1)
            # go through epg history save last relevant state (all apart from relaxation)
            tmp_state = self.get_history_state(idx_q)
            # find start and end indexes dependent on sampled time array
            while time_array[start_idx] - tmp_state.start_time < 0:
                start_idx += 1
            end_idx = start_idx
            while time_array[end_idx] - tmp_state.end_time < 0:
                end_idx += 1

            # process
            if tmp_state.desc == "rf":
                if not first_rf:
                    first_rf = True
                    first_rf_idx = start_idx
                # plot epg
                # plot midpoint
                ax_epg.scatter(time_array[start_idx + int((end_idx - start_idx) / 2)], 0.0, marker='o', s=10,
                               color=rf_color)
                # plot state swaps
                for n in range(last_state.matrix.shape[-1]):
                    if np.abs(last_state.matrix[0, n]) > np.finfo(float).eps:
                        # val = np.abs(tmp_state.matrix[0, n] - tmp_state.matrix[1, n])
                        ax_epg.plot(
                            time_array[start_idx:end_idx], np.linspace(n, -n, end_idx - start_idx),
                            color=rf_color
                        )
                        if np.abs(tmp_state.matrix[0, n]) > np.finfo(float).eps:
                            ax_epg.plot(
                                time_array[start_idx:end_idx], np.linspace(n, n, end_idx - start_idx),
                                linestyle="dashed",
                                color=rf_color,
                                alpha=0.6
                            )
                    if np.abs(last_state.matrix[1, n]) > np.finfo(float).eps :
                        ax_epg.plot(
                            time_array[start_idx:end_idx], np.linspace(-n, -n, end_idx - start_idx),
                            linestyle="dashed",
                            color=rf_color,
                            alpha=0.6
                        )
                # plot midline state extend y
                ax_epg.vlines(
                    time_array[start_idx + int((end_idx - start_idx) / 2)],
                    -tmp_state.get_state_size(),
                    tmp_state.get_state_size(),
                    color=rf_color, linestyles="dotted"
                )
                # plot seq
                sinc = event.flip_angle / np.pi * np.sinc((
                                                                  time_array[start_idx:end_idx] - time_array[
                                                              start_idx + int((end_idx - start_idx) / 2)]) * 3)
                rf_array[start_idx:end_idx] = np.abs(sinc)
                ax_seq.annotate(
                    f"$\\alpha$: {event.flip_angle / np.pi * 180:.0f}$^\circ$",
                    (time_array[start_idx + int((end_idx - start_idx) / 2) + 5], 0.9 * event.flip_angle / np.pi),
                    color=rf_color
                )

            if tmp_state.desc == "grad":
                shift = event.moment
                if last_state.desc == "grad":
                    # in between
                    if np.abs(last_state.matrix[0, 0]) > np.finfo(float).eps:
                        ax_epg.plot(time_array[last_idx:start_idx], np.linspace(0, 0, start_idx - last_idx),
                                    color=grad_color)
                    for n in np.arange(1, last_state.matrix.shape[-1]):
                        if np.abs(last_state.matrix[0, n]) > np.finfo(float).eps:
                            ax_epg.plot(time_array[last_idx:start_idx],
                                        np.linspace(n, n, start_idx - last_idx),
                                        color=grad_color, linestyle="dashed")
                        if np.abs(last_state.matrix[1, n]) > np.finfo(float).eps:
                            ax_epg.plot(time_array[last_idx:start_idx],
                                        np.linspace(-n, -n, start_idx - last_idx),
                                        color=grad_color, linestyle="dashed")

                # plot epg
                if np.abs(tmp_state.matrix[0, 0]) > np.finfo(float).eps:
                    ax_epg.plot(time_array[start_idx:end_idx], np.linspace(- shift, 0, end_idx - start_idx),
                                color=grad_color)
                for n in np.arange(1, tmp_state.get_state_size()):
                    if np.abs(tmp_state.matrix[0, n]) > np.finfo(float).eps:
                        ax_epg.plot(time_array[start_idx:end_idx], np.linspace(n - shift, n, end_idx - start_idx),
                                    color=grad_color)
                    if np.abs(tmp_state.matrix[1, n]) > np.finfo(float).eps:
                        ax_epg.plot(time_array[start_idx:end_idx],
                                    np.linspace(-n - 1, -n - 1 + shift, end_idx - start_idx),
                                    color=grad_color)
                # plot seq
                val = - 0.6 * event.moment
                ramp_idx = int((end_idx - start_idx) / 8)
                grad_array[start_idx:start_idx + ramp_idx] = np.linspace(0, val, ramp_idx)
                grad_array[start_idx + ramp_idx:end_idx - ramp_idx] = val
                grad_array[end_idx - ramp_idx:end_idx] = np.linspace(val, 0, ramp_idx)

            if tmp_state.desc not in ["delay", "echo"]:
                last_idx = end_idx
                last_state = tmp_state

            if tmp_state.desc == "echo":
                # plot epg
                echo_val = np.abs(tmp_state.matrix[0, 0])
                ax_epg.scatter(time_array[start_idx], 0.0, marker='o', s=7, zorder=5, color=mapper.to_rgba(echo_val))
                ax_epg.annotate(f"{echo_val:.3f}", (time_array[start_idx], 0.3), color=mapper.to_rgba(echo_val))
                # plot seq
                echo_array[start_idx] = 0.0
                ax_seq.annotate("echo", (time_array[start_idx - int(num_samples / 100)], 0.1),
                                color='#fa9016')

            last_rf_idx = end_idx

        if last_state.desc == "grad":
            # in between
            if np.abs(last_state.matrix[0, 0]) > np.finfo(float).eps:
                ax_epg.plot(time_array[last_idx:start_idx], np.linspace(0, 0, start_idx - last_idx),
                            color=grad_color)
            for n in np.arange(1, last_state.matrix.shape[-1]):
                if np.abs(last_state.matrix[0, n]) > np.finfo(float).eps:
                    ax_epg.plot(time_array[last_idx:start_idx],
                                np.linspace(n, n, start_idx - last_idx),
                                color=grad_color, linestyle="dashed")
                if np.abs(last_state.matrix[1, n]) > np.finfo(float).eps:
                    ax_epg.plot(time_array[last_idx:start_idx],
                                np.linspace(-n, -n, start_idx - last_idx),
                                color=grad_color, linestyle="dashed")

        ax_epg.plot(time_array[first_rf_idx:last_rf_idx], np.zeros(last_rf_idx-first_rf_idx),
                    zorder=3, alpha=0.6, color=rf_color, linestyle="dashed")

        # plot seq
        ax_seq.fill_between(time_array, rf_array, color=rf_color, alpha=0.8)
        ax_seq.plot(time_array, rf_array, color=rf_color)

        ax_seq.fill_between(time_array, grad_array, color=grad_color, alpha=0.8)
        ax_seq.plot(time_array, grad_array, color=grad_color)

        ax_seq.scatter(time_array, echo_array, marker='X', s=4, color='#fa9016')

        ax_seq.set_yticklabels([])
        ax_seq.set_xlabel("time [ms]")

        plt.tight_layout()
        plt.show()

    def plot_echoes(self):
        # variables
        plt.style.use('ggplot')
        color='#5c25ba'
        echo_times = np.arange(self.seq.params.ETL+1) * self.seq.params.ESP
        # get echoes
        echo_vals = [1.0]
        for ev_idx in range(self.history_len):
            tmp_state = self.get_history_state(ev_idx)
            if tmp_state.desc == "echo":
                echo_vals.append(np.abs(tmp_state.matrix[0, 0]))
        echo_vals = np.array(echo_vals)

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot()
        ax.set_xlabel('Echo Time [ms]')
        ax.set_ylabel('Normalized Echo Intensity')

        ax.plot(echo_times, echo_vals, color=color)
        ax.scatter(echo_times[1:], echo_vals[1:], color=color, marker='o')

        plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    semc_seq = sequence.semc_sequence()
    epg = EPG(semc_seq)
    epg.simulate()
    epg.plot_epg()
    epg.plot_echoes()
    logModule.info("done")
