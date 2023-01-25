import numpy as np
import logging
import sequence
import utils
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpc
import matplotlib.patheffects as path_effects
import matplotlib as mpl

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
        self.q_matrix = np.zeros([3, etl ** 2 * utils.GlobalValues().grad_shift_raster_factor], dtype=complex)
        self.state_size: int = 1
        # we initialize the fn states with a size of etl**2 * grad shift raster
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
        if np.array(np.abs(self.q_matrix[:, -1]) > utils.GlobalValues().eps).any():
            err = f"allocated q matrix completely filled"
            logModule.error(err)
            # handle
            logModule.debug(f"adjusting size + 5")
            self.q_matrix = np.concatenate(
                [self.q_matrix, np.zeros([3, 5 * utils.GlobalValues().grad_shift_raster_factor])],
                axis=-1
            )

    def _add_to_history(self, desc: str, time: float, duration: float):
        size = self.get_q_size()
        self._check_q_size()
        tmp_state = self.TempState(matrix=self.q_matrix[:, :size], start_time=time, duration=duration, desc=desc)
        self.q_history.append(tmp_state)
        self.history_len += 1

    def get_q_size(self) -> int:
        idxs = np.argwhere(np.abs(self.q_matrix) > utils.GlobalValues().eps)[:, 1]
        # first dim is elements found, second is the index,
        # we dont need to know whether its x,y, or z dim. just want the highest n state
        size = np.max(idxs) + 1
        if size > self.state_size:
            self.state_size = size
        return self.state_size

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

    def plot_epg(self, save: str = None):
        plt.style.use('ggplot')
        rf_color = '#5c25ba'
        rf_color_2 = '#b300b3'
        grad_color = '#25baa6'
        state_spread_size = self.get_q_size()
        num_samples = 2000
        time_array = np.linspace(-5.0, np.max(self.get_history_state(-1).end_time), num_samples)
        # rf and grad
        rf_array = np.full_like(time_array, np.nan)
        grad_array = np.full_like(time_array, np.nan)
        echo_array = np.full_like(time_array, np.nan)

        # plotting
        mpl.rcParams['lines.linewidth'] = 2
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1])
        ax_epg = fig.add_subplot(gs[0])
        ax_epg.set_facecolor('#D1C7C5')
        ax_seq = fig.add_subplot(gs[1])
        ax_seq.set_facecolor('#D1C7C5')
        ax_epg.set_ylabel(f"$F_n, Z_n$")
        ax_epg.set_xticklabels([])
        # epgs
        # set up arrays needed to plot
        # get color mapping
        vmax = -1
        max_shift = 6
        for state_idx in range(self.history_len):
            state = self.get_history_state(state_idx)
            if state.desc == "echo":
                val = np.abs(state.matrix[0, 0])
                m = np.max(val)
                if m > vmax:
                    vmax = m
            if state.desc == "grad":
                event, _ = self.seq.get_event(state_idx - 1)
                shift_val = event.phase_shift
                if shift_val > max_shift:
                    max_shift = int(np.ceil(np.abs(shift_val)))
        norm = mpc.Normalize(vmin=0.0, vmax=1.2 * vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
        color_lines = cm.nipy_spectral(np.linspace(0, 1, 2 * state_spread_size + max_shift + 1))
        cl_half = int(color_lines.__len__() / 2)

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
                for n in range(last_state.get_state_size()):
                    if np.abs(last_state.matrix[0, n]) > utils.GlobalValues().eps:
                        # val = np.abs(tmp_state.matrix[0, n] - tmp_state.matrix[1, n])
                        ax_epg.plot(
                            time_array[start_idx:end_idx], np.linspace(n, -n, end_idx - start_idx),
                            color=rf_color
                        )
                        if np.abs(tmp_state.matrix[0, n]) > utils.GlobalValues().eps:
                            ax_epg.plot(
                                time_array[start_idx:end_idx], np.linspace(n, n, end_idx - start_idx),
                                linestyle="dashed",
                                color=rf_color,
                                alpha=0.6
                            )
                    if np.abs(last_state.matrix[1, n]) > utils.GlobalValues().eps:
                        ax_epg.plot(
                            time_array[start_idx:end_idx], np.linspace(-n, n, end_idx - start_idx),
                            linestyle="dashed",
                            color=rf_color_2,
                            alpha=0.8
                        )
                        if np.abs(tmp_state.matrix[1, n]) > utils.GlobalValues().eps:
                            ax_epg.plot(
                                time_array[start_idx:end_idx], np.linspace(-n, -n, end_idx - start_idx),
                                linestyle="dashed",
                                color=rf_color_2,
                                alpha=0.6
                            )
                    if np.abs(tmp_state.matrix[1, n]) > utils.GlobalValues().eps and np.abs(
                            tmp_state.matrix[0, n]) > utils.GlobalValues().eps:
                        mid_idx = start_idx + int((end_idx - start_idx) / 2)
                        ax_epg.plot(
                            time_array[mid_idx:end_idx], np.linspace(0, n, end_idx - mid_idx),
                            linestyle="dashed",
                            color=rf_color,
                            alpha=0.6
                        )
                        ax_epg.plot(
                            time_array[mid_idx:end_idx], np.linspace(0, - n, end_idx - mid_idx),
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
                # check for whole step shifts
                shift = event.phase_shift * utils.GlobalValues().grad_shift_raster_factor

                shift = int(np.round(shift))
                # plot lines in between gradients
                if last_state.desc == "grad":
                    # in between
                    for n in range(last_state.get_state_size()):
                        if np.abs(last_state.matrix[0, n]) > utils.GlobalValues().eps:
                            ax_epg.plot(time_array[last_idx:start_idx],
                                        np.linspace(n, n, start_idx - last_idx),
                                        color=color_lines[cl_half + n], linestyle="dashed")
                        if np.abs(last_state.matrix[1, n]) > utils.GlobalValues().eps and n > 0:
                            ax_epg.plot(time_array[last_idx:start_idx],
                                        np.linspace(-n, -n, start_idx - last_idx),
                                        color=color_lines[cl_half - n], linestyle="dashed")

                # plot epg gradient shift
                for n in range(last_state.get_state_size()):
                    if np.abs(last_state.matrix[0, n]) > utils.GlobalValues().eps:
                        ax_epg.plot(time_array[start_idx:end_idx], np.linspace(n, n + shift, end_idx - start_idx),
                                    color=color_lines[cl_half + n + shift])
                    if np.abs(last_state.matrix[1, n]) > utils.GlobalValues().eps and n > 0:
                        ax_epg.plot(time_array[start_idx:end_idx],
                                    np.linspace(-n, -n + shift, end_idx - start_idx),
                                    color=color_lines[cl_half - n + shift])
                # plot seq
                val = - 0.6 * event.phase_shift
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
                txt = ax_epg.annotate(f"{echo_val:.3f}", (time_array[start_idx], 0.3), color=mapper.to_rgba(echo_val))
                txt.set_path_effects([path_effects.Stroke(linewidth=0.4, foreground='#A1501E'), path_effects.Normal()])
                # plot seq
                echo_array[start_idx] = 0.0
                txt = ax_seq.annotate("echo", (time_array[start_idx - int(num_samples / 100)], 0.1),
                                      color='#fa9016')
                txt.set_path_effects([path_effects.Stroke(linewidth=0.4, foreground='#A1501E'), path_effects.Normal()])

            last_rf_idx = end_idx

        if last_state.desc == "grad":
            # in between
            for n in range(last_state.get_state_size()):
                if np.abs(last_state.matrix[0, n]) > utils.GlobalValues().eps:
                    ax_epg.plot(time_array[last_idx:start_idx],
                                np.linspace(n, n, start_idx - last_idx),
                                color=color_lines[cl_half + n], linestyle="dashed")
                if np.abs(last_state.matrix[1, n]) > utils.GlobalValues().eps and n > 0:
                    ax_epg.plot(time_array[last_idx:start_idx],
                                np.linspace(-n, -n, start_idx - last_idx),
                                color=color_lines[cl_half - n], linestyle="dashed")

        ax_epg.plot(time_array[first_rf_idx:last_rf_idx], np.zeros(last_rf_idx - first_rf_idx),
                    zorder=3, alpha=0.6, color=rf_color, linestyle="dashed")

        # plot seq
        ax_seq.fill_between(time_array, rf_array, color=rf_color, alpha=0.8)
        ax_seq.plot(time_array, rf_array, color=rf_color)

        ax_seq.fill_between(time_array, grad_array, color=grad_color, alpha=0.8)
        ax_seq.plot(time_array, grad_array, color=grad_color)

        ax_seq.scatter(time_array, echo_array, marker='X', s=4, color='#fa9016')

        ax_seq.set_yticklabels([])
        ax_seq.set_xlabel("time [ms]")

        ax_seq.set_xlim(time_array[0], time_array[last_rf_idx])
        ax_epg.set_xlim(time_array[0], time_array[last_rf_idx])

        def scale_to_int_states(state_num, tick_num):
            return f"{int(np.round(state_num / utils.GlobalValues().grad_shift_raster_factor))}"

        ax_epg.yaxis.set_major_formatter(plt.FuncFormatter(scale_to_int_states))
        plt.tight_layout()
        if save:
            plt.savefig(save, bbox_inches='tight')
        plt.show()

    def plot_echoes(self, save: str = None):
        # variables
        plt.style.use('ggplot')
        color = '#5c25ba'
        color_2 = '#C95711'
        echo_times = np.arange(self.seq.params.ETL + 1) * self.seq.params.ESP
        # toggle first rf
        exci_rf = False
        # get echoes
        echo_vals = []
        for ev_idx in range(self.history_len):
            tmp_state = self.get_history_state(ev_idx)
            if tmp_state.desc == "rf" and not exci_rf:
                # add initial excited magnitude
                echo_vals.append(np.abs(tmp_state.matrix[0, 0]))
                exci_rf = True
            if tmp_state.desc == "echo":
                echo_vals.append(np.abs(tmp_state.matrix[0, 0]))
        echo_vals = np.array(echo_vals)

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot()
        ax.set_xlabel('Echo Time [ms]')
        ax.set_ylabel('Normalized Echo Intensity')
        ax.set_ylim(0, 1.1)
        # ax.set_xlim(0, echo_times[-1]+np.diff(echo_times)[-1])

        ax.plot(echo_times, echo_vals, color=color)
        ax.annotate("excitation efficiency", (np.diff(echo_times)[0] / 4, echo_vals[0]), color=color_2)
        ax.scatter(echo_times[0], echo_vals[0], color=color_2)
        ax.scatter(echo_times[1:], echo_vals[1:], color=color, marker='o')

        if save:
            plt.savefig(save, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    semc_seq = sequence.create_semc()
    epg = EPG(semc_seq)
    epg.simulate()
    epg.plot_epg()
    epg.plot_echoes()
    logModule.info("done")
