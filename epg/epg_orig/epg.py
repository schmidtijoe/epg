# Extended Phase Graph (EPG) simulation for multi-echo MRI sequences
#   The EPG algorithm operates in the frequency domain
#   Gehua Tong, Feb 13 2019
#   Adapted from MATLAB code by Gehua Tong & Sairam Geethanath
#   Using math from the paper:
#   Weigel, M. (2015). Extended phase graphs: dephasing, RF pulses, and echoes‐pure and simple.
#                       Journal of Magnetic Resonance Imaging, 41(2), 266-295.

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import operators as op
import utils


# Main EPG classes: Sequence and EPG , with plotting and simulation methods
class Sequence(object):
    def __init__(self, rf, grad, events, time, t1t2=(0, 0), name=""):
        self.rf = rf
        self.grad = grad
        self.events = events
        self.time = time
        self.t1t2 = t1t2
        self.name = name

    def plot_seq(self):
        plt.figure(num=1)
        plt.plot([0, self.time[len(self.time) - 1]], [0.5, 0.5], 'k-')
        # Plot rf as vertical lines and annotate flip angles
        rf_ind = 0
        for k in range(len(self.time)):
            if self.events[k] == "rf":
                # Draw a line and annotate flip angle
                plt.plot([self.time[k], self.time[k]], [0.5, 1], 'b-')
                plt.text(self.time[k], 1, str(self.rf[rf_ind]) + "$^\circ$")
                rf_ind += 1
            elif self.events[k] == "grad":
                print("")
                # Fill area with gradient polarity & annotate dk
        plt.title(self.name)
        plt.show()


class EPG(object):

    def __init__(self, seq):
        self.seq = seq
        self.omega = np.matrix([])
        self.simulated = False
        self.om_store = []

    def simulate(self):
        # Initialize with (F+(0)=0,F-(0)=0,Z(0)=1)
        self.omega = np.matrix([[0], [0], [1]])
        self.om_store = []
        rf = self.seq.rf
        grad = self.seq.grad
        events = self.seq.events
        timing = self.seq.time
        uniq_times = np.unique(timing)
        t1t2 = self.seq.t1t2
        rf_index = 0
        grad_index = 0

        # Begin simulation
        n = len(events)

        for k in range(n):
            event = events[k]
            if event == "rf":
                phi = rf[rf_index][0]
                alpha = rf[rf_index][1]
                self.omega = op.rf_rotation(phi, alpha) * self.omega
                rf_index += 1

            elif event == "grad":
                dk = grad[grad_index]
                self.omega = op.grad_shift(dk, self.omega)
                grad_index += 1

            elif event == "relax":
                q = np.where(uniq_times == timing[k])[0]
                tau = uniq_times[q] - uniq_times[q - 1]
                self.omega = op.relaxation(tau, t1t2[0], t1t2[1], self.omega)

            self.om_store.append(self.omega)
        self.simulated = True
        print("Simulation complete!")

    def find_echoes(self):
        echoes = []
        timing = self.seq.time

        if self.seq.name == "STAR":
            if self.simulated:
                # proceed to find echoes

                for v in range(len(self.om_store)):
                    # Check for non-zero k=0 state
                    new_echo = (timing[v], np.absolute(self.om_store[v][0, 0]))
                    # if two non-zero F+'s happen at the same time, only save the second one as the proper echo
                    if len(echoes) != 0 \
                            and np.absolute(echoes[-1][0] - timing[v]) < 10 * np.finfo(float).eps:
                        echoes.pop()
                    echoes.append(new_echo)
                echoes = np.unique(np.array(echoes), axis=0)
        else:
            if self.simulated:
                # proceed to find echoes
                for v in range(len(self.om_store)):
                    # Check for non-zero k=0 state
                    if np.absolute(self.om_store[v][0, 0]) > 5 * np.finfo(float).eps:
                        new_echo = (timing[v], np.absolute(self.om_store[v][0, 0]))
                        # if two non-zero F+'s happen at the same time, only save the second one as the proper echo
                        if len(echoes) != 0 \
                                and np.absolute(echoes[-1][0] - timing[v]) < 10 * np.finfo(float).eps:
                            echoes.pop()
                        echoes.append(new_echo)
                echoes = np.unique(np.array(echoes), axis=0)

        return echoes

    def plot_echoes(self):
        echoes = self.find_echoes()
        plt.plot(echoes[:, 0], echoes[:, 1], '-k')
        plt.xlabel("Time(ms)", fontsize=12)
        plt.ylabel("Intensity", fontsize=12)
        plt.title("EPG echoes:" + self.seq.name)
        plt.show()

    def reset(self):
        self.omega = np.array([])
        self.simulated = False
        self.om_store = []
        print("EPG has been reset!")

    def display(self, annot=1):
        rf = self.seq.rf
        grad = self.seq.grad
        events = self.seq.events
        time = self.seq.time
        uniqtimes = np.unique(time)
        kmax = np.shape(self.om_store[-1])[1]
        if kmax != 1:
            kstates = np.arange(-kmax + 1, kmax)
        else:
            kstates = np.array([-1, 1])

        grad_ind = 0
        rf_ind = 0

        """ t = 0 """
        # Horizontal axis
        plt.plot([0, time[-1]], [0, 0], 'k-', linewidth=1.5)
        """ t > 0 """
        for event_ind in range(len(events)):
            # Get data
            if event_ind > 0:
                om_past = self.om_store[event_ind - 1]
                fpp = om_past[0, :]
                fmp = om_past[1, :]
                zp = om_past[2, :]
            else:
                fpp = []
                fmp = []
                zp = []

            if events[event_ind] == "rf":
                t = time[event_ind] * np.ones(len(kstates))
                plt.plot(t, kstates, color='r', linewidth=5)
                flip = rf[rf_ind]
                plt.text(t[0], max(kmax - 1, 1) - 0.2, "(" + str(flip[0]) + "$^\circ$," + str(flip[1]) + "$^\circ$)",
                         fontsize=10, color='r')
                rf_ind += 1

            elif events[event_ind] == "grad":
                grad_ind += 1
                # (+) Fp state plot
                fpp_kstates = np.matrix(np.where(np.absolute(fpp) > 5 * np.finfo(float).eps))
                for k in range(np.shape(fpp_kstates)[1]):
                    fp_plot = [fpp_kstates[1, k], fpp_kstates[1, k] + grad[grad_ind - 1]]
                    t = [uniqtimes[grad_ind - 1], uniqtimes[grad_ind]]
                    plt.plot(t, fp_plot, '-', color=(0.01, 0.58, 0.53))
                    if annot:
                        intensity = utils.rounder(fpp[0, fpp_kstates[1, k]])
                        plt.text(t[0], fp_plot[0] + 0.5, str(intensity), color=(0.01, 0.58, 0.53), fontsize=9)

                # (-) Fm state plot
                fmp_kstates = -1 * np.matrix(np.where(np.absolute(fmp) > 5 * np.finfo(float).eps))
                for k in range(np.shape(fmp_kstates)[1]):
                    fp_plot = [fmp_kstates[1, k], fmp_kstates[1, k] + grad[grad_ind - 1]]
                    t = [uniqtimes[grad_ind - 1], uniqtimes[grad_ind]]
                    if fmp_kstates[1, k] != 0:
                        plt.plot(t, fp_plot, '-', color=(0.02, 0.02, 0.67))

                    # Place circles for echoes (only happens for F0 states from positive gradient)
                    # Consider disabling this?
                    fmp_echo = np.where(np.array(fp_plot) == 0)[0]
                    if len(fmp_echo) != 0:
                        plt.plot(t[fmp_echo[0]], 0, '--ro',
                                 linewidth=2, markersize=8, markeredgecolor='b', markerfacecolor='#F9D40A')

                    if annot:
                        intensity = utils.rounder(fmp[0, -fmp_kstates[1, k]])
                        plt.text(t[0], fp_plot[0] - 0.5, str(intensity), color=(0.02, 0.02, 0.67), fontsize=9)

                # Zp state plot
                zp_kstates = np.matrix(np.where(np.absolute(zp) > 5 * np.finfo(float).eps))
                for k in range(np.shape(zp_kstates)[1]):
                    fp_plot = [zp_kstates[1, k], zp_kstates[1, k]]
                    t = [uniqtimes[grad_ind - 1], uniqtimes[grad_ind]]
                    plt.plot(t, fp_plot, '--', color=(1, 0.47, 0.42))

                    if annot:
                        intensity = utils.rounder(zp[0, zp_kstates[1, k]])
                        plt.text(t[0], fp_plot[0], str(intensity), color=(1, 0.47, 0.42), fontsize=9)

        baseline = -kmax - 1
        for m in range(1, len(uniqtimes)):
            if grad[m - 1] > 0:
                col = 'g'
            else:
                col = 'r'
            plt.fill([uniqtimes[m - 1], uniqtimes[m - 1], uniqtimes[m], uniqtimes[m]],
                     [baseline + grad[m - 1], baseline, baseline, baseline + grad[m - 1]], color=col)

        if kmax == 1:
            plt.axis([0, time[-1], -1, 1])
        else:
            plt.axis([0, time[-1], -kmax - 1 - max(np.absolute(grad)), kmax - 1])

        plt.xticks(uniqtimes)
        plt.title("EPG: " + self.seq.name, fontsize=12)
        plt.xlabel("Time(ms)", fontsize=12)
        plt.ylabel("k states", fontsize=12)
        plt.grid(True)

        plt.show()

    def get_sim_history(self):
        return self.om_store
