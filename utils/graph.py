import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache


class Graph:
    @staticmethod
    def terminate():
        plt.close('all')

    @staticmethod
    @lru_cache(maxsize=2)
    def gaussian_weights(n: int, standard_deviation: int):
        x = np.arange(n)
        weights = np.exp(-((x - (n - 1) / 2) ** 2) / (2 * standard_deviation ** 2))
        weights /= np.sum(weights)
        return weights

    # Init graph
    def __init__(self, update_freq: int, score_avg_weight_sd):
        self.score_avg_weight_sd = score_avg_weight_sd
        self.update_freq = update_freq
        self.loss_values: np.ndarray = np.array([], dtype=np.float16)
        self.loss_x: np.ndarray = np.array([], dtype=np.uint32)
        self.avg_loss_values: np.ndarray = np.array([], dtype=np.float16)
        self.avg_loss_x_values: np.ndarray = np.array([], dtype=np.uint32)
        self.scores: np.ndarray = np.array([], dtype=np.float16)
        self.score_x_values: np.ndarray = np.array([], dtype=np.uint32)
        self.episode = 0
        self.n_sliced = 1
        self.slice_freq = 1200

        self.fig, self.ax1 = plt.subplots()
        self.loss_line, = self.ax1.plot([], [], label='Loss', linewidth=0.5)
        self.avg_loss_line, = self.ax1.plot([], [], label='Average Loss')

        # Create a second y-axis for score
        self.ax2 = self.ax1.twinx()  # Create a second y-axis
        self.score_line, = self.ax2.plot([], [], label='Score', color='r')

        self.ax1.grid(True)
        self.ax1.legend(loc='upper left')  # Display the labels in a legend
        self.ax2.legend(loc='upper center')

        self.ylim = (float('inf'), float('-inf'))
        self.xlim = -5
        self.score_ylim = 4

    def update_plot(self, loss_value: float, avg_loss: np.ndarray, score: np.ndarray):  # Append the new values to plot
        avg_loss_value = float(np.mean(avg_loss))
        self.loss_values = np.append(self.loss_values, loss_value)
        self.avg_loss_values = np.append(self.avg_loss_values, np.mean(avg_loss))
        self.scores = np.append(self.scores,
                                np.average(score, weights=self.gaussian_weights(len(score), self.score_avg_weight_sd)))

        self.loss_x = np.append(self.loss_x, self.episode)
        self.score_x_values = np.append(self.score_x_values, self.episode - len(score) / 2)
        self.avg_loss_x_values = (
            np.append(self.avg_loss_x_values, (self.episode - (len(avg_loss) * self.update_freq) / 2)))

        # Update the data of the line objects
        self.loss_line.set_ydata(self.loss_values)
        self.loss_line.set_xdata(self.loss_x)
        self.avg_loss_line.set_ydata(self.avg_loss_values)
        self.avg_loss_line.set_xdata(self.avg_loss_x_values)
        self.score_line.set_ydata(self.scores)
        self.score_line.set_xdata(self.score_x_values)

        # Adjust the plot limits
        self.xlim += self.update_freq
        plt.xlim(0, max(1, self.xlim))
        self.ylim = (min(self.ylim[0], loss_value, avg_loss_value),
                     max(self.ylim[1], loss_value, avg_loss_value))
        self.ax1.set_ylim(self.ylim[0] - 1, self.ylim[1] + 1)
        self.score_ylim = max(self.score_ylim, self.scores[-1].item())  # Update the score y-limit
        self.ax2.set_ylim(3, self.score_ylim + 1)  # Set the y-limits for the score axis

        if len(self.loss_values) >= self.slice_freq:  # decrease resolution to conserve memory
            self.n_sliced += 1
            # decrease by 1/n_sliced and 1-1/n_sliced to half and conserve even resolution
            indices = np.append(
                np.linspace(0, self.slice_freq // 2, int(self.slice_freq / 2 * (1 - 1 / self.n_sliced)), dtype=int),
                np.linspace(self.slice_freq // 2, self.slice_freq - 1, int(self.slice_freq / 2 * (1 / self.n_sliced)),
                            dtype=int)
            )
            self.loss_values = self.loss_values[indices]
            self.avg_loss_values = self.avg_loss_values[indices]
            self.scores = self.scores[indices]
            self.loss_x = self.loss_x[indices]
            self.avg_loss_x_values = self.avg_loss_x_values[indices]
            self.score_x_values = self.score_x_values[indices]

            # reset the lim to avoid early high values
            self.ylim = (self.ylim[0], np.max(self.avg_loss_values[self.slice_freq // 100:] + 5))

        self.episode += self.update_freq

        # plt.pause() to update
        plt.pause(0.0001)
