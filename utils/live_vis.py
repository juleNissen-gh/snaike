"""
A module for real-time visualization of time series data with matplotlib.

This module provides classes and utilities for creating live-updating plots
with support for multiple data series, dual y-axes, and various smoothing functions.
The visualization automatically handles data reduction for long-running series
to maintain performance.

Main components:
    - SeriesConfig: Configuration class for individual data series
    - LiveVis: Main visualization class for creating and updating live plots
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from typing import Callable, Dict, Any, Optional, Union, List, Tuple


@dataclass
class SeriesConfig:
    """
    Configuration for a single data series in the visualization.

    This class holds all parameters and state for a single plotted series,
    including plot styling, axis assignment, and smoothing settings.

    Attributes:
        plot_args: Positional arguments passed to matplotlib plot function (default: none)
        plot_kwargs: Keyword arguments passed to matplotlib plot function (default: none)
        axis: Index of the y-axis to use (0 for primary, 1 for secondary) (default: axis 0)
        smoothing_window_size: Number of points to include in moving average (default: 1)
        smoothing_fn: Function to apply to smoothing window to get displayed value (default: lambda x: x[-1])
        centered_smoothing: Whether to center the smoothing window on current point (default: True)
        y_vals: Array of y-values for the series (initialized post-init)
        x_vals: Array of x-values for the series (initialized post-init)
        smoothing_window: Array of recent values for smoothing (initialized post-init)
        line: matplotlib Line2D object representing the series (initialized post-init)
    """
    plot_args: tuple = ()
    plot_kwargs: Dict[str, Any] = field(default_factory=dict)
    axis: int = 0
    smoothing_window_size: int = 1
    smoothing_fn: Callable[[np.ndarray], Union[float, np.floating]] = lambda x: x[-1]
    centered_smoothing: bool = True

    y_vals: np.ndarray = field(init=False)
    x_vals: np.ndarray = field(init=False)
    smoothing_window: np.ndarray = field(init=False)
    line: Line2D = field(init=False)

    def __post_init__(self) -> None:
        """Initialize arrays after dataclass initialization."""
        self.y_vals = np.array([], dtype=np.float16)
        self.x_vals = np.array([], dtype=np.uint32)
        self.smoothing_window = np.array([])

    def set_line(self, ax: Axes) -> None:
        """
        Initialize the matplotlib line object for this series.

        Args:
            ax: The matplotlib Axes object to plot on
        """
        self.line, = ax.plot([], [], *self.plot_args, **self.plot_kwargs)


class LiveVis:
    """
    A class for creating and updating a live plot of training metrics.

    This class manages a matplotlib figure with optional dual y-axes.
    It provides methods to update the plot with new data and handles
    automatic data reduction to maintain performance over long runs.

    Attributes:
        update_freq: Number of steps between plot updates
        episode: Current episode/step number
        n_reduced: Counter for number of data reduction operations
        reduc_freq: Frequency of data reduction operations
        graphs: Dictionary mapping series names to their configurations
        twiny: Whether to use dual y-axes
        axes: List of matplotlib Axes objects
        ylims: List of y-axis limits for both axes
        xlim: Current x-axis limit
    """

    @staticmethod
    def terminate() -> None:
        """Terminates all open matplotlib plots."""
        plt.close('all')

    @staticmethod
    def event_loop() -> None:
        """Triggers a matplotlib GUI event loop update."""
        plt.pause(0.05)

    @staticmethod
    @lru_cache(maxsize=2)
    def gaussian_weights(n: int, standard_deviation: float) -> np.ndarray:
        """
        Calculates Gaussian weights for smoothing data.

        Args:
            n: The number of weights to generate
            standard_deviation: The standard deviation of the Gaussian distribution

        Returns:
            An array of Gaussian weights normalized to sum to 1
        """
        x = np.arange(n)
        weights = np.exp(-((x - (n - 1) / 2) ** 2) / (2 * standard_deviation ** 2))
        weights /= np.sum(weights)
        return weights

    @staticmethod
    def linspace_float(*args, **kwargs) -> np.ndarray:
        """
        np.linspace wrapper that supports float values for num parameter.

        The decimal portion of num is treated as a probability of rounding up.

        Args:
            *args: Positional arguments for np.linspace
            **kwargs: Keyword arguments for np.linspace

        Returns:
            Array of evenly spaced numbers
        """
        if len(args) >= 3:
            start, stop, num, *rest = args
            num = int(num) + ((num % 1) > random.random())
            args = (start, stop, num) + tuple(rest)
        elif 'num' in kwargs:
            num = kwargs['num']
            kwargs['num'] = int(num) + ((num % 1) > random.random())
        return np.linspace(*args, **kwargs)

    def __init__(
            self,
            graphs: Dict[str, SeriesConfig],
            update_freq: int = 1,
            reduce_freq: int = 2000,
            twiny: bool = False,
            title: str = 'Stats',
            x_name: str = 'x-axis',
            y_names: Tuple[str, str] = ('y-axis', 'Second y-axis'),
            y_scales: Tuple[str, str] = ('linear', 'linear'),
            legend_loc: str = "best",
            gridaxis: int = 0
    ) -> None:
        """
        Initialize the LiveVis visualization.

        Args:
            graphs: Dictionary mapping series names to their configurations
            update_freq: Number of steps between plot updates
            reduce_freq: Frequency of data reduction operations
            twiny: Whether to use dual y-axes
            title: Plot title
            x_name: Label for x-axis
            y_names: Labels for y-axes (primary, secondary)
            y_scales: Scale types for y-axes ('linear' or 'log')
            legend_loc: Location of the legend
            gridaxis: Which axis to show grid lines on (-1 for none)

        Raises:
            Exception: If gridaxis=1 is specified without twiny=True
        """
        self.update_freq = update_freq
        self.episode = 0
        self.n_reduced = 1
        self.reduc_freq = reduce_freq

        self.graphs = graphs
        self.twiny = twiny

        self.axes: List[Optional[Axes]] = [plt.subplots()[1], None]
        self.axes[0].set_title(title)
        self.axes[0].set_xlabel(x_name)
        self.axes[0].set_ylabel(y_names[0])
        self.axes[0].set_yscale(y_scales[0])

        if self.twiny:
            self.axes[1] = self.axes[0].twinx()
            self.axes[1].set_ylabel(y_names[1])
            self.axes[1].set_yscale(y_scales[1])

        if gridaxis == 1 and not twiny:
            raise Exception('The grid cannot be on a nonexistent second axis')
        if gridaxis != -1:
            self.axes[gridaxis].grid(True)

        for graph in self.graphs.values():
            graph.set_line(self.axes[graph.axis and twiny])

        if self.twiny:
            lines = [graph.line for graph in self.graphs.values()]
            labels = [graph.plot_kwargs.get('label', '') for graph in self.graphs.values()]
            self.axes[0].legend(lines, labels, loc=legend_loc)
        else:
            self.axes[0].legend(loc=legend_loc)

        self.ylims: List[Tuple[float, float]] = [(float('inf'), float('-inf')), (float('inf'), float('-inf'))]
        self.xlim = 0

    def update(self, values: Dict[str, float]) -> Dict:
        """
        Updates the plot with new data points.

        Args:
            values: Dictionary mapping series names to their new values

        Raises
            ValueError: If any input value is NaN
            Exception: If a value is missing for any configured series

        Returns
            Dict of averaged input values
        """
        return_dict = dict()

        self.episode += self.update_freq

        for graph_name, graph in self.graphs.items():

            try:
                graph.smoothing_window = np.append(graph.smoothing_window, values[graph_name])
            except KeyError:
                raise Exception(f"Missing update value for graph series: '{graph_name}'")

            if len(graph.smoothing_window) > graph.smoothing_window_size:
                graph.smoothing_window = graph.smoothing_window[1::]
            # noinspection PyArgumentList
            graph.y_vals = np.append(graph.y_vals, graph.smoothing_fn(graph.smoothing_window))

            return_dict[graph_name] = graph.y_vals[-1]

            if np.isnan(graph.y_vals[-1].item()):
                raise ValueError("Input value cannot be NaN")

            graph.x_vals = np.append(
                graph.x_vals,
                self.episode - self.update_freq * (graph.centered_smoothing * (graph.smoothing_window_size - 1) / 2)
            )

            graph.line.set_ydata(graph.y_vals)
            graph.line.set_xdata(graph.x_vals)

            if graph.x_vals[-1] > 0:
                self.ylims[graph.axis] = (
                    min(self.ylims[graph.axis][0], graph.y_vals[-1].item()),
                    max(self.ylims[graph.axis][1], graph.y_vals[-1].item())
                )

        self.xlim += self.update_freq
        plt.xlim(0, max(1, self.xlim))

        for ax, lims in zip(self.axes, self.ylims):
            if ax is None or float('inf') in lims or float('-inf') in lims:
                continue
            ax.set_ylim(lims[0], lims[1] + 1)

        if (length := len(next(iter(self.graphs.values())).y_vals)) >= self.reduc_freq:
            self._reduce_data_resolution(length)

        self.event_loop()
        return return_dict

    def _reduce_data_resolution(self, length: int) -> None:
        """
        Reduces the resolution of stored data to conserve memory.

        Args:
            length: Current length of data arrays
        """
        self.n_reduced += 1
        indices = np.append(
            np.mod(
                self.linspace_float(
                    0,
                    self.reduc_freq // 2,
                    int(self.reduc_freq / 2 * (1 - 1 / self.n_reduced)),
                    dtype=int
                ) + 3 * self.n_reduced,
                self.reduc_freq // 2
            ),
            np.mod(
                self.linspace_float(
                    0,
                    self.reduc_freq // 2,
                    int(self.reduc_freq / 2 * 1 / self.n_reduced),
                    dtype=int
                ) + 3 * self.n_reduced,
                self.reduc_freq // 2
            ) + self.reduc_freq // 2,
        )

        mask = np.zeros(length)
        mask[indices] = 1

        for graph in self.graphs.values():
            graph.y_vals = graph.y_vals[mask.astype(bool)]
            graph.x_vals = graph.x_vals[mask.astype(bool)]


def live_vis_test() -> None:
    """
    Example usage of the LiveVis class.

    Creates a visualization with two series:
    1. A noisy increasing linear trend with Gaussian smoothing
    2. An exponential decay series
    """
    def gaussian_smooth(values: np.ndarray, std: float) -> float:
        """Apply Gaussian smoothing to an array of values."""
        weights = LiveVis.gaussian_weights(len(values), std)
        return np.average(values, weights=weights)

    series = SeriesConfig(
        plot_kwargs={'color': 'red', 'label': 'data'},
        smoothing_window_size=10,
        smoothing_fn=lambda x: gaussian_smooth(x, 5)
    )

    series2 = SeriesConfig(
        plot_kwargs={'color': 'blue', 'linewidth': 0.5, 'label': 'data2'},
        axis=1,
    )

    vis = LiveVis(
        {'series1': series, 'series2': series2},
        update_freq=3,
        reduce_freq=500,
        twiny=True,
        title='stats',
        x_name='xaxis',
        y_names=('y-axis', 'second y-axis'),
        y_scales=('log', 'linear'),
        legend_loc="upper right",
        gridaxis=0
    )

    for i in range(5000):
        time.sleep(0.1)
        vis.update({
            'series1': 1 + 2 * i + np.random.normal(0, 0.1),
            'series2': 1.5 * (0.9 ** i)
        })

    while True:
        pass


if __name__ == '__main__':
    live_vis_test()
