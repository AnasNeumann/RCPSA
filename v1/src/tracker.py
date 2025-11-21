import matplotlib.pyplot as plt
import pickle

# ===========================
# =*= Dynamic Data Tacker =*=
# ===========================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

class Tracker():
    def __init__(self, xlabel: str, ylabel: str, title: str, color: str, show: bool = True, width=14.08, height=9.60):
        """
            Create a new vizual traker
            Args:
                xlabel (str): Label for the x-axis.
                ylabel (str): Label for the y-axis.
                title (str): Title of the plot.
                show (bool): Whether to display the chart interactively. Defaults to True.
        """
        self.show = show
        if self.show:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.x_data = []
        self.y_data = []
        self.episode = 0
        self.line, = self.ax.plot(self.x_data, self.y_data, label=title, color=color)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        if self.show:
            plt.ioff()
    
    def update(self, loss_value: float):
        """
            Update values of the traker
            Args:
                loss_value (float): The new loss value to add to the plot.
        """
        self.episode = self.episode + 1
        self.x_data.append(self.episode)
        self.y_data.append(loss_value)
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        if self.show:
            plt.pause(0.0001)

    def save(self, filepath: str):
        """
            Save the current plot to a png file and also save numerical values
            Args:
                filepath (str): The path where the plot and values should be saved.
        """
        self.fig.savefig(filepath + ".png")
        with open(filepath + '_x_data.pkl', 'wb') as f:
            pickle.dump(self.x_data, f)
        with open(filepath + '_y_data.pkl', 'wb') as f:
            pickle.dump(self.y_data, f)