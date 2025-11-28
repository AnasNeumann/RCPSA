import matplotlib.pyplot as plt
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

# ========================================================
# =*= Dynamic Data Tacker: for loss, epsilon, and Cmax =*=
# ========================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

class Tracker():
    def __init__(self, xlabel: str, ylabel: str, title: str, color: str, show: bool = True, width=7.04, height=4.80):
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

class TreeTracker:
    def __init__(self, title: str = "Search Tree Exploration", show: bool = True, update_frequency: int = 1):
        self.show: bool            = show
        self.update_frequency: int = update_frequency
        self.episode_counter: int  = 0
        self.G                     = nx.DiGraph()
        self.known_nodes: set      = set()
        self.best_path_nodes: set  = set()
        if self.show:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.ax.set_title(title)
            self.ax.axis('off')

    def update(self, transitions: list, is_best: bool = False):
        """
            Updates the tree with new transitions.
            transitions: List[Transition] from the current episode.
        """
        if not self.show:
            return
        self.episode_counter += 1
        new_nodes             = []
        for t in transitions:
            t_id = id(t)
            if t_id not in self.known_nodes:
                self.known_nodes.add(t_id)
                new_nodes.append(t_id)
                lbl = str(t.action.item())
                self.G.add_node(t_id, label=lbl, color='green') # Mark as new
                if t.parent is not None:
                    p_id = id(t.parent)
                    if p_id in self.known_nodes:
                        self.G.add_edge(p_id, t_id)
                    else:
                        pass
                else:
                    self.G.nodes[t_id]['shape'] = 's' # Square for root
        if is_best:
            self.best_path_nodes.clear()
            for t in transitions:
                self.best_path_nodes.add(id(t))
        if self.episode_counter % self.update_frequency == 0:
            self._render(new_nodes)

    def _render(self, new_nodes):
        self.ax.clear()
        try:
            pos = graphviz_layout(self.G, prog="dot") 
        except:
            pos = nx.kamada_kawai_layout(self.G)
        node_colors = []
        for n in self.G.nodes():
            if n in self.best_path_nodes:
                node_colors.append('red')        # Best Solution
            elif n in new_nodes:
                node_colors.append('#00ff00')  # Bright Green (Just added)
            else:
                node_colors.append('#cccccc')  # Gray (Old history)
        nx.draw(self.G, pos, ax=self.ax, 
                node_color=node_colors, 
                with_labels=True, 
                labels=nx.get_node_attributes(self.G, 'label'),
                node_size=300, 
                font_size=8,
                arrows=True,
                arrowsize=10,
                edge_color='#999999',
                width=0.5)
        plt.pause(0.001)

    def save(self, path):
        if self.show:
            self.fig.savefig(path + "_tree.png")