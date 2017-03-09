"""Data Saver module to save output from different learning mechanism."""
import pickle
from copy import deepcopy
from contextlib import contextmanager


class DataSaver():
    """Save data for analysis."""

    def __init__(self, label=None):
        """Initialize the DataSaver."""
        self.data = []
        if label is None:
            label = 'root'
        self.labels = [label]

    def add(self, label=None, **kwargs):
        """Add element in the data saver."""
        if label is None:
            label = self.labels[-1]
        self.data.append((label, deepcopy(kwargs)))

    def write(self, path):
        """Write the data into a file."""
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    @contextmanager
    def set_context(self, label):
        """Change the default label for a context."""
        self.labels.append(label)
        try:
            yield
        finally:
            self.labels.pop()


class QuietDataSaver(DataSaver):
    """DataSaver that does nothing."""

    def add(self, label=None, **args):
        """Do nothing on add."""
        pass

    def write(self, path):
        """Do not write anything."""
        pass
