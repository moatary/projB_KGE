from collections import defaultdict
from tools_metrics import nests_weights
from datastructs import tensor3d
from utils import file2list
import numpy as np
from scipy import sparse
import pickle

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

