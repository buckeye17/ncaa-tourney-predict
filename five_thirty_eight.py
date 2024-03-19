import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

PROJ_ROOT_DIR = r"C:\Users\adiad\Documents\GitHub\ncaa-tourney-predict"
PROJ_ROOT_DIR = PROJ_ROOT_DIR.replace("\\", "/") + "/"

def read_538_odds(fn: str):
    f = open(fn, "r")
    Lines = f.readlines()

    region_ls = ["Midwest", "East", "South", "West", "Multi-Region Playoff"]
    region_lookup_ls = "MESWF"

    tourn_dict = {}
    tourn_ls = []
    for line in Lines:
        


def generate(
    odds_arr: np.ndarray,
    seed: int = None,
    print_bracket: bool = False,
    east_vs: str = "midwest",
    return_list: bool = True) -> list:
    pass