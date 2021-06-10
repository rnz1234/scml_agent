
from scml.oneshot import OneShotAgent
# required for typing
from typing import Any, Dict, List, Optional


import numpy as np
from negmas import (
    AgentMechanismInterface,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    ResponseType,
)
from negmas.helpers import humanize_time


# required for running tournaments and printing
import time
from tabulate import tabulate
from scml.scml2020.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from scml.oneshot.agents import (
    RandomOneShotAgent,
    SyncRandomOneShotAgent,
)

import sys
import math
import numpy as np
import os

from agents_pool import SimpleAgent, BetterAgent, LearningAgent, AdaptiveAgent

price_delta_up_range = np.arange(1,5,0.5)
price_delta_down_range = np.arange(1,5,0.5)
profit_epsilon_range = [0]#np.arange(0.1,1.1,0.1)
acceptance_price_th = np.arange(0.1,1.1,0.1)
acceptance_quantity_th = np.arange(0.1,1.1,0.1)
cutoff_rate = np.arange(0.1,1.1,0.1)
cutoff_precentile = 0.2
cutoff_stop_amount = 1


for price_delta_up in price_delta_up_range:
    for price_delta_down in price_delta_down_range:
        for profit_epsilon in profit_epsilon_range:
            os.system(f'python3.8 hunteragent.py 0 {price_delta_down} {price_delta_up} {profit_epsilon} 0.1 0.1 0.2 0.2 1')