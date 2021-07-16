#!/usr/bin/env python
"""
**Submitted to ANAC 2021 SCML (OneShot track)**
*Authors* 
Ran Sandhaus
Ophir Haroche
Nadav Spitzer
Laor Spitz

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.

This module implements a factory manager for the SCM 2021 league of ANAC 2021
competition (one-shot track).
Game Description is available at:
http://www.yasserm.com/scml/scml2021oneshot.pdf

Your agent can sense and act in the world by calling methods in the AWI it has.
For all properties/methods available only to SCM agents check:
  http://www.yasserm.com/scml/scml2020docs/api/scml.oneshot.OneShotAWI.html

Documentation, tutorials and other goodies are available at:
  http://www.yasserm.com/scml/scml2020docs/

Competition website is: https://scml.cs.brown.edu

To test this template do the following:

0. Let the path to this file be /{path-to-this-file}/myagent.py

1. Install a venv (recommended)
>> python3 -m venv .venv

2. Activate the venv (required if you installed a venv)
On Linux/Mac:
    >> source .venv/bin/activate
On Windows:
    >> \.venv\Scripts\activate.bat

3. Update pip just in case (recommended)

>> pip install -U pip wheel

4. Install SCML

>> pip install scml

5. [Optional] Install last year's agents for STD/COLLUSION tracks only

>> pip install scml-agents

6. Run the script with no parameters (assuming you are )

>> python /{path-to-this-file}/myagent.py

You should see a short tournament running and results reported.

"""

from collections import defaultdict
# required for development
from numpy.lib.arraysetops import isin
from scml.oneshot import OneShotAgent
import itertools
from scml.oneshot.world import SCML2020OneShotWorld
from scml.oneshot.world import is_system_agent
import ast

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

# other libraries
import sys
import math
import numpy as np
import pickle
import argparse
import csv
import random


# Flags
DEBUG = False #True
DEBUG2 = False
ENABLE_GRAPH = False #True
TO_SUBMISSION = True #False

# enabling inspection graphs
if ENABLE_GRAPH:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    from scipy.signal import butter,filtfilt

# Debug print type #1
def DEBUG_PRINT(msg):
    if DEBUG:
        print(msg)

# Debug print type #2
def DEBUG2_PRINT(msg):
    if DEBUG2:
        print(msg)

# Offer field enum
class OFFER_FIELD_IDX:
    QUANTITY = 0
    TIME = 1
    UNIT_PRICE = 2

# State type enum
# Some are used directly, some are representing a range of values (expanded within the agent's code)
class STATE_TYPE:
    NO_NEGO = -1
    INIT = 0
    #OPP_OVERRIDE = 1 
    END = 2
    ACCEPT = 3  
    MY_COUNTER = 4 
    OPP_COUNTER = 5

""""
The QlAgent class
-----------------
The QlAgent class implements our agent for the SCML competition. 
The QlAgent implements an SCML negotation agent by implementing Q-Learning algorithm 
with adaptation to the Negotation problem and SCML framework.
"""
class QlAgent(OneShotAgent):
    """
    Contructor
    """
    def __init__(self, load_q=True,                     # load Q table from external file (pickle)
                    load_q_type="exact",            # loading mode - "exact" for loading directly from the file without adaptation
                    save_q=False,                   # save Q table to external file (pickle)
                    save_q_type="exact",            # saving mode - "exact" for saving directly from the file without adaptation
                    alpha=0.1,                      # Q learning alpha parameter (learning rate)
                    prune_quan=1.2,                 # used to reduce the amount of quantities the agent can offer
                    gamma=0.95,                     # Q learning gamma parameter
                    price_res=5,                    # price resolution (amount of prices that can be offered / used in state)
                    quantity_res=5,                 # quantity resolution (amount of quantities that can be offered / used in state)
                    needs_res=5,                    # needs resolution (amount of possible values to represent needs status - how much do I left to sell/buy)
                    epsilon_start=0.001,#01,            # epsilon-greedy start value 
                    epsilon_end=0.0001,             # epsilon-greedy end value
                    epsilon_decay=10,              # epsilon-greedy decay value
                    smart_init=True,                # should always be True. Makes the Q table init with good values, if no external Q file exists
                    complex_state=True,            # should always be True. Makes the state space rich and represent the actual env state more accurately (the states are explained in the report)
                    concession_exponent=0.3,#None,
                    acc_price_slack=float("inf"),
                    step_price_slack=0.0,
                    opp_price_slack=0.0,
                    opp_acc_price_slack=0.2,
                    range_slack = 0.03,
                    epsilon2_start=0.5,            # epsilon2-greedy start value 
                    epsilon2_end=0.01,#0.1,#0.01             # epsilon2-greedy end value
                    epsilon2_decay=200,              # epsilon2-greedy decay value
                    *args, **kwargs
                    ):
        super().__init__(*args, **kwargs)
        # attributes init
        self.secured = 0 
        self.started = dict()
        self.opposite_price_gap = dict()
        self.success = 0 
        self.failure = 0
        self.profit = 0
        self.state_per_opp = dict()
        self.last_a_per_opp = dict()
        self.q_table_per_opp = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.price_res = price_res
        self.quantity_res = quantity_res
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_vec = dict()
        self.r_after_episode = dict()
        self.q_after_episode = dict()
        self.q_steps = dict()
        self.profit_per_opp = dict()
        self.smart_init = smart_init
        self.complex_state = complex_state
        self.episodes = 0
        self.num_of_negotiations_per_step = dict()
        self.num_of_negotiations_per_step_all = 0
        self.prune_quan = prune_quan
        self.needs_res = needs_res
        self.load_q = load_q
        self.load_q_type = load_q_type
        self.save_q = save_q
        self.save_q_type = save_q_type
        self.is_seller = True
        self.epsilon2_start = epsilon2_start
        self.epsilon2_end = epsilon2_end
        self.epsilon2_decay = epsilon2_decay

        self.counters = dict()
        for i in range(10):
            self.counters[i+1] = 0
        
        # loading external Q tables (per "seller" / "buyer")
        # "seller" = L0 position for agent
        # "buyer" = L1 position for agent
        # both may require different learning, because goals are opposite from negotiation price perspective.
        # that's why we have learning database for each type.
        if self.load_q:
            try:
                # with open('q_seller.pickle', 'rb') as handle:
                #     self.learned_q_seller_t = pickle.load(handle)
                with open('q_seller.txt', 'r') as handle_s:
                    self.learned_q_seller_t = ast.literal_eval(handle_s.read())
                DEBUG_PRINT("UPLOADED FROM q_seller.pickle")
            except FileNotFoundError:
                self.load_q = False

            try: 
                # with open('q_buyer.pickle', 'rb') as handle:
                #     self.learned_q_buyer_t = pickle.load(handle)
                with open('q_buyer.txt', 'r') as handle_b:
                    self.learned_q_buyer_t = ast.literal_eval(handle_b.read())
                DEBUG_PRINT("UPLOADED FROM q_buyer.pickle")
            except FileNotFoundError:
                self.load_q = False

        if concession_exponent is None:
            concession_exponent = 0.2 + random.random() * 0.8
        if step_price_slack is None:
            step_price_slack = random.random() * 0.1 + 0.05
        if opp_price_slack is None:
            opp_price_slack = random.random() * 0.1 + 0.05
        if opp_acc_price_slack is None:
            opp_acc_price_slack = random.random() * 0.1 + 0.05
        if range_slack is None:
            range_slack = random.random() * 0.2 + 0.05

        self._e = concession_exponent
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack

        self.num_of_nego_total = 0

    def init(self):
        """Initialize the quantities and best prices received so far"""
        self.ne_range = np.concatenate((np.array([-1]),np.linspace(0, self.awi.n_lines+1, self.needs_res)))
        
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))
        self._sales = self._supplies = 0
        


    """
    A method called in each step of the simulation
    """
    def step(self):
        """Initialize the quantities and best prices received for next step"""
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._sales = self._supplies = 0

        #print(self.state_per_opp)
        # if self.awi.current_step == 0:
        #     print("-----------------------------------------------")
        DEBUG_PRINT("# STEP : " + str(self.awi.current_step))
        DEBUG_PRINT("# Episodes  : " + str(self.episodes))
        DEBUG_PRINT("# Needed : " + str(self._needed()))

        if self.awi.current_step % 100 == 0:
            DEBUG2_PRINT("# STEP : " + str(self.awi.current_step))
            DEBUG2_PRINT(self.counters)

        
        for partner in self.started.keys():
            DEBUG_PRINT(partner)
        #    print(self.q_table_per_opp[partner])
        # print("s: " + str(self.success) + ", f: " + str(self.failure))
        # #print(self.current_agreed_price)
        #print("profit: " + str(self.profit))
        # print(self.q_table_per_opp)
        
        self.secured = 0

        # various actions for inspection graphs purposes
        for partner in self.r_after_episode.keys():
            self.r_after_episode[partner].append(0)
        
        for partner in self.q_steps.keys():
            self.q_steps[partner] = 0

        # saving Q tables if enabled
        if self.awi.current_step == self.awi.n_steps-1:
            if self.save_q:
                if self.is_seller:
                    postfix = "_seller"
                else:
                    postfix = "_buyer"
                with open('q'+postfix+'.txt', 'w') as handle: #.pickle
                    if self.save_q_type == "exact":
                        my_q = list(self.q_table_per_opp.values())[0]
                        if self.load_q:
                            # if we came from a loaded table, we add entries from there we don't have in our new 
                            # learned table, in order to gather as much Q value as possible for different scenarios
                            if self.is_seller:
                                for s in self.learned_q_seller_t:
                                    if s not in my_q.keys():
                                        my_q[s] = self.learned_q_seller_t[s]
                                    else:
                                        for a in self.learned_q_seller_t[s]:
                                            if a not in my_q[s].keys():
                                                my_q[s][a] = self.learned_q_seller_t[s][a]
                            else:
                                for s in self.learned_q_buyer_t:
                                    if s not in my_q.keys():
                                        my_q[s] = self.learned_q_buyer_t[s]
                                    else:
                                        for a in self.learned_q_buyer_t[s]:
                                            if a not in my_q[s].keys():
                                                my_q[s][a] = self.learned_q_buyer_t[s][a]
                        
                        handle.write(str(my_q))
                        #pickle.dump(my_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        handle.write(str(self.map_q(list(self.q_table_per_opp.values())[0])))
                        #pickle.dump(self.map_q(list(self.q_table_per_opp.values())[0]), handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # drawing graphs if enabled
            if ENABLE_GRAPH:
                for partner in self.r_after_episode.keys():
                    plt.plot(self.r_after_episode[partner])
                    b, a = butter(2, 0.01, btype='low', analog=False)
                    y = filtfilt(b, a, self.r_after_episode[partner])
                    plt.plot(y)
                    #plt.plot(np.convolve(self.r_after_episode[partner], np.ones(int(self.awi.n_steps/4))/int(self.awi.n_steps/4), mode='valid'))
                #plt.show()
                
                    #plt.plot(self.r_after_episode[partner])
                # for partner in self.q_after_episode.keys():
                #     plt.plot(self.q_after_episode[partner])
                # plt.show()
                # for partner in self.profit_per_opp.keys():
                #     #plt.plot(self.profit_per_opp[partner])
                #     plt.plot(np.convolve(self.profit_per_opp[partner], np.ones(int(self.awi.n_steps/4))/int(self.awi.n_steps/4), mode='valid'))
                plt.show()
                


    """
    Called when negotiation succeeded
    :param contract - the contract that was accepted
    :param mechanism - an object from which we can know various negotiation info
    """
    def on_negotiation_success(self, contract, mechanism):
        self.num_of_nego_total += 1
        DEBUG_PRINT("^^^^^ start of on_negotiation_success " + str(id(self)))
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        self.success += 1
        self.episodes += 1
        #self.profit += contract.agreement["unit_price"]*contract.agreement["quantity"]
        self.secured += contract.agreement["quantity"]

        # print("SUCCESS to QL")
        # print("QL price : " + str(contract.agreement["unit_price"]))
        # print("QL quantity : " + str(contract.agreement["quantity"]))
        
        # Getting partner ID and calculating profit (or good estimation to it) we got from this negotation
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            cur_profit = (contract.agreement["unit_price"]-float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity-self.awi.profile.cost)*contract.agreement["quantity"]#-self.awi.current_exogenous_input_price#*self.awi.current_exogenous_input_quantity
            fctr = ((contract.agreement["unit_price"]))#**2)
        else:
            partner = contract.annotation["seller"]
            cur_profit = (float(self.awi.current_exogenous_output_price)/self.awi.current_exogenous_output_quantity-contract.agreement["unit_price"]-self.awi.profile.cost)*contract.agreement["quantity"]#*self.awi.current_exogenous_output_quantity
            fctr = -((contract.agreement["unit_price"]))#**2)

        #cur_profit = cur_profit*(1+1.02**(self.num_of_nego_total-100))

        DEBUG_PRINT("on_negotiation_success, " + partner)

        ami = self.get_ami(partner)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
        # print("QL p range : " + str(unit_price_issue))
        # print("QL q range : " + str(quantity_issue))
        DEBUG_PRINT("state: " + str(self.state_per_opp[partner]))
        DEBUG_PRINT("action: " + str(self.last_a_per_opp[partner]))
        # for s in self.q_table_per_opp[partner].keys():
        #     for a in self.q_table_per_opp[partner][s].keys():
        #         if self.q_table_per_opp[partner][s][a] != 0:
        #             print(s, a, self.q_table_per_opp[partner][s][a])

        if partner not in self.profit_per_opp:
            self.profit_per_opp[partner] = [cur_profit]
        else:
            self.profit_per_opp[partner].append(cur_profit)

        #DEBUG_PRINT("on_negotiation_success " + partner)
        DEBUG_PRINT("--------------------------------")
        DEBUG_PRINT("SUCCESS <=>" + partner)
        
        DEBUG_PRINT("price: " + str(contract.agreement["unit_price"]))
        if self._is_selling(mechanism):
            DEBUG_PRINT("exo price (sell): " + str(float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity))
        else:
            DEBUG_PRINT("exo price (buy): " + str(float(self.awi.current_exogenous_output_price)/self.awi.current_exogenous_output_quantity))
        DEBUG_PRINT("cost :" + str(self.awi.profile.cost))

        # getting current needs
        my_needs = self._needed()
        #print("QL needs: " + str(my_needs))
        self.q_steps[partner] += 1
        if partner not in self.num_of_negotiations_per_step:
            self.num_of_negotiations_per_step[partner] = 0

        # updating num of negotations, for scaling purposes
        self.num_of_negotiations_per_step[partner] += 1
        self.num_of_negotiations_per_step_all += 1

        # Setting the correct reward per case, and updateing Q-table
        if my_needs == 0:
            # No more needs - we sold/bought everything
            DEBUG_PRINT("success : no needs")
            
            # setting reward
            if cur_profit < 0:
                r = cur_profit #+ fctr#10*cur_profit# + fctr
                self.counters[1] += 1
            else:
                r = cur_profit #+ fctr#100*cur_profit# + fctr
                self.counters[2] += 1
            #r = r/(unit_price_issue.max_value*quantity_issue.max_value)

            # updating Q table
            
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.ACCEPT)
        elif my_needs < 0:
            # No more needs - we sold/bought more than needed
            DEBUG_PRINT("success : - needs")
            
            # setting reward
            # fctr + 
            r = cur_profit+my_needs*self._too_much_penalty(mechanism)/self.num_of_negotiations_per_step_all#self.num_of_negotiations_per_step[partner]#*self.awi.current_step
            #print(my_needs, r)
            #r = r/(unit_price_issue.max_value*quantity_issue.max_value)
            self.counters[3] += 1
            # updating Q tables
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.ACCEPT)
        else: # my_needs > 0
            # Still there are needs
            DEBUG_PRINT("success : + needs")
            
            # setting reward
            # fctr + 
            r = cur_profit-my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step_all#self.num_of_negotiations_per_step[partner]#*self.awi.current_step
            #r = r/(unit_price_issue.max_value*quantity_issue.max_value)
            self.counters[4] += 1
            # updating Q tables
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.ACCEPT)
        
        DEBUG_PRINT("state: " + str(self.state_per_opp[partner]))
        DEBUG_PRINT("action: " + str(self.last_a_per_opp[partner]))
        DEBUG_PRINT("reward: " + str(r))
        
        if partner not in self.r_after_episode:
            self.r_after_episode[partner] = [0]

        if partner not in self.q_after_episode:
            self.q_after_episode[partner] = [0]
        
        self.r_after_episode[partner][-1] = (self.r_after_episode[partner][-1]+r)#/self.q_steps[partner]
        
        # setting current state correctlys
        self.state_per_opp[partner] = STATE_TYPE.ACCEPT
        self.last_a_per_opp[partner] = None


        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if contract.annotation["product"] == self.awi.my_output_product:
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

        
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        DEBUG_PRINT("^^^^^ end of on_negotiation_success " + str(id(self)))
        
    """
    Called when negotiation fails
    :param partners - unused
    :param annotation - an object from which we can know various info like partner ID
    :param mechanism - an object from which we can know various negotiation info
    """
    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        DEBUG_PRINT("^^^^^ start of on_negotiation_failure " + str(id(self)))
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        
        # Getting partner ID and calculating profit (or good estimation to it) we got from this negotation
        if self._is_selling(mechanism):
            is_seller = True
            partner = annotation["buyer"]
        else:
            is_seller = False
            partner = annotation["seller"]

        DEBUG_PRINT("on_negotiation_failure, " + partner)

        
        ami = self.get_ami(partner)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]

        # Various filters to avoid errorneus cases
        if partner not in self.state_per_opp:
            return None

        if self.state_per_opp[partner] == STATE_TYPE.NO_NEGO:
            return 

        if self.last_a_per_opp is None:
            return
        
        if partner not in self.last_a_per_opp:
            return 


        self.num_of_nego_total += 1

        self.episodes += 1
        self.failure += 1   

        
        

        #DEBUG_PRINT("on_negotiation_failure " + partner)
        DEBUG_PRINT("--------------------------------")
        DEBUG_PRINT("FAILURE <=>" + partner)

        my_needs = self._needed()
        self.q_steps[partner] += 1
        # if partner not in self.num_of_negotiations_per_step:
        #     self.num_of_negotiations_per_step[partner] = 0
        # self.num_of_negotiations_per_step[partner] += 1

        # Setting the correct reward per case, and updateing Q-table
        if my_needs == 0:
            # No more needs - we sold/bought everything

            # setting reward
            r = 0.5*unit_price_issue.max_value*quantity_issue.max_value #0.7

            self.counters[5] += 1
            # update Q-table
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.END)
        elif my_needs < 0:
            # No more needs - we sold/bought more than needed
            DEBUG_PRINT("need -")
            # 0.1*

            # setting reward
            if isinstance(self.last_a_per_opp[partner], str):
                if self.last_a_per_opp[partner] == "end":
                    r = 0.5*unit_price_issue.max_value*quantity_issue.max_value #0.7 #1*100
                    self.counters[6] += 1
                else:
                    r = 0#0.5*unit_price_issue.max_value*quantity_issue.max_value
                    self.counters[7] += 1
                    print("not supposed to happen")
            else:
                r = -0.5*unit_price_issue.max_value*quantity_issue.max_value #-1*100
             
            #r = 1*10#
            #r -= my_needs*self._too_much_penalty(mechanism)/self.num_of_negotiations_per_step_all#self.num_of_negotiations_per_step[partner]#*self.awi.current_step
            
            # update Q-table
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.END)
        else: # my_needs > 0
            # There are still needs to sell/buy
            DEBUG_PRINT("needs +")

            # setting reward
            if isinstance(self.state_per_opp[partner], tuple):
                #if isinstance(self.last_a_per_opp, str):
                if len(self.state_per_opp[partner]) == 5: #4:
                    self.counters[8] += 1
                    if is_seller: 
                        r = self.state_per_opp[partner][0]-self.state_per_opp[partner][2] #-1000*my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step[partner]#*self.awi.current_step
                    else:
                        r = self.state_per_opp[partner][2]-self.state_per_opp[partner][0]
                elif len(self.state_per_opp[partner]) == 3: #2:
                    self.counters[9] += 1
                    if is_seller:
                        r = unit_price_issue.max_value-self.state_per_opp[partner][0]
                    else:
                        r = self.state_per_opp[partner][0]-unit_price_issue.min_value 
                else:
                    print("not supposed to happen")    
                #r = r/self.awi.current_step#*10
                DEBUG_PRINT("case 1")
                r = r/(unit_price_issue.max_value)
                if r == 0:
                    r = 1    
                r = 0.5*r*quantity_issue.max_value*unit_price_issue.max_value*(1-(2.0*my_needs)/(quantity_issue.min_value+quantity_issue.max_value))
                #else:
                #r -= my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step[partner]
            else:
                self.counters[10] += 1
                DEBUG_PRINT("case 2")
                r = 0
                #r = -my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step[partner]

            DEBUG_PRINT("state: " + str(self.state_per_opp[partner]))
            DEBUG_PRINT("action: " + str(self.last_a_per_opp[partner]))
            DEBUG_PRINT("reward: " + str(r))
            # for k in self.q_table_per_opp[partner].keys():
            #     print(self.q_table_per_opp[partner][k])

            # updating Q-table
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.END)
        
        # Setting the correct state
        self.state_per_opp[partner] = STATE_TYPE.END
        #print("reward: ", r)
        
        if partner not in self.profit_per_opp:
            self.profit_per_opp[partner] = [0]
        else:
            self.profit_per_opp[partner].append(0) 

        if partner not in self.r_after_episode:
            self.r_after_episode[partner] = [0]

        self.r_after_episode[partner][-1] = (self.r_after_episode[partner][-1]+r)#/self.q_steps[partner]
        
        self.last_a_per_opp[partner] = None

        
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        DEBUG_PRINT("^^^^^ end of on_negotiation_failure " + str(id(self)))

    """
    Called in order so our agent will propose to an opposing one
    :param: negotiator_id - the ID of the opposing negotiator
    :param: state - unused (not related to Q-learning)
    :return: the offer to be proposed
    """        
    def propose(self, negotiator_id: str, state) -> "Outcome":
        #print("# STEP : " + str(self.awi.current_step))
        DEBUG_PRINT("^^^^^ start of propose " + negotiator_id  + " " + str(id(self)))
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        #print("propose " + negotiator_id)

        # Getting our current needs
        my_needs = self._needed(negotiator_id)
        #DEBUG_PRINT("needs : " + str(my_needs))
        ami = self.get_ami(negotiator_id)
        if not ami:
            print("No AMI !")
            return None
        
        # Getting opposing partner ID
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            is_seller = True
        else:
            partner = ami.annotation["seller"]
            is_seller = False

        self.is_seller = is_seller
        # if partner in self.finished:
        #     return None
        DEBUG_PRINT("propose " + partner)
        DEBUG_PRINT("------------------------------")
        #self._init_opposites_if_needed(ami)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]

        # initializing offer structure
        offer = [-1] * 3
        
        # checking if we already encountered this partner
        if partner not in self.started.keys():
            # first proposal with this negotiator. this means first proposal in the simulation 
            # against it.
            DEBUG_PRINT("INIT, " + partner)
            self.started[partner] = True
            self.q_table_per_opp[partner] = dict()
            #print(self.q_table_per_opp)
            price_gap = unit_price_issue.max_value-unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value-quantity_issue.min_value

            # initializing an action vector with all action and a Q table against this partner.
            self.action_vec[partner] = []
            #self.q_table_per_opp[partner], self.action_vec[partner] = self._q_learning_q_init(self.q_table_per_opp[partner], unit_price_issue, quantity_issue, self.action_vec[partner], is_seller, ami)
            self._q_learning_q_init(self.q_table_per_opp[partner], unit_price_issue, quantity_issue, self.action_vec[partner], is_seller, ami)
            

            #print("a", partner, len(self.q_table_per_opp[partner]))

            # if no needs, this negotiation is skipped
            if my_needs <= 0:
                self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
                DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
                DEBUG_PRINT("^^^^^ end of propose 1" + negotiator_id  + " " + str(id(self)))
                return None

            # initializing the state
            if not self.complex_state:
                self.state_per_opp[partner] = STATE_TYPE.INIT
            else:
                self.state_per_opp[partner] = str(self._find_nearest(self.ne_range, my_needs)) #str(int(np.ceil(my_needs)))

            self.q_steps[partner] = 1
            
        else:
            #print("b", partner, len(self.q_table_per_opp[partner]))
            # not first proposal with this negotiator
            if partner not in self.state_per_opp:
                # this is not supposed to happen - there should be coordination between the different
                # data structs in use against a partner
                print("issue, " + str(self.state_per_opp.keys()) +", " +str(self.started))
                return None 
            if self.state_per_opp[partner] == STATE_TYPE.END or self.state_per_opp[partner] == STATE_TYPE.ACCEPT or self.state_per_opp[partner] == STATE_TYPE.NO_NEGO:
                # This means that a negotiation against this partner just finished, so starting from beginning
                #DEBUG_PRINT("INIT")

                # if no needs, this negotiation is skipped
                if my_needs <= 0:
                    self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                    DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
                    DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
                    DEBUG_PRINT("^^^^^ end of propose 2" + negotiator_id  + " " + str(id(self)))
                    return None

                # last state was terminal, thus we now start new q learning step
                if not self.complex_state:
                    self.state_per_opp[partner] = STATE_TYPE.INIT
                else:
                    self.state_per_opp[partner] = str(self._find_nearest(self.ne_range, my_needs)) #str(int(np.ceil(my_needs)))
                # price_gap = unit_price_issue.max_value-unit_price_issue.min_value
                # quantity_gap = quantity_issue.max_value-quantity_issue.min_value
                # self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap)

                self.q_steps[partner] = 1

            # if we get to the "elif", this means this is during negotiation. 
            elif self.state_per_opp[partner] == STATE_TYPE.OPP_COUNTER or isinstance(self.state_per_opp[partner], tuple):
                # We are in the middle of negotiation (got here to offer something in return to partner's offer - we give a counter offer)
                
                #DEBUG_PRINT("OPP_COUNTER")
                #print("propose: " + str(self.state_per_opp[partner]))
                # after the respond 
                # update q
                #self._q_learning_update_q(STATE_TYPE.OPP_COUNTER, self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, self.state_per_opp[partner])
                #self.state_per_opp[partner] = STATE_TYPE.MY_COUNTER
                self.q_steps[partner] += 1
            else:
                # This case is in order to overcome issues in the infra, like propose after propose for same partner.
                # We just start from scratch.

                # print(self.state_per_opp[partner])
                # print("error - invalid state")
                # sys.exit(1)
                #DEBUG_PRINT("INIT2")
                self.started[partner] = True
                #self.q_table_per_opp[partner] = dict()
                #print(self.q_table_per_opp)
                price_gap = unit_price_issue.max_value-unit_price_issue.min_value
                quantity_gap = quantity_issue.max_value-quantity_issue.min_value
                #self.action_vec[partner] = []
                #self._q_learning_q_init(self.q_table_per_opp[partner], unit_price_issue, quantity_issue, self.action_vec[partner])

                
                if not self.complex_state:
                    self.state_per_opp[partner] = STATE_TYPE.INIT
                else:
                    self.state_per_opp[partner] = str(self._find_nearest(self.ne_range, my_needs)) #str(int(np.ceil(my_needs)))
                self.q_steps[partner] = 1


        #print("here6")
        # print(self.q_table_per_opp)
        # print(self.state_per_opp[partner])

        # setting action correctly (might been decided before)
        if not self.complex_state:
            if self.state_per_opp[partner] == STATE_TYPE.INIT:
                a = self._q_learning_select_action(self.q_table_per_opp[partner], self.state_per_opp[partner], self.action_vec[partner],
                                                caller="propose", negotiator_id=negotiator_id, state_sys=state, offer=None)
                self.last_a_per_opp[partner] = a
            else:
                # coming from OPP_COUNTER state, I already selected my action
                a = self.last_a_per_opp[partner]
                if a == "acc" or a == "end":
                    print("should not happen")
                    return None
        else:
            if isinstance(self.state_per_opp[partner], str):
                a = self._q_learning_select_action(self.q_table_per_opp[partner], self.state_per_opp[partner], self.action_vec[partner],
                                                caller="propose", negotiator_id=negotiator_id, state_sys=state, offer=None)
                self.last_a_per_opp[partner] = a
            else:
                # coming from OPP_COUNTER state, I already selected my action
                a = self.last_a_per_opp[partner]
                if a == "acc" or a == "end":
                    print("should not happen")
                    return None

        #print("here7")
        DEBUG_PRINT("-------------------------")
        DEBUG_PRINT("PROPOSE -->" + partner)
        DEBUG_PRINT(ami.annotation["buyer"] + " " + ami.annotation["seller"])
        DEBUG_PRINT("executed action : " + str(a))
        DEBUG_PRINT("state : " + str(self.state_per_opp[partner]))

        # Building the offer
        offer[OFFER_FIELD_IDX.UNIT_PRICE] = a[0]
        offer[OFFER_FIELD_IDX.QUANTITY] = a[1]
        offer[OFFER_FIELD_IDX.TIME] = self.awi.current_step

        # print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("# Needed to produce : " + str(self._needed()))
        # print("propose to " + str(negotiator_id) + ":" + str(self.best_offer(negotiator_id)))
        #print(partner)
        #print(self.q_table_per_opp[partner])
        #print("proposing !")
        
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        DEBUG_PRINT("^^^^^ end of propose 3" + negotiator_id  + " " + str(id(self)))
        return tuple(offer)

    """
    Called in order so our agent will respond to an opposing one
    :param: negotiator_id - the ID of the opposing negotiator
    :param: state - unused (not related to Q-learning)
    :param: offer - the offer given by the opposing partner
    :return: the response for the opponent partner (accept/reject(and offer something else in next propose())/end)
    """  
    def respond(self, negotiator_id, state, offer):
        #DEBUG_PRINT("respond " + negotiator_id)
        DEBUG_PRINT("^^^^^ start of respond " + negotiator_id  + " " + str(id(self)))
        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        # print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("# Needed to produce : " + str(self._needed()))

        # Getting current left needs
        my_needs = self._needed(negotiator_id)

        ami = self.get_ami(negotiator_id)
        if not ami:
            DEBUG_PRINT("No AMI !")
            return None

        # Getting the opposite partner ID
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self.is_seller = True
        else:
            partner = ami.annotation["seller"]
            self.is_seller = False

        # Filtering for debug / catch issues / avoid infra bugs
        if partner in self.last_a_per_opp:
            if self.last_a_per_opp[partner] == "acc" or self.last_a_per_opp[partner] == "end":
                #print("should not happen - we should have junped to success or failure methods")
                return None

        # Calculation potential profit - current this value won't be used
        if self._is_selling(ami):
            if self.awi.current_exogenous_input_price <= 0:
                #print(self.awi.current_exogenous_input_price)
                potential_profit = -1
            else:
                potential_profit = (offer[OFFER_FIELD_IDX.UNIT_PRICE]-float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity-self.awi.profile.cost)*offer[OFFER_FIELD_IDX.QUANTITY]
            is_seller = True
        else:
            if self.awi.current_exogenous_output_quantity <= 0:
                #print(self.awi.current_exogenous_output_quantity)
                potential_profit = -1
            else:   
                potential_profit = (float(self.awi.current_exogenous_output_price)/self.awi.current_exogenous_output_quantity-offer[OFFER_FIELD_IDX.UNIT_PRICE]-self.awi.profile.cost)*offer[OFFER_FIELD_IDX.QUANTITY]
            is_seller = False
        
        # Check if this the first encounter with this partner
        if partner not in self.state_per_opp:
            # it is the first encounter. Init everything - state, action vec, Q table
            #return None
            self.q_steps[partner] = 0
            self.started[partner] = True
            if not self.complex_state:
                self.state_per_opp[partner] = STATE_TYPE.INIT
            else:
                self.state_per_opp[partner] = str(self._find_nearest(self.ne_range, my_needs)) #str(int(np.ceil(my_needs)))
            self.q_table_per_opp[partner] = dict()
            unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
            quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
            price_gap = unit_price_issue.max_value-unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value-quantity_issue.min_value
            self.action_vec[partner] = []
            #self.q_table_per_opp[partner], self.action_vec[partner] = self._q_learning_q_init(self.q_table_per_opp[partner], unit_price_issue, quantity_issue, self.action_vec[partner], is_seller, ami)
            self._q_learning_q_init(self.q_table_per_opp[partner], unit_price_issue, quantity_issue, self.action_vec[partner], is_seller, ami)

        #print("respond: " + str(self.state_per_opp[partner]))

        # Filtering unneeded calls to respond() in case nothing to do
        if partner in self.state_per_opp:
            if self.state_per_opp[partner] == STATE_TYPE.NO_NEGO:
                return ResponseType.REJECT_OFFER

        #print("responding !")
        DEBUG_PRINT("-------------------------")
        DEBUG_PRINT("RESPOND <--" + partner)
        if partner in self.last_a_per_opp:
            DEBUG_PRINT("my last offer : " + str(self.last_a_per_opp[partner]))
        DEBUG_PRINT("new offer : " + str([offer[OFFER_FIELD_IDX.UNIT_PRICE], offer[OFFER_FIELD_IDX.QUANTITY]]))
        DEBUG_PRINT("last state : " + str(self.state_per_opp[partner]))
        

        if not self.complex_state:
            new_state = STATE_TYPE.OPP_COUNTER
        else:
            # Calculating new state (mapping from env status to best matching state in Q table)
            # Setting the new state if needed
            unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
            quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
            price_gap = unit_price_issue.max_value-unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value-quantity_issue.min_value
            #print(self.last_a_per_opp)
            if self.last_a_per_opp == {} or self.last_a_per_opp is None or \
                partner not in self.last_a_per_opp or \
                self.last_a_per_opp[partner] is None:
                # This case is when we init here, first encounter or starting new nogotiation.

                new_state = self._find_state_mapping(0, 
                                                    0,
                                                    offer[OFFER_FIELD_IDX.UNIT_PRICE],
                                                    offer[OFFER_FIELD_IDX.QUANTITY],
                                                    unit_price_issue, quantity_issue, my_needs)[2:]
                if not self.complex_state:
                    self.state_per_opp[partner] = STATE_TYPE.INIT
                else:
                    self.state_per_opp[partner] = str(self._find_nearest(self.ne_range, my_needs)) #str(int(np.ceil(my_needs)))
            else:
                # We're in the middle of existing negotiation
                new_state = self._find_state_mapping(self.last_a_per_opp[partner][0], 
                                                    self.last_a_per_opp[partner][1],
                                                    offer[OFFER_FIELD_IDX.UNIT_PRICE],
                                                    offer[OFFER_FIELD_IDX.QUANTITY],
                                                    unit_price_issue, quantity_issue, my_needs)
                #print(self.last_a_per_opp[partner])
                # update q 
                #print("update 1")

                # Since we're in the middle of a negotiation process (Q learning episode) - update Q-table
                self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, new_state)

        
        self.q_steps[partner] += 1
            
        
        # progress state
        self.state_per_opp[partner] = new_state

        
        # choose new action 
        a = self._q_learning_select_action(self.q_table_per_opp[partner], new_state, self.action_vec[partner],
                                                caller="respond", negotiator_id=negotiator_id, state_sys=state, offer=offer)

        # DEBUG_PRINT("new state : " + str(self.state_per_opp[partner]))
        # DEBUG_PRINT("next action : " + str(a))
        # in case no more needs - end
        # if my_needs <= 0:
        #     a = "end"

        
        up = offer[OFFER_FIELD_IDX.UNIT_PRICE]
        if self._is_selling(ami):
            self._best_selling = max(up, self._best_selling)
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = self._best_selling
        else:
            self._best_buying = min(up, self._best_buying)
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = self._best_buying

        DEBUG_PRINT("started: " + str(self.started) + " " + str(id(self)))
        DEBUG_PRINT("state_per_opp: " + str(self.state_per_opp) + " " + str(id(self)))
        DEBUG_PRINT("^^^^^ end of respond " + negotiator_id  + " " + str(id(self)))

        # remember action for next method calls to update the Q-learning DBs and execute needed actions 
        # if needed there. Return the correct negotiation responce as needed by the framework, according to 
        # the action needed to be executed.
        self.last_a_per_opp[partner] = a
        if isinstance(a, str):
            # state will be updated in on_failure and in on_success
            if a == "end":
                return ResponseType.END_NEGOTIATION
            elif a == "acc":
                return ResponseType.ACCEPT_OFFER
            
        else:
            # this means we're going to propose()
            #print("rejecting !")
            return ResponseType.REJECT_OFFER

    



        
    """
    Q-Learning update - a method to update the Q table according to algorithm
    :param: state - QL. state
    :param: action - QL. action
    :param: q - Q table
    :param: reward - the reward obtained
    :param: new_state - the new state
    """
    def _q_learning_update_q(self, state, action, q, reward, new_state):
        # print(id(self))
        # print(state)
        # print(action)
        # print(new_state)
        # print(reward)
        # print(len(q.keys()))
        # print("--------")
        #q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]) - q[state][action])
        if state in q:
            if action in q[state]:
                if new_state in q:
                    if q[new_state]:
                        q[state][action] = (1-self.alpha)*q[state][action] + self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
                    else:
                        q[new_state][action] = 0
                        q[state][action] = self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
                else:
                    q[new_state] = dict()
                    q[new_state][action] = 0
                    q[state][action] = self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
            else:
                q[state] = dict()
                q[state][action] = 0
                if new_state in q:
                    if q[new_state]:
                        q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]) - q[state][action])
                    else:
                        q[new_state][action] = 0
                        q[state][action] = self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
                else:
                    q[new_state] = dict()
                    q[new_state][action] = 0
                    q[state][action] = self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
        else:
            q[state] = dict()
            q[state][action] = 0
            if action in q[state]:
                if new_state in q:
                    if q[new_state]:
                        q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]) - q[state][action])
                    else:
                        q[new_state][action] = 0
                        q[state][action] = self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
                else:
                    q[new_state] = dict()
                    q[new_state][action] = 0
                    q[state][action] = self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
            else:
                q[state] = dict()
                q[state][action] = 0
                if new_state in q:
                    if q[new_state]:
                        q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]) - q[state][action])
                    else:
                        q[new_state][action] = 0
                        q[state][action] = self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
                else:
                    q[new_state] = dict()
                    q[new_state][action] = 0
                    q[state][action] = self.alpha*(reward + self.gamma*max([q[new_state][a] for a in q[new_state].keys()]))
        
    """
    Q-Learning action selection - a method to select the next action according to algorithm
    :param: state - QL. state
    :param: q - Q table
    :param: the vector with all possible actions
    :return: the selected action
    """
    def _q_learning_select_action(self, q, state, action_vec, caller=None, negotiator_id=None, state_sys=None, offer=None):
        # DEBUG_PRINT("_q_learning_select_actionq.keys(state)
        
        # epsilon-greedy action selection with decay 
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                                          math.exp(-1. * (self.awi.current_step+1) / self.epsilon_decay)
        #self.awi.current_step

        # random sampling
        p = np.random.random()

        # act
        if p < eps_threshold:
            # select randomly
            if not self.complex_state:
                if state != STATE_TYPE.INIT:
                    #print(np.array(action_vec, dtype=object))
                    a = np.random.choice(np.array(action_vec, dtype=object))
                    #print(a)
                    return a
                else: 
                    #print(np.array(action_vec, dtype=object))
                    # without "acc" and "end" since not available in INIT
                    a = np.random.choice(np.array(action_vec, dtype=object)[:-2])
                    #print(a)
                    return a
            else:
                if not isinstance(state, str):
                    #print(np.array(action_vec, dtype=object))
                    a = np.random.choice(np.array(action_vec, dtype=object))
                    #print(a)
                    return a
                else: 
                    #print(np.array(action_vec, dtype=object))
                    # without "acc" and "end" since not available in INIT
                    a = np.random.choice(np.array(action_vec, dtype=object)[:-2])
                    #print(a)
                    return a
        else:
            q_max_greedy = False #False

            # epsilon-greedy action selection with decay 
            eps2_threshold = self.epsilon2_end + (self.epsilon2_start - self.epsilon2_end) * \
                                                            math.exp(-1. * (self.awi.current_step+1) / self.epsilon2_decay)

            #self.awi.current_step
            #num_of_nego_total
            # random sampling
            p2 = np.random.random()

            # if q_max_greedy:
            #if not self.is_seller:
            #if q_max_greedy:
            #if (self.is_seller and p2 < eps2_threshold) or (not self.is_seller):
            if p2 < eps2_threshold:
                #print("greedy")
                if caller == "propose":
                    return self.propose_greedy(negotiator_id, state_sys)
                if caller == "respond":
                    return self.respond_greedy(negotiator_id, state_sys, offer)
            else:
                #print("q")
                # select greedily
                max_q = max([q[state][action] for action in q[state].keys()])
                # print([q[state][action] for action in q[state].keys()])
                # print(max_q)
                for a in q[state].keys():
                    if q[state][a] == max_q:
                        #print(a)
                        return a

    """
    Q-Learning Q table init - a method to initialize the Q table
    :param: q_t - Q table
    :param: unit_price_issue - includes the min and max prices
    :param: quantity_issue - includes the min and max quantities
    :param: action_vec - an empty vector to be initialized with all possible actions
    :param: is_seller - True if I'm "seller" (L0)
    :param: ami - the framework's AMI
    :param: action_vec - the vector with all possible actions
    """
    def _q_learning_q_init(self, q_t, unit_price_issue, quantity_issue, action_vec, is_seller=True, ami=None):
        #DEBUG_PRINT("_q_learning_q_init")

        # print(quantity_issue)
        # print(unit_price_issue)

        # if self.q_table_per_opp:
        #     if self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]:
        #         #print(len(self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]))
        #         q_t = self.q_table_per_opp[list(self.q_table_per_opp.keys())[0]]
        #         action_vec = self.action_vec[list(self.q_table_per_opp.keys())[0]]
        #         return [q_t, action_vec]
            
        
        # limiting possible actions for the agent: this gives better performance
        min_q = quantity_issue.min_value + self.prune_quan*(quantity_issue.max_value-quantity_issue.min_value)/2 
        max_q = quantity_issue.max_value
        
        # Calculating ranges for all states and actions
        p_range = np.linspace(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res)
        my_q_range = np.linspace(min_q, max_q, self.quantity_res)#[0:int(np.ceil(self.quantity_res/5))]
        #print(my_q_range)
        q_range = np.linspace(quantity_issue.min_value, quantity_issue.max_value, self.quantity_res)
        
        # Going over all state type and building a basic table
        for s in [STATE_TYPE.INIT, STATE_TYPE.OPP_COUNTER, STATE_TYPE.END, STATE_TYPE.ACCEPT]:   # , STATE_TYPE.MY_COUNTER     
            if not self.complex_state:
                q_t[s] = dict()
            else:
                if s == STATE_TYPE.INIT:
                    #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                    for needs in self.ne_range:
                        q_t[str(needs)] = dict()
                elif s == STATE_TYPE.OPP_COUNTER:
                    for p_so in p_range:
                        for q_so in q_range:
                            for p_s in p_range:
                                for q_s in my_q_range:
                            
                                    #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                    for needs in self.ne_range:
                                        q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)] = dict()
                            #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                            for needs in self.ne_range:
                                q_t[(p_so,q_so,needs)] = dict()
                else:
                    q_t[s] = dict()

            # print(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res)
            # print(quantity_issue.min_value, quantity_issue.max_value, self.quantity_res)

            # initializing all values in the table
            for p in p_range:
                #print(q_t)
                for q in my_q_range:
                    #print(q_t)
                    #print((p,q))
                    if self.smart_init:
                        if is_seller:                            
                            if s == STATE_TYPE.INIT:
                               #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                for needs in self.ne_range:
                                    q_t[str(needs)][(p,np.ceil(q))] = self._initial_q_seller(p,np.ceil(q),needs, ami, str(needs))#10000
                            elif s == STATE_TYPE.OPP_COUNTER:
                                for p_so in p_range:
                                    for q_so in q_range:
                                        for p_s in p_range:
                                            for q_s in my_q_range:
                                                #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                                for needs in self.ne_range:
                                                    q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)][(p,np.ceil(q))] = self._initial_q_seller(p,np.ceil(q),needs, ami, (p_s,np.ceil(q_s),p_so,q_so,needs))#10000
                                        #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                        for needs in self.ne_range:
                                            q_t[(p_so,q_so,needs)][(p,np.ceil(q))] = self._initial_q_seller(p,np.ceil(q),needs, ami, (p_so,q_so,needs))#0 
                            else:
                                q_t[s][(p,np.ceil(q))] = 0#self._initial_q_seller(p,np.ceil(q),needs,ami)
                        else:
                            if s == STATE_TYPE.INIT:
                                #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                for needs in self.ne_range:
                                    q_t[str(needs)][(p,np.ceil(q))] = self._initial_q_buyer(p,np.ceil(q),needs, ami, str(needs))#10000
                            elif s == STATE_TYPE.OPP_COUNTER:
                                for p_so in p_range:
                                    for q_so in q_range:
                                        for p_s in p_range:
                                            for q_s in my_q_range:
                                                #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                                for needs in self.ne_range:
                                                    q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)][(p,np.ceil(q))] = self._initial_q_buyer(p,np.ceil(q),needs, ami, (p_s,np.ceil(q_s),p_so,q_so,needs))#10000
                                        #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                        for needs in self.ne_range:
                                            q_t[(p_so,q_so,needs)][(p,np.ceil(q))] = self._initial_q_buyer(p,np.ceil(q),needs, ami, (p_so,q_so,needs))#0 
                            else:
                                q_t[s][(p,np.ceil(q))] = 0#self._initial_q_buyer(p, np.ceil(q), )
                    else:
                        if s == STATE_TYPE.INIT:
                            #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                            for needs in self.ne_range:
                                q_t[str(needs)][(p,np.ceil(q))] = 0#10000
                        elif s == STATE_TYPE.OPP_COUNTER:
                            for p_so in p_range:
                                for q_so in q_range:
                                    for p_s in p_range:
                                        for q_s in my_q_range:
                                            #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                            for needs in self.ne_range:
                                                q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)][(p,np.ceil(q))] = 0
                                    #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                    for needs in self.ne_range:
                                        q_t[(p_so,q_so,needs)][(p,np.ceil(q))] = 0 
                        else:
                            q_t[s][(p,np.ceil(q))] = 0
            
            #if s == STATE_TYPE.OPP_COUNTER:
            #if s != STATE_TYPE.INIT:
            if s == STATE_TYPE.OPP_COUNTER:
                if not self.complex_state:
                    q_t[s]["end"] = 0
                    q_t[s]["acc"] = 0
                else:
                    for p_so in p_range:
                        for q_so in q_range:
                            for p_s in p_range:
                                for q_s in my_q_range:
                                    #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                                    for needs in self.ne_range:
                                        q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)]["end"] = 0
                                        if is_seller:
                                            q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)]["acc"] = self._initial_q_seller(p_so,q_so,needs,ami, (p_s,np.ceil(q_s),p_so,q_so,needs), "acc")#p_so*q_so-self.awi.current_exogenous_input_price
                                        else:
                                            q_t[(p_s,np.ceil(q_s),p_so,q_so,needs)]["acc"] = self._initial_q_buyer(p_so,q_so,needs,ami, (p_s,np.ceil(q_s),p_so,q_so,needs), "acc") #self.awi.current_exogenous_input_price-(p_so-p_s)*(q_so-q_s)
                            #for needs in range(-1,self.awi.n_lines+1,self.needs_res):
                            for needs in self.ne_range:
                                q_t[(p_so,q_so,needs)]["end"] = 0  
                                if is_seller:
                                    q_t[(p_so,q_so,needs)]["acc"] = self._initial_q_seller(p_so,q_so,needs,ami, (p_so,q_so,needs), "acc")
                                else:
                                    q_t[(p_so,q_so,needs)]["acc"] = self._initial_q_buyer(p_so,q_so,needs,ami, (p_so,q_so,needs), "acc") 

        # initializing the action vector
        for p in np.linspace(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res):
            for q in my_q_range:#np.linspace(min_q, quantity_issue.max_value, self.quantity_res):
                action_vec.append((p,np.ceil(q)))

        action_vec.append("end")
        action_vec.append("acc")

        #return [q_t, action_vec]
        #print("init ", len(q_t.keys()))

    """
    Map env's statuses to best matching state
    :param: p - my last offer's unit price
    :param: q - my last offer's quantity
    :param: po - opponent offer's unit price
    :param: qo - opponent offer's quantity
    :param: unit_price_issue - includes the min and max prices
    :param: quantity_issue - includes the min and max quantities
    :param: needs - current needs
    :return: the matching state
    """
    def _find_state_mapping(self, p, q, po, qo, unit_price_issue, quantity_issue, needs):
        #min_q = quantity_issue.min_value #+ self.prune_quan*(quantity_issue.max_value-quantity_issue.min_value)/2
        p_range = np.linspace(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res)
        q_range = np.linspace(quantity_issue.min_value, quantity_issue.max_value, self.quantity_res)
        
        if needs >= 0:
            return (p, q, self._find_nearest(p_range, po), self._find_nearest(q_range, qo), self._find_nearest(self.ne_range, needs))
        else:
            return (p, q, self._find_nearest(p_range, po), self._find_nearest(q_range, qo), -1)

    """
    Find nearest array item to value:
    :param: array - the array
    :param: value - the value we search closest item to it in the array
    :return: the found item closest to value
    """
    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    """
    Needed quanity still need to sell/buy (from exogenous contracts)
    :param: negotiator_id - unused
    :return: the needs
    """
    def _needed(self, negotiator_id=None):
        return self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity - \
               self.secured

    """
    Check if we are a "seller" (L0)
    :param: ami - the AMI object of the framework
    :return: True iff "seller", else "buyer"
    """
    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    """
    Penalty calculating for buying too much 
    :param: ami - the AMI object of the framework
    :return: the penalty per unit
    """
    # storage : buying to much == disposal, delivery : selling too much == shortfall
    def _too_much_penalty(self, ami):
        if self._is_selling(ami):
            return self.awi.profile.shortfall_penalty_mean#, self.awi.profile.shortfall_penalty_dev
        else:
            return self.awi.profile.disposal_cost_mean#, self.awi.profile.disposal_cost_dev

    """
    Penalty calculating for buying too less 
    :param: ami - the AMI object of the framework
    :return: the penalty per unit
    """  
    def _too_less_penalty(self, ami):
        if self._is_selling(ami):
            return self.awi.profile.disposal_cost_mean#, self.awi.profile.disposal_cost_dev
        else:
            return self.awi.profile.shortfall_penalty_mean#, self.awi.profile.shortfall_penalty_dev

    """
    Initialize Q values for "seller" (L0)
    :param: p - action's price
    :param: q - action's quantity
    :param: needs - the needs 
    :param: ami - AMI object
    :param: state - the state
    :param: terminal_action - in case not known, this is the action and not (p,q)
    :return: value to initialize in Q table for the state,action pair
    """
    def _initial_q_seller(self, p, q, needs, ami, state, terminal_action=None):
        if self.load_q:
            if self.load_q_type == "exact":
                if state in self.learned_q_seller_t:
                    if terminal_action == "acc":
                        if "acc" in self.learned_q_seller_t[state]:
                            return self.learned_q_seller_t[state]["acc"]
                    else:
                        if (p,q) in self.learned_q_seller_t[state]:
                            return self.learned_q_seller_t[state][(p,q)]
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        #return 0
        #if q <= needs:
        if needs < 0:
            return -unit_price_issue.max_value#*(q/needs)#p*q
        else:
            return p#p*q#p#p*q-(needs-q)*self._too_much_penalty(ami)

    """
    Initialize Q values for "buyer" (L1)
    :param: p - action's price
    :param: q - action's quantity
    :param: needs - the needs 
    :param: ami - AMI object
    :param: state - the state
    :param: terminal_action - in case not known, this is the action and not (p,q)
    :return: value to initialize in Q table for the state,action pair
    """
    def _initial_q_buyer(self, p, q, needs, ami, state, terminal_action=None):
        if self.load_q:
            if self.load_q_type == "exact":
                if state in self.learned_q_buyer_t:
                    if terminal_action == "acc":
                        if "acc" in self.learned_q_buyer_t[state]:
                            return self.learned_q_buyer_t[state]["acc"]
                    else:
                        if (p,q) in self.learned_q_buyer_t[state]:
                            return self.learned_q_buyer_t[state][(p,q)]
        #return 0
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        #if q <= needs:
        if needs < 0:
            return -unit_price_issue.max_value#*(needs/q)#q*(unit_price_issue.max_value-p)
        else:
            return unit_price_issue.max_value-p#q*(unit_price_issue.max_value-p)#-p#q*(unit_price_issue.max_value-p)-(needs-q)*self._too_much_penalty(ami)

    
    """
    Mapping vector of values linearly to fixed targets
    :param: v : (v0, v1, ...) - the values to map
    :param: mx_mn : [[max0, min0], [max1, min1], ...] - the values' ranges
    :param: sizes : [size0, size1, ...] - the number of possible values in each range
    :return: mapped value
    """
    def mapl_v(self, v, mx_mn, sizes):
        v_mapped = []
        for i, val in enumerate(v):
            v_mapped.append(sizes[i]*(float(val-mx_mn[i][0])/(mx_mn[i][1]-mx_mn[i][0])))
        return tuple(v_mapped)

    """
    Mapping scalar linearly to fixed target
    :param: z - the scalar to map
    :param: mx - the max of the z's range
    :param: mn - the min of the z's range
    :param: size - the number of possible values z's range
    :return: mapped value
    """
    def mapl_s(self, z, mx, mn, size):
        # print("mapl_s")
        # print(z)
        # print(mx)
        # print(mn)
        # print(size)
        return size*(float(z-mn)/(mx-mn))
    
    """
    Mapping Q table's keys linearly target Q with fixed global keys
    :param: q_t - the Q table
    :return: the mapped Q table 
    """
    def map_q(self, q_t):
        new_q = dict()
        tuple_5_s = []
        tuple_3_s = []
        for s in q_t.keys():
            if isinstance(s, tuple):
                if len(s) == 5:
                    tuple_5_s.append(s)
                elif len(s) == 3:
                    tuple_3_s.append(s)
                else:
                    print("fatal1")
            

        pv_s5 = [p for p, _, _, _, _ in tuple_5_s]
        qv_s5 = [q for _, q, _, _, _ in tuple_5_s]
        pv_o5 = [p for _, _, p, _, _ in tuple_5_s]
        qv_o5 = [q for _, _, _, q, _ in tuple_5_s]
        len_p_s5 = len(np.unique(pv_s5))
        len_q_s5 = len(np.unique(qv_s5))
        len_p_o5 = len(np.unique(pv_o5))
        len_q_o5 = len(np.unique(qv_o5))
        max_p_s5 = max(pv_s5)
        min_p_s5 = min(pv_s5)
        max_p_o5 = max(pv_o5)
        min_p_o5 = min(pv_o5)
        max_q_s5 = max(qv_s5)
        min_q_s5 = min(qv_s5)
        max_q_o5 = max(qv_o5)
        min_q_o5 = min(qv_o5)


        pv_o3 = [p for p, _, _ in tuple_3_s]
        qv_o3 = [q for _, q, _ in tuple_3_s]
        len_p_o3 = len(np.unique(pv_o3))
        len_q_o3 = len(np.unique(qv_o3))
        max_p_o3 = max(pv_o3)
        min_p_o3 = min(pv_o3)
        max_q_o3 = max(qv_o3)
        min_q_o3 = min(qv_o3)

        for s in q_t.keys():
            if isinstance(s, str):
                # INIT 
                pv_a = [p for p, _ in q_t[s].keys()]
                qv_a = [q for _, q in q_t[s].keys()]
                len_p_a = len(np.unique(pv_a))
                len_q_a = len(np.unique(qv_a))
                max_p_a = max(pv_a)
                min_p_a = min(pv_a)
                max_q_a = max(qv_a)
                min_q_a = min(qv_a)
                
                mapped_s = self.mapl_s(int(s), -1, self.awi.n_lines, self.needs_res)
                new_q[mapped_s] = dict()
                #print(list(q[s].keys()))
                for p, q in list(q_t[s].keys()):
                    # print(p)
                    # print(q)
                    mapped_p = self.mapl_s(p, max_p_a, min_p_a, len_p_a)
                    mapped_q = self.mapl_s(q, max_q_a, min_q_a, len_q_a)
                    # print(mapped_s, (mapped_p, mapped_q))
                    # print(new_q)
                    # print(q_t[s])
                    # print(q_t[s][(p, q)])
                    new_q[mapped_s][(mapped_p, mapped_q)] = q_t[s][(p, q)]

            

            elif isinstance(s, tuple):
                # TODO : need to work on all the 5-tuples under q, not q[s]. need to fix !
                actions = list(q_t[s].keys())
                if len(s) == 5:
                    # OPP 
                    pv_a = [p for p, _ in actions[:-2]]
                    qv_a = [q for _, q in actions[:-2]]
                    len_p_a = len(np.unique(pv_a))
                    len_q_a = len(np.unique(qv_a))
                    max_p_a = max(pv_a)
                    min_p_a = min(pv_a)
                    max_q_a = max(qv_a)
                    min_q_a = min(qv_a)

                    p_s, q_s, p_o, q_o, n = s
                    mapped_p_s = self.mapl_s(p_s, max_p_s5, min_p_s5, len_p_s5)
                    mapped_q_s = self.mapl_s(q_s, max_q_s5, min_q_s5, len_q_s5)
                    mapped_p_o = self.mapl_s(p_o, max_p_o5, min_p_o5, len_p_o5)
                    mapped_q_o = self.mapl_s(q_o, max_q_o5, min_q_o5, len_q_o5)
                    
                    mapped_s = self.mapl_s(n, -1, self.awi.n_lines, self.needs_res)
                    new_q[(mapped_p_s, mapped_q_s, mapped_p_o, mapped_q_o, mapped_s)] = dict()
                    for p, q in actions[:-2]:
                        mapped_p = self.mapl_s(p, max_p_a, min_p_a, len_p_a)
                        mapped_q = self.mapl_s(q, max_q_a, min_q_a, len_q_a)
                        new_q[(mapped_p_s, mapped_q_s, mapped_p_o, mapped_q_o, mapped_s)][(mapped_p, mapped_q)] = q_t[s][(p, q)]
                    new_q[(mapped_p_s, mapped_q_s, mapped_p_o, mapped_q_o, mapped_s)]["acc"] = q_t[s]["acc"]
                    new_q[(mapped_p_s, mapped_q_s, mapped_p_o, mapped_q_o, mapped_s)]["end"] = q_t[s]["end"]

                elif len(s) == 3:
                    # OPP 
                    pv_a = [p for p, _ in actions[:-2]]
                    qv_a = [q for _, q in actions[:-2]]
                    len_p_a = len(np.unique(pv_a))
                    len_q_a = len(np.unique(qv_a))
                    max_p_a = max(pv_a)
                    min_p_a = min(pv_a)
                    max_q_a = max(qv_a)
                    min_q_a = min(qv_a)

                    p_o, q_o, n = s
                    mapped_p_o = self.mapl_s(p_o, max_p_o3, min_p_o3, len_p_o3)
                    mapped_q_o = self.mapl_s(q_o, max_q_o3, min_q_o3, len_q_o3)
                    
                    mapped_s = self.mapl_s(n, -1, self.awi.n_lines, self.needs_res)
                    new_q[(mapped_p_o, mapped_q_o, mapped_s)] = dict()
                    for p, q in actions[:-2]:
                        mapped_p = self.mapl_s(p, max_p_a, min_p_a, len_p_a)
                        mapped_q = self.mapl_s(q, max_q_a, min_q_a, len_q_a)
                        new_q[(mapped_p_o, mapped_q_o, mapped_s)][(mapped_p, mapped_q)] = q_t[s][(p, q)]
                    new_q[(mapped_p_o, mapped_q_o, mapped_s)]["acc"] = q_t[s]["acc"]
                    new_q[(mapped_p_o, mapped_q_o, mapped_s)]["end"] = q_t[s]["end"]
                    
                else:
                    print("fatal2")

            elif s == STATE_TYPE.END or s == STATE_TYPE.ACCEPT:
                pv_a = [p for p, _ in actions[:-2]]
                qv_a = [q for _, q in actions[:-2]]
                len_p_a = len(np.unique(pv_a))
                len_q_a = len(np.unique(qv_a))
                max_p_a = max(pv_a)
                min_p_a = min(pv_a)
                max_q_a = max(qv_a)
                min_q_a = min(qv_a)
                
                new_q[s] = dict()
                for p, q in actions[:-2]:
                    mapped_p = self.mapl_s(p, max_p_a, min_p_a, len_p_a)
                    mapped_q = self.mapl_s(q, max_q_a, min_q_a, len_q_a)
                    new_q[s][(mapped_p, mapped_q)] = q_t[s][(p, q)]
                #new_q[s]["acc"] = q_t[s]["acc"]
                #new_q[s]["end"] = q_t[s]["end"]
                
            else:
                print("fatal3")
                # END , ACCEPT
        
        return new_q

    def best_offer(self, negotiator_id):
        my_needs = int(self._needed(negotiator_id))
        if my_needs <= 0:
            return None
        ami = self.get_ami(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        offer = [-1] * 3
        # mx = max(min(my_needs, quantity_issue.max_value), quantity_issue.min_value)
        # offer[OFFER_FIELD_IDX.QUANTITY] = random.randint(
        #     max(1, int(0.5 + mx * self.awi.current_step / self.awi.n_steps)), mx
        # )
        offer[OFFER_FIELD_IDX.QUANTITY] = max(
            min(my_needs, quantity_issue.max_value),
            quantity_issue.min_value
        )
        offer[OFFER_FIELD_IDX.TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[OFFER_FIELD_IDX.UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[OFFER_FIELD_IDX.UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _is_good_price(self, ami, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(ami):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, ami, state):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(ami):
            return int(mn + th * (mx - mn))
        else:
            return int(mx - th * (mx - mn))

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE].min_value
        mx = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE].max_value
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = ami.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        return int(mn), int(mx)

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e




    def propose_greedy(self, negotiator_id: str, state):
        # find the absolute best offer for me. This will most likely has an
        # unrealistic price
        offer = self.best_offer(negotiator_id)

        # if there are no best offers, just return None to end the negotiation
        if not offer:
            return None

        # over-write the unit price in the best offer with a good-enough price
        offer = list(offer)
        offer[OFFER_FIELD_IDX.UNIT_PRICE] = self._find_good_price(self.get_ami(negotiator_id), state)
        return (offer[OFFER_FIELD_IDX.UNIT_PRICE], offer[OFFER_FIELD_IDX.QUANTITY])

    def respond_greedy(self, negotiator_id, state, offer):
        # find the quantity I still need and end negotiation if I need nothing more
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return "end"

        # reject any offers with quantities above my needs
        response = (
            "acc"
            if offer[OFFER_FIELD_IDX.QUANTITY] <= my_needs
            else self.propose_greedy(negotiator_id, state)
        )
        if response != "acc":
            return response

        # reject offers with prices that are deemed NOT good-enough
        ami = self.get_ami(negotiator_id)
        response = (
            response
            if self._is_good_price(ami, state, offer[OFFER_FIELD_IDX.UNIT_PRICE])
            else self.propose_greedy(negotiator_id, state)
        )

        # update my current best price to use for limiting concession in other
        # negotiations
        up = offer[OFFER_FIELD_IDX.UNIT_PRICE]
        if self._is_selling(ami):
            self._best_selling = max(up, self._best_selling)
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = self._best_selling
        else:
            self._best_buying = min(up, self._best_buying)
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = self._best_buying
        return response          

# This is in order to run tournaments
if not TO_SUBMISSION:
    from agents_pool import * #SimpleAgent, BetterAgent, LearningAgent, AdaptiveAgent


    def run(
        competition="oneshot",
        reveal_names=True,
        n_steps=101,#150,
        n_configs=1#,
        #controlled_env=True
    ):
        """
        **Not needed for submission.** You can use this function to test your agent.

        Args:
            competition: The competition type to run (possibilities are oneshot, std,
                        collusion).
            n_steps:     The number of simulation steps.
            n_configs:   Number of different world configurations to try.
                        Different world configurations will correspond to
                        different number of factories, profiles
                        , production graphs etc

        Returns:
            None

        Remarks:

            - This function will take several minutes to run.
            - To speed it up, use a smaller `n_step` value

        """
        if competition == "oneshot":
            competitors = [
                #HunterAgent, 
                QlAgent,
                #QlAgent,
                #GreedyOneShotAgent,
                #RandomOneShotAgent, 
                #SyncRandomOneShotAgent,
                #SimpleAgent, 
                # BetterAgent,
                #AdaptiveAgent,
                LearningAgent
            ]
        else:
            from scml.scml2020.agents import DecentralizingAgent, BuyCheapSellExpensiveAgent

            competitors = [
                #HunterAgent,
                DecentralizingAgent,
                BuyCheapSellExpensiveAgent
            ]

        start = time.perf_counter()

        # if controlled_env:
        #     if competition == "oneshot":
        #         world = SCML2020OneShotWorld(
        #             **SCML2020OneShotWorld.generate(
        #                 agent_types = competitors,
        #                 n_agents_per_process=1,
        #                 n_processes=2,
        #                 n_steps=n_steps,
        #                 construct_graphs=True,
        #                 compact=True
        #             )
        #         )

        #         world.draw(what=["contracts-concluded"])
        #         plt.show()
        #         world.run()
        #         plt.show()
        if competition == "std":
            runner = anac2021_std
        elif competition == "collusion":
            runner = anac2021_collusion
        else:
            runner = anac2021_oneshot
        results = runner(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            #parallelism="serial",
            compact=True,
            #random_agent_types=False,
            #max_worlds_per_config=1,
            #n_runs_per_world=1,
            #min_factories_per_level=1
            disable_agent_printing=False
        )
        # just make names shorter
        results.total_scores.agent_type = results.total_scores.agent_type.str.split(
            "."
        ).str[-1]
        # display results
        print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
        print(f"Finished in {humanize_time(time.perf_counter() - start)}")
        # scores_df = results.total_scores
        # max_score = scores_df['score'].max()
        # final_score = scores_df[scores_df['agent_type']=='HunterAgent']['score'].values[0]
        # place = scores_df[scores_df['agent_type']=='HunterAgent']['score'].index[0]
        # values = [args.start_price_type,args.price_delta_down,args.price_delta_up,args.profit_epsilon,
        #         args.acceptance_price_th,args.acceptance_quantity_th,args.cutoff_rate,args.cutoff_precentile,
        #         args.cutoff_stop_amount,final_score-max_score]#place,final_score]
        # with open(r'parametrs_scores.csv','a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(values)
        # print(f"Finished in {humanize_time(time.perf_counter() - start)}")

    #if __name__ == "__main__":
    import sys

    run("oneshot")
    #run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
