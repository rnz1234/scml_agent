#!/usr/bin/env python
"""
**Submitted to ANAC 2021 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here


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


# required for development
from scml.oneshot import OneShotAgent
import itertools
from scml.oneshot.world import SCML2020OneShotWorld
from scml.oneshot.world import is_system_agent


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


import argparse
import csv

# parser = argparse.ArgumentParser()
# parser.add_argument('start_price_type',type=float)
# parser.add_argument('price_delta_down',type=float)
# parser.add_argument('price_delta_up',type=float)
# parser.add_argument('profit_epsilon',type=float)
# parser.add_argument('acceptance_price_th',type=float)
# parser.add_argument('acceptance_quantity_th',type=float)
# parser.add_argument('cutoff_rate',type=float)
# parser.add_argument('cutoff_precentile',type=float)
# parser.add_argument('cutoff_stop_amount',type=float)
# args = parser.parse_args()

"""
Notes:
- get_ami() will return something through which we can see price range
but an offer has specific price
- self.awi.my_suppiers , self.awi.my_consumers to know target competitors

Strategy Notes:
Q learning:

state = {INIT=0, OPP_OVERRIDE, END, ACCEPT, MY_COUNTER, OPP_COUNTER}
actions = {Accept, End, Counter(p,q) for p,q in range}
rewards = {0, p*q for p,q in range}

method for improving : learn how to better allocate quantity per opposite agents

"""

DEBUG = False #True
ENABLE_GRAPH = False #True
TO_SUBMISSION = False #False

if ENABLE_GRAPH:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    from scipy.signal import butter,filtfilt

def DEBUG_PRINT(msg):
    if DEBUG:
        print(msg)

class OFFER_FIELD_IDX:
    QUANTITY = 0
    TIME = 1
    UNIT_PRICE = 2

class START_PRICE_TYPE:
    SIMPLE_AVG = 0
    GREEDY_START = 1

class STATE_TYPE:
    NO_NEGO = -1
    INIT = 0
    #OPP_OVERRIDE = 1 
    END = 2
    ACCEPT = 3  
    MY_COUNTER = 4 
    OPP_COUNTER = 5


# epsilon=0.00005
# TODO : check that on the cases that in respond() and my_needs <= 0, we always go to on_neg_fail - DONE
# TODO : check that on the cases that in propose() and my_needs <= 0, we always go to on_neg_fail / do something equivalent - DONE
# TODO : take these into account in rewards http://www.yasserm.com/scml/scml2020docs/api/scml.oneshot.OneShotProfile.html?highlight=cost#scml.oneshot.OneShotProfile.cost - DONE
class QlAgent(OneShotAgent):
    def init(self, alpha=0.05, prune_quan=1, gamma=0.9, price_res=10, quantity_res=10, epsilon_start=0.00095, epsilon_end=0.00001, epsilon_decay=500, smart_init=True, complex_state=True): #, start_price_type=START_PRICE_TYPE.SIMPLE_AVG, price_delta_down=0.5, price_delta_up=1, profit_epsilon=0.1, acceptance_price_th=0.1, acceptance_quantity_th=0.1, cutoff_rate=0.2, cutoff_precentile=0.2, cutoff_stop_amount=1):
        self.secured = 0 #0.00005
        # self.start_price_type = start_price_type
        # self.price_delta_down = price_delta_down
        # self.price_delta_up = price_delta_up
        # self.profit_epsilon = profit_epsilon
        # self.acceptance_price_th = acceptance_price_th
        # self.acceptance_quantity_th = acceptance_quantity_th
        # self.current_agreed_price = dict()
        # self.next_proposed_price = dict()
        self.started = dict()
        # self.finished = []
        # self.opposites = -1
        # self.cutoff_rate = cutoff_rate
        # self.cutoff_precentile = cutoff_precentile
        # self.cutoff_stop_amount = cutoff_stop_amount
        # self.opposite_price_gap = dict()
        

        # self.secured = 0
        # self.start_price_type = args.start_price_type
        # self.price_delta_down = args.price_delta_down
        # self.price_delta_up = args.price_delta_up
        # self.profit_epsilon = args.profit_epsilon
        # self.acceptance_price_th = args.acceptance_price_th
        # self.acceptance_quantity_th = args.acceptance_quantity_th
        # self.current_agreed_price = dict()
        # self.next_proposed_price = dict()
        # self.started = dict()
        # self.finished = []
        # self.opposites = -1
        # self.cutoff_rate = args.cutoff_rate
        # self.cutoff_precentile = args.cutoff_precentile
        # self.cutoff_stop_amount = args.cutoff_stop_amount
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
        self.prune_quan = prune_quan

        

    def step(self):
        # if self.awi.current_step == 0:
        #     print("-----------------------------------------------")
        DEBUG_PRINT("# STEP : " + str(self.awi.current_step))
        DEBUG_PRINT("# Episodes  : " + str(self.episodes))
        DEBUG_PRINT(" # Needed : " + str(self._needed()))

        
        for partner in self.started.keys():
            DEBUG_PRINT(partner)
        #    print(self.q_table_per_opp[partner])
        # print("s: " + str(self.success) + ", f: " + str(self.failure))
        # #print(self.current_agreed_price)
        #print("profit: " + str(self.profit))
        # print(self.q_table_per_opp)
        
        self.secured = 0
        for partner in self.r_after_episode.keys():
            self.r_after_episode[partner].append(0)
        

        for partner in self.q_steps.keys():
            self.q_steps[partner] = 0

        if self.awi.current_step == self.awi.n_steps-1:
            if ENABLE_GRAPH:
                for partner in self.r_after_episode.keys():
                    #plt.plot(self.r_after_episode[partner])
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
                


    # when we succeed
    def on_negotiation_success(self, contract, mechanism):
        self.success += 1
        self.episodes += 1
        #self.profit += contract.agreement["unit_price"]*contract.agreement["quantity"]
        self.secured += contract.agreement["quantity"]
        
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            cur_profit = (contract.agreement["unit_price"]-float(self.awi.current_exogenous_input_price)/self.awi.current_exogenous_input_quantity-self.awi.profile.cost)*contract.agreement["quantity"]#-self.awi.current_exogenous_input_price#*self.awi.current_exogenous_input_quantity
        else:
            partner = contract.annotation["seller"]
            cur_profit = (float(self.awi.current_exogenous_output_price)/self.awi.current_exogenous_output_quantity-contract.agreement["unit_price"]-self.awi.profile.cost)*contract.agreement["quantity"]#*self.awi.current_exogenous_output_quantity

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

        my_needs = self._needed()
        self.q_steps[partner] += 1
        if partner not in self.num_of_negotiations_per_step:
            self.num_of_negotiations_per_step[partner] = 0
        self.num_of_negotiations_per_step[partner] += 1
        
        # terminal state
        #print(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], contract.agreement["unit_price"]*contract.agreement["quantity"], STATE_TYPE.END)
        
        # print("done good")
        # print(self.state_per_opp[partner])
        #print("update 3")
        if my_needs == 0:
            DEBUG_PRINT("success : no needs")
            # if self.last_a_per_opp[partner] == "acc":
            #     r = 0 # already taken in account
            # else:
            # 10000*
            if cur_profit < 0:
                r = cur_profit
            else:
                r = cur_profit
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.ACCEPT)
        elif my_needs < 0:
            DEBUG_PRINT("success : - needs")
            # if self.last_a_per_opp[partner] == "acc":
            #     r = 0 # already taken in account
            # else:
            r = cur_profit+my_needs*self._too_much_penalty(mechanism)/self.num_of_negotiations_per_step[partner]#*self.awi.current_step
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.ACCEPT)
        else: # my_needs > 0
            DEBUG_PRINT("success : + needs")
            # if self.last_a_per_opp[partner] == "acc":
            #     r = 0 # already taken in account
            # else:
            # 10000*
            r = (cur_profit-my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step[partner])#*self.awi.current_step
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.ACCEPT)
        DEBUG_PRINT("reward: " + str(r))

        #print("reward: ", r)
        
        if partner not in self.r_after_episode:
            self.r_after_episode[partner] = [0]

        if partner not in self.q_after_episode:
            self.q_after_episode[partner] = [0]
        
        self.r_after_episode[partner][-1] = (self.r_after_episode[partner][-1]+r)/self.q_steps[partner]
        #print(self.q_table_per_opp[partner][STATE_TYPE.ACCEPT])
        #self.q_after_episode[partner].append(self.q_table_per_opp[partner][STATE_TYPE.INIT]["acc"])

        self.state_per_opp[partner] = STATE_TYPE.ACCEPT
        self.last_a_per_opp[partner] = None

        
    # when we fail
    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        if self._is_selling(mechanism):
            partner = annotation["buyer"]
        else:
            partner = annotation["seller"]

        if partner not in self.state_per_opp:
            return None

        if self.state_per_opp[partner] == STATE_TYPE.NO_NEGO:
            return 

        if self.last_a_per_opp is None:
            return
        
        if partner not in self.last_a_per_opp:
            return 

        self.episodes += 1
        self.failure += 1   

        

        #DEBUG_PRINT("on_negotiation_failure " + partner)
        DEBUG_PRINT("--------------------------------")
        DEBUG_PRINT("FAILURE <=>" + partner)

        my_needs = self._needed()
        self.q_steps[partner] += 1
        if partner not in self.num_of_negotiations_per_step:
            self.num_of_negotiations_per_step[partner] = 0
        self.num_of_negotiations_per_step[partner] += 1

        # terminal state
        #print(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], contract.agreement["unit_price"]*contract.agreement["quantity"], STATE_TYPE.END)
        #self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, STATE_TYPE.END)
        # print("----U---")
        # print(self.state_per_opp[partner], self.last_a_per_opp[partner])
        # print("done bad")
        # print(self.state_per_opp[partner])
        #print("update 2")
        if my_needs == 0:
            r = 0
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.END)
        elif my_needs < 0:
            DEBUG_PRINT("need -")
            # 0.1*
            r = 0#my_needs*self._too_much_penalty(mechanism)/self.num_of_negotiations_per_step[partner]#*self.awi.current_step
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.END)
        else: # my_needs > 0
            DEBUG_PRINT("needs +")
            # 10000*
            r = -1000*my_needs*self._too_less_penalty(mechanism)/self.num_of_negotiations_per_step[partner]#*self.awi.current_step
            self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], r, STATE_TYPE.END)
        
        self.state_per_opp[partner] = STATE_TYPE.END
        #print("reward: ", r)
        DEBUG_PRINT("reward: " + str(r))

        if partner not in self.profit_per_opp:
            self.profit_per_opp[partner] = [0]
        else:
            self.profit_per_opp[partner].append(0) 

        if partner not in self.r_after_episode:
            self.r_after_episode[partner] = [0]

        self.r_after_episode[partner][-1] = (self.r_after_episode[partner][-1]+r)/self.q_steps[partner]
        
        self.last_a_per_opp[partner] = None


        
    def propose(self, negotiator_id: str, state) -> "Outcome":
        #print("# STEP : " + str(self.awi.current_step))
        #DEBUG_PRINT("propose " + negotiator_id)
        #print("propose " + negotiator_id)
        #print("here1")
        my_needs = self._needed(negotiator_id)
        #DEBUG_PRINT("needs : " + str(my_needs))
        ami = self.get_ami(negotiator_id)
        if not ami:
            print("No AMI !")
            return None

        #print("here2")
        
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            is_seller = True
        else:
            partner = ami.annotation["seller"]
            is_seller = False
        # if partner in self.finished:
        #     return None
        #DEBUG_PRINT("propose " + partner)
        #DEBUG_PRINT("------------------------------")
        #self._init_opposites_if_needed(ami)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
        offer = [-1] * 3
        
        if partner not in self.started.keys():
            # first proposal with this negotiator. this means first proposal in the simulation 
            # against it.
            #DEBUG_PRINT("INIT")
            self.started[partner] = True
            self.q_table_per_opp[partner] = dict()
            #print(self.q_table_per_opp)
            price_gap = unit_price_issue.max_value-unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value-quantity_issue.min_value
            self.action_vec[partner] = []
            self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap, unit_price_issue, quantity_issue, self.action_vec[partner], is_seller)

            #print("here3")
            if my_needs <= 0:
                self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                #print("here4")
                return None

            self.state_per_opp[partner] = STATE_TYPE.INIT

            self.q_steps[partner] = 1

            #print("here5")
            
        else:
            # not first proposal with this negotiator
            if partner not in self.state_per_opp:
                print("issue, " + str(self.state_per_opp.keys()) +", " +str(self.started))
                return None 
            if self.state_per_opp[partner] == STATE_TYPE.END or self.state_per_opp[partner] == STATE_TYPE.ACCEPT or self.state_per_opp[partner] == STATE_TYPE.NO_NEGO:
                #DEBUG_PRINT("INIT")
                if my_needs <= 0:
                    self.state_per_opp[partner] = STATE_TYPE.NO_NEGO
                    return None
                # last state was terminal, thus we now start new q learning step
                self.state_per_opp[partner] = STATE_TYPE.INIT
                # price_gap = unit_price_issue.max_value-unit_price_issue.min_value
                # quantity_gap = quantity_issue.max_value-quantity_issue.min_value
                # self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap)

                self.q_steps[partner] = 1

            # if we get to the "elif", this means this is during negotiation. 
            elif self.state_per_opp[partner] == STATE_TYPE.OPP_COUNTER or isinstance(self.state_per_opp[partner], tuple):
                #DEBUG_PRINT("OPP_COUNTER")
                #print("propose: " + str(self.state_per_opp[partner]))
                # after the respond 
                # update q
                #self._q_learning_update_q(STATE_TYPE.OPP_COUNTER, self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, self.state_per_opp[partner])
                #self.state_per_opp[partner] = STATE_TYPE.MY_COUNTER
                self.q_steps[partner] += 1
            else:
                # TODO : is that a valid state ?
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
                #self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap, unit_price_issue, quantity_issue, self.action_vec[partner])

                self.state_per_opp[partner] = STATE_TYPE.INIT
                self.q_steps[partner] = 1


        #print("here6")
        # print(self.q_table_per_opp)
        # print(self.state_per_opp[partner])
        if self.state_per_opp[partner] == STATE_TYPE.INIT:
            a = self._q_learning_select_action(self.q_table_per_opp[partner], self.state_per_opp[partner], self.action_vec[partner])
            self.last_a_per_opp[partner] = a
        else:
            # coming from OPP_COUNTER state, I already selected my action
            a = self.last_a_per_opp[partner]
            if a == "acc" or a == "end":
                print("should not happen")
                return None

        #print("here7")
        # DEBUG_PRINT("-------------------------")
        # DEBUG_PRINT("PROPOSE -->" + partner)
        # DEBUG_PRINT(ami.annotation["buyer"], ami.annotation["seller"])
        # DEBUG_PRINT("executed action : " + str(a))
        # DEBUG_PRINT("state : " + str(self.state_per_opp[partner]))
        offer[OFFER_FIELD_IDX.UNIT_PRICE] = a[0]
        offer[OFFER_FIELD_IDX.QUANTITY] = a[1]
        offer[OFFER_FIELD_IDX.TIME] = self.awi.current_step

        # print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("# Needed to produce : " + str(self._needed()))
        # print("propose to " + str(negotiator_id) + ":" + str(self.best_offer(negotiator_id)))
        #print(partner)
        #print(self.q_table_per_opp[partner])
        return tuple(offer)

    def respond(self, negotiator_id, state, offer):
        #DEBUG_PRINT("respond " + negotiator_id)
        # print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("# Needed to produce : " + str(self._needed()))
        my_needs = self._needed(negotiator_id)

        
        
        
        ami = self.get_ami(negotiator_id)
        if not ami:
            DEBUG_PRINT("No AMI !")
            return None
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
        else:
            partner = ami.annotation["seller"]

        if partner in self.last_a_per_opp:
            if self.last_a_per_opp[partner] == "acc" or self.last_a_per_opp[partner] == "end":
                #print("should not happen - we should have junped to success or failure methods")
                return None

        
        # if partner in self.finished:
        #     return None
        #DEBUG_PRINT("respond " + partner)
        #DEBUG_PRINT("------------------------------")
        #unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        #quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]

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
        
        if partner not in self.state_per_opp:
            #return None
            self.q_steps[partner] = 0
            self.started[partner] = True
            self.state_per_opp[partner] = STATE_TYPE.INIT
            self.q_table_per_opp[partner] = dict()
            unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
            quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
            price_gap = unit_price_issue.max_value-unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value-quantity_issue.min_value
            self.action_vec[partner] = []
            self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap, unit_price_issue, quantity_issue, self.action_vec[partner], is_seller)

        #print("respond: " + str(self.state_per_opp[partner]))

        if partner in self.state_per_opp:
            if self.state_per_opp[partner] == STATE_TYPE.NO_NEGO:
                return ResponseType.REJECT_OFFER

        # DEBUG_PRINT("-------------------------")
        # DEBUG_PRINT("RESPOND <--" + partner)
        # DEBUG_PRINT("my last offer : " + str(self.last_a_per_opp[partner]))
        # DEBUG_PRINT("new offer : " + str([offer[OFFER_FIELD_IDX.UNIT_PRICE], offer[OFFER_FIELD_IDX.QUANTITY]]))
        # DEBUG_PRINT("last state : " + str(self.state_per_opp[partner]))
        

        if not self.complex_state:
            new_state = STATE_TYPE.OPP_COUNTER
        else:
            unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
            quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
            price_gap = unit_price_issue.max_value-unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value-quantity_issue.min_value
            #print(self.last_a_per_opp)
            if self.last_a_per_opp == {} or self.last_a_per_opp is None or \
                partner not in self.last_a_per_opp or \
                self.last_a_per_opp[partner] is None:
                new_state = self._find_state_mapping(0, 
                                                    0,
                                                    offer[OFFER_FIELD_IDX.UNIT_PRICE],
                                                    offer[OFFER_FIELD_IDX.QUANTITY],
                                                    price_gap, quantity_gap, unit_price_issue, quantity_issue)[2:]
                self.state_per_opp[partner] = STATE_TYPE.INIT
            else:
                new_state = self._find_state_mapping(self.last_a_per_opp[partner][0], 
                                                    self.last_a_per_opp[partner][1],
                                                    offer[OFFER_FIELD_IDX.UNIT_PRICE],
                                                    offer[OFFER_FIELD_IDX.QUANTITY],
                                                    price_gap, quantity_gap, unit_price_issue, quantity_issue)
        
                # update q 
                #print("update 1")
                self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, new_state)

        
        self.q_steps[partner] += 1
            
        
        # advance state
        self.state_per_opp[partner] = new_state

        # choose new action 
        a = self._q_learning_select_action(self.q_table_per_opp[partner], new_state, self.action_vec[partner])

        # DEBUG_PRINT("new state : " + str(self.state_per_opp[partner]))
        # DEBUG_PRINT("next action : " + str(a))
        # in case no more needs - end
        if my_needs <= 0:
            a = "end"

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

        
    
    def _q_learning_update_q(self, state, action, q, reward, new_state):
        # print(state)
        # print(action)
        # print(new_state)
        # print(reward)
        # print("--------")
        q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*max([q[new_state][action] for action in q[new_state].keys()]) - q[state][action])
        
    def _q_learning_select_action(self, q, state, action_vec):
        # DEBUG_PRINT("_q_learning_select_actionq.keys(state)
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                                          math.exp(-1. * (self.awi.current_step+1) / self.epsilon_decay)

        p = np.random.random()
        #print(np.array(action_vec, dtype=object))
        if p < eps_threshold:
            if state != STATE_TYPE.INIT:
                return np.random.choice(np.array(action_vec, dtype=object))
            else: 
                # without "acc" and "end" since not available in INIT
                return np.random.choice(np.array(action_vec, dtype=object)[:-2])
        else:
            #print(q)
            #print([q[state][action] for action in q[state].keys()])
            max_q = max([q[state][action] for action in q[state].keys()])
            for a in q[state].keys():
                if q[state][a] == max_q:
                    return a

    def _q_learning_q_init(self, q_t, price_gap, quantity_gap, unit_price_issue, quantity_issue, action_vec, is_seller=True):
        #DEBUG_PRINT("_q_learning_q_init")
        min_q = quantity_issue.min_value + self.prune_quan*(quantity_issue.max_value-quantity_issue.min_value)/2
        for s in [STATE_TYPE.INIT, STATE_TYPE.OPP_COUNTER, STATE_TYPE.END, STATE_TYPE.ACCEPT]:   # , STATE_TYPE.MY_COUNTER     
            if not self.complex_state:
                q_t[s] = dict()
            else:
                if s != STATE_TYPE.OPP_COUNTER:
                    q_t[s] = dict()
                else:
                    for p_s in p_range:
                        for q_s in q_range:
                            for p_so in p_range:
                                for q_so in q_range:
                                    q_t[(p_s,np.ceil(q_s),p_so,q_so)] = dict()
                            q_t[(p_s,q_s)] = dict()
    
            # print(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res)
            # print(quantity_issue.min_value, quantity_issue.max_value, self.quantity_res)
            p_range = np.linspace(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res)
            q_range = np.linspace(min_q, quantity_issue.max_value, self.quantity_res)
            for p in p_range:
                #print(q_t)
                for q in q_range:
                    #print(q_t)
                    #print((p,q))
                    if self.smart_init:
                        if is_seller:
                            if p == p_range[-1]:#int(len(p_range)/2)]:
                                if not self.complex_state:
                                    q_t[s][(p,np.ceil(q))] = self._initial_q_seller(p)#0
                                else:
                                    if s != STATE_TYPE.OPP_COUNTER:
                                        q_t[s][(p,np.ceil(q))] = self._initial_q_seller(p)#10000
                                    else:
                                        for p_s in p_range:
                                            for q_s in q_range:
                                                for p_so in p_range:
                                                    for q_so in q_range:
                                                        q_t[(p_s,np.ceil(q_s),p_so,q_so)][(p,np.ceil(q))] = self._initial_q_seller(p)#10000
                                                q_t[(p_s,q_s)][(p,np.ceil(q))] = self._initial_q_seller(p)#0   
                            else:
                                if not self.complex_state:
                                    q_t[s][(p,np.ceil(q))] = self._initial_q_seller(p)#0
                                else:
                                    if s != STATE_TYPE.OPP_COUNTER:
                                        q_t[s][(p,np.ceil(q))] = self._initial_q_seller(p)#0
                                    else:
                                        for p_s in p_range:
                                            for q_s in q_range:
                                                for p_so in p_range:
                                                    for q_so in q_range:
                                                        q_t[(p_s,np.ceil(q_s),p_so,q_so)][(p,np.ceil(q))] = self._initial_q_seller(p)#0
                                                q_t[(p_s,q_s)][(p,np.ceil(q))] = self._initial_q_seller(p)#0  
                        else:
                            if p == p_range[0]:#int(len(p_range)/2)]:
                                if not self.complex_state:
                                    q_t[s][(p,np.ceil(q))] = self._initial_q_buyer(p)#10000
                                else:
                                    if s != STATE_TYPE.OPP_COUNTER:
                                        q_t[s][(p,np.ceil(q))] = self._initial_q_buyer(p)#10000
                                    else:
                                        for p_s in p_range:
                                            for q_s in q_range:
                                                for p_so in p_range:
                                                    for q_so in q_range:
                                                        q_t[(p_s,np.ceil(q_s),p_so,q_so)][(p,np.ceil(q))] = self._initial_q_buyer(p)#0#10000
                                                q_t[(p_s,q_s)][(p,np.ceil(q))] = self._initial_q_buyer(p)#0  
                            else:
                                if not self.complex_state:
                                    q_t[s][(p,np.ceil(q))] = self._initial_q_buyer(p)#0
                                else:
                                    if s != STATE_TYPE.OPP_COUNTER:
                                        q_t[s][(p,np.ceil(q))] = self._initial_q_buyer(p)#0
                                    else:
                                        for p_s in p_range:
                                            for q_s in q_range:
                                                for p_so in p_range:
                                                    for q_so in q_range:
                                                        q_t[(p_s,np.ceil(q_s),p_so,q_so)][(p,np.ceil(q))] = self._initial_q_buyer(p)#0
                                                q_t[(p_s,q_s)][(p,np.ceil(q))] = self._initial_q_buyer(p)#0  
                    else:
                        if not self.complex_state:
                            q_t[s][(p,np.ceil(q))] = 0#0
                        else:
                            if s != STATE_TYPE.OPP_COUNTER:
                                q_t[s][(p,np.ceil(q))] = 0#0
                            else:
                                for p_s in p_range:
                                    for q_s in q_range:
                                        for p_so in p_range:
                                            for q_so in q_range:
                                                q_t[(p_s,np.ceil(q_s),p_so,q_so)][(p,np.ceil(q))] = 0
                                        q_t[(p_s,q_s)][(p,np.ceil(q))] = 0  

            
            #if s == STATE_TYPE.OPP_COUNTER:
            if s != STATE_TYPE.INIT:
                if not self.complex_state:
                    q_t[s]["end"] = 0
                    q_t[s]["acc"] = 0
                else:
                    for p_s in p_range:
                        for q_s in q_range:
                            for p_so in p_range:
                                for q_so in q_range:
                                    q_t[(p_s,np.ceil(q_s),p_so,q_so)]["end"] = 0
                                    if is_seller:
                                        q_t[(p_s,np.ceil(q_s),p_so,q_so)]["acc"] = self._initial_q_seller(p_so)#p_so*q_so-self.awi.current_exogenous_input_price
                                    else:
                                        q_t[(p_s,np.ceil(q_s),p_so,q_so)]["acc"] = self._initial_q_buyer(p_so) #self.awi.current_exogenous_input_price-(p_so-p_s)*(q_so-q_s)
                            q_t[(p_s,q_s)]["end"] = 0  
                            q_t[(p_s,q_s)]["acc"] = 0  

        for p in np.linspace(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res):
            for q in np.linspace(min_q, quantity_issue.max_value, self.quantity_res):
                action_vec.append((p,np.ceil(q)))

        action_vec.append("end")
        action_vec.append("acc")

    def _find_state_mapping(self, p, q, po, qo, price_gap, quantity_gap, unit_price_issue, quantity_issue):
        min_q = quantity_issue.min_value + self.prune_quan*(quantity_issue.max_value-quantity_issue.min_value)/2
        p_range = np.linspace(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res)
        q_range = np.linspace(min_q, quantity_issue.max_value, self.quantity_res)
        return (p, q, self._find_nearest(p_range, po), self._find_nearest(q_range, qo))

    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    def _needed(self, negotiator_id=None):
        return self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity - \
               self.secured

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    # storage : buying to much == disposal, delivery : selling too much == shortfall
    def _too_much_penalty(self, ami):
        if self._is_selling(ami):
            return self.awi.profile.shortfall_penalty_mean#, self.awi.profile.shortfall_penalty_dev
        else:
            return self.awi.profile.disposal_cost_mean#, self.awi.profile.disposal_cost_dev
             
    def _too_less_penalty(self, ami):
        if self._is_selling(ami):
            return self.awi.profile.disposal_cost_mean#, self.awi.profile.disposal_cost_dev
        else:
            return self.awi.profile.shortfall_penalty_mean#, self.awi.profile.shortfall_penalty_dev

    def _initial_q_seller(self, p):
        return 0#1000*p

    def _initial_q_buyer(self, p):
        return 0#1000.0/p

    # def __del__(self):
    #     if ENABLE_GRAPH:
    #         for partner in self.q_after_episode.keys():
    #             plt.plot(self.q_after_episode[partner])
            
    #         plt.show()
                
    # def __del__(self):
    #     avg_q = dict()
    #     for q in self.q_table_per_opp:
            

if not TO_SUBMISSION:
    from agents_pool import * #SimpleAgent, BetterAgent, LearningAgent, AdaptiveAgent


    def run(
        competition="oneshot",
        reveal_names=True,
        n_steps=50,
        n_configs=4#,
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
                GreedyOneShotAgent,
                #RandomOneShotAgent, 
                #SyncRandomOneShotAgent,
                #SimpleAgent, 
                # BetterAgent,
                #AdaptiveAgent,
                #LearningAgent
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
