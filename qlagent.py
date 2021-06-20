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

DEBUG = False#True

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
    INIT = 0
    #OPP_OVERRIDE = 1 
    END = 2
    ACCEPT = 3  
    MY_COUNTER = 4 
    OPP_COUNTER = 5

class QlAgent(OneShotAgent):
    def init(self, alpha=0.1, gamma=0.9, price_res=10, quantity_res=10): #, start_price_type=START_PRICE_TYPE.SIMPLE_AVG, price_delta_down=0.5, price_delta_up=1, profit_epsilon=0.1, acceptance_price_th=0.1, acceptance_quantity_th=0.1, cutoff_rate=0.2, cutoff_precentile=0.2, cutoff_stop_amount=1):
        self.secured = 0
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
        

    def step(self):
        # if self.awi.current_step == 0:
        #     print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("s: " + str(self.success) + ", f: " + str(self.failure))
        # #print(self.current_agreed_price)
        # print("profit: " + str(self.profit))
        # print(self.q_table_per_opp)
        
        self.secured = 0
        
        

    # when we succeed
    def on_negotiation_success(self, contract, mechanism):
        self.success += 1
        self.profit += contract.agreement["unit_price"]*contract.agreement["quantity"]
        self.secured += contract.agreement["quantity"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
        else:
            partner = contract.annotation["seller"]

        DEBUG_PRINT("on_negotiation_success " + partner)
        
        # terminal state
        #print(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], contract.agreement["unit_price"]*contract.agreement["quantity"], STATE_TYPE.END)
        self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], contract.agreement["unit_price"]*contract.agreement["quantity"], STATE_TYPE.END)
        self.state_per_opp[partner] = STATE_TYPE.ACCEPT

        
    # when we fail
    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        self.failure += 1
        my_needs = self._needed()
        if my_needs <= 0:
            DEBUG_PRINT("No more needs !")
            return
            
            
        if self._is_selling(mechanism):
            partner = annotation["buyer"]
        else:
            partner = annotation["seller"]

        DEBUG_PRINT("on_negotiation_failure " + partner)

        # terminal state
        #print(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], contract.agreement["unit_price"]*contract.agreement["quantity"], STATE_TYPE.END)
        #self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, STATE_TYPE.END)
        self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], -10, STATE_TYPE.END)
        self.state_per_opp[partner] = STATE_TYPE.END

        
    def propose(self, negotiator_id: str, state) -> "Outcome":
        DEBUG_PRINT("propose " + negotiator_id)
        #print("propose " + negotiator_id)
        my_needs = self._needed(negotiator_id)
        DEBUG_PRINT("needs : " + str(my_needs))
        if my_needs <= 0:
            DEBUG_PRINT("No more needs !")
            return None
        ami = self.get_ami(negotiator_id)
        if not ami:
            DEBUG_PRINT("No AMI !")
            return None
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
        else:
            partner = ami.annotation["seller"]
        # if partner in self.finished:
        #     return None
        DEBUG_PRINT("propose " + partner)
        DEBUG_PRINT("------------------------------")
        #self._init_opposites_if_needed(ami)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        quantity_issue = ami.issues[OFFER_FIELD_IDX.QUANTITY]
        offer = [-1] * 3
        
        if partner not in self.started.keys():
            # first proposal with this negotiator. this means first proposal in the simulation 
            # against it.
            DEBUG_PRINT("INIT")
            self.started[partner] = True
            self.q_table_per_opp[partner] = dict()
            #print(self.q_table_per_opp)
            price_gap = unit_price_issue.max_value-unit_price_issue.min_value
            quantity_gap = quantity_issue.max_value-quantity_issue.min_value
            self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap, unit_price_issue, quantity_issue)

            self.state_per_opp[partner] = STATE_TYPE.INIT
            
        else:
            # not first proposal with this negotiator
            if self.state_per_opp[partner] == STATE_TYPE.END or self.state_per_opp[partner] == STATE_TYPE.ACCEPT:
                DEBUG_PRINT("INIT")
                # last state was terminal, thus we now start new q learning step
                self.state_per_opp[partner] = STATE_TYPE.INIT
                # price_gap = unit_price_issue.max_value-unit_price_issue.min_value
                # quantity_gap = quantity_issue.max_value-quantity_issue.min_value
                # self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap)

            # if we get to the "elif", this means this is during negotiation. This can only happen 
            # after respond which set the state as MY_COUNTER
            elif self.state_per_opp[partner] == STATE_TYPE.MY_COUNTER:
                DEBUG_PRINT("MY_COUNTER")
                # after the respond 
                # update q
                # update q 
                self._q_learning_update_q(STATE_TYPE.OPP_COUNTER, self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, self.state_per_opp[partner])
            else:
                # TODO : is that a valid state ?
                # print(self.state_per_opp[partner])
                # print("error - invalid state")
                # sys.exit(1)
                DEBUG_PRINT("INIT2")
                self.started[partner] = True
                self.q_table_per_opp[partner] = dict()
                #print(self.q_table_per_opp)
                price_gap = unit_price_issue.max_value-unit_price_issue.min_value
                quantity_gap = quantity_issue.max_value-quantity_issue.min_value
                self._q_learning_q_init(self.q_table_per_opp[partner], price_gap, quantity_gap, unit_price_issue, quantity_issue)

                self.state_per_opp[partner] = STATE_TYPE.INIT


        # print(self.q_table_per_opp)
        # print(self.state_per_opp[partner])
        a = self._q_learning_select_action(self.q_table_per_opp[partner], self.state_per_opp[partner])
        self.last_a_per_opp[partner] = a

        offer[OFFER_FIELD_IDX.UNIT_PRICE] = a[0]
        offer[OFFER_FIELD_IDX.QUANTITY] = a[1]
        offer[OFFER_FIELD_IDX.TIME] = self.awi.current_step

        # print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("# Needed to produce : " + str(self._needed()))
        # print("propose to " + str(negotiator_id) + ":" + str(self.best_offer(negotiator_id)))
        return tuple(offer)

    def respond(self, negotiator_id, state, offer):
        DEBUG_PRINT("respond " + negotiator_id)
        # print("-----------------------------------------------")
        # print("# STEP : " + str(self.awi.current_step))
        # print("# Needed to produce : " + str(self._needed()))
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            DEBUG_PRINT("No more needs !")
            #print("response to " + str(negotiator_id) + ": END NEGO.")
            return ResponseType.END_NEGOTIATION
        ami = self.get_ami(negotiator_id)
        if not ami:
            DEBUG_PRINT("No AMI !")
            return None
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
        else:
            partner = ami.annotation["seller"]
        # if partner in self.finished:
        #     return None
        DEBUG_PRINT("respond " + partner)
        DEBUG_PRINT("------------------------------")
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        
        if partner not in self.started.keys():
            # first proposal with this negotiator
            self.started[partner] = True
        
        new_state = STATE_TYPE.OPP_COUNTER

        # update q 
        self._q_learning_update_q(self.state_per_opp[partner], self.last_a_per_opp[partner], self.q_table_per_opp[partner], 0, new_state)

        # choose new action 
        a = self._q_learning_select_action(self.q_table_per_opp[partner], new_state)

        self.last_a_per_opp[partner] = a
        if isinstance(a, str):
            # state will be updated in on_failure and in on_success
            if a == "end":
                return ResponseType.END_NEGOTIATION
            elif a == "acc":
                return ResponseType.ACCEPT_OFFER
            
        else:
            self.state_per_opp[partner] = STATE_TYPE.MY_COUNTER
            # this means we're going to propose()
            return ResponseType.REJECT_OFFER

        
    
    def _q_learning_update_q(self, state, action, q, reward, new_state):
        q[state][action] = q[state][action] + self.alpha*(reward + self.gamma*max([q[new_state][action] for action in q[new_state].keys()]) - q[state][action])

    def _q_learning_select_action(self, q, state):
        # DEBUG_PRINT("_q_learning_select_action")
        # DEBUG_PRINT(q.keys())
        # DEBUG_PRINT(state)
        max_q = max([q[state][action] for action in q[state].keys()])
        for a in q[state].keys():
            if q[state][a] == max_q:
                return a

    def _q_learning_q_init(self, q_t, price_gap, quantity_gap, unit_price_issue, quantity_issue):
        #DEBUG_PRINT("_q_learning_q_init")
        for s in [STATE_TYPE.INIT, STATE_TYPE.MY_COUNTER, STATE_TYPE.OPP_COUNTER, STATE_TYPE.END, STATE_TYPE.ACCEPT]:        
            q_t[s] = dict()
            # print(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res)
            # print(quantity_issue.min_value, quantity_issue.max_value, self.quantity_res)
            for p in np.linspace(unit_price_issue.min_value, unit_price_issue.max_value, self.price_res):
                for q in np.linspace(quantity_issue.min_value, quantity_issue.max_value, self.quantity_res):
                    #print((p,q))
                    q_t[s][(p,q)] = 0
            
            if s == STATE_TYPE.OPP_COUNTER:
                q_t[s]["end"] = 0
                q_t[s]["acc"] = 0

    
    def _needed(self, negotiator_id=None):
        return self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity - \
               self.secured

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    # def _init_opposites_if_needed(self, ami):
    #     if self.opposites == -1:
    #         if self._is_selling(ami):
    #             self.opposites = len(self.awi.my_consumers)
    #         else:
    #             self.opposites = len(self.awi.my_suppliers)

    # def _in_range(self, price, quantity, pivot_price, pivot_quanity):
    #     if abs(price-pivot_price) <= self.acceptance_price_th and \
    #         abs(quantity-pivot_quanity) <= self.acceptance_quantity_th:
    #         return True
    #     else:
    #         return False


    # def _update_opposites(self):
    #     pass


from agents_pool import * #SimpleAgent, BetterAgent, LearningAgent, AdaptiveAgent


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=50,
    n_configs=2,
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
            GreedyOneShotAgent
            #RandomOneShotAgent, 
            #SyncRandomOneShotAgent,
            #SimpleAgent, 
            # BetterAgent,
            # AdaptiveAgent,
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
        #parallelism="serial"
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
