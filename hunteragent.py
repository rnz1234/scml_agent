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

parser = argparse.ArgumentParser()
parser.add_argument('start_price_type',type=float)
parser.add_argument('price_delta_down',type=float)
parser.add_argument('price_delta_up',type=float)
parser.add_argument('profit_epsilon',type=float)
parser.add_argument('acceptance_price_th',type=float)
parser.add_argument('acceptance_quantity_th',type=float)
parser.add_argument('cutoff_rate',type=float)
parser.add_argument('cutoff_precentile',type=float)
parser.add_argument('cutoff_stop_amount',type=float)
args = parser.parse_args()

"""
Notes:
- get_ami() will return something through which we can see price range
but an offer has specific price
- self.awi.my_suppiers , self.awi.my_consumers to know target competitors

Strategy Notes:
- we can start from prices which are related to the get_ami().unit_price.min_value,max_value
for example:
    - average between them
    - smart average that keeps us from lossing (taking in acount)
    - smart average from other kind 
- proposals - according to best acceptance price
- responses - in case no more quantity left, reject; else accept if around (hyperparam) 
best acceptance price else propose accorsing to best acceptance price
- as time passes, try to work more with the most profitable partners; less with others
(from exploration to exploitation, "genetic" style ?)
- can use naive search method to look for best price
(from time to time re-doing it to look for new best price). Later do more sophisticated 
search (learning based) like generalized speedy q-learning




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

class HunterAgent(OneShotAgent):

    # TODO : add greediness_rate, cutoff_rate, cutoff_ratio, exporation_epoch
    # every exporation_epoch steps, the price delta for exploration is mult by greediness_rate   
    # every 1/cut_off rate steps, cutoff_ratio of the opposites which yields the agent the worst acculated income, are cut from propose negotiation 
    # idea : overbooking rate (put more quantity than available)
    def init(self): #, start_price_type=START_PRICE_TYPE.SIMPLE_AVG, price_delta_down=0.5, price_delta_up=1, profit_epsilon=0.1, acceptance_price_th=0.1, acceptance_quantity_th=0.1, cutoff_rate=0.2, cutoff_precentile=0.2, cutoff_stop_amount=1):
        # self.secured = 0
        # self.start_price_type = start_price_type
        # self.price_delta_down = price_delta_down
        # self.price_delta_up = price_delta_up
        # self.profit_epsilon = profit_epsilon
        # self.acceptance_price_th = acceptance_price_th
        # self.acceptance_quantity_th = acceptance_quantity_th
        # self.current_agreed_price = dict()
        # self.next_proposed_price = dict()
        # self.started = dict()
        # self.finished = []
        # self.opposites = -1
        # self.cutoff_rate = cutoff_rate
        # self.cutoff_precentile = cutoff_precentile
        # self.cutoff_stop_amount = cutoff_stop_amount
        # self.opposite_price_gap = dict()
        

        self.secured = 0
        self.start_price_type = args.start_price_type
        self.price_delta_down = args.price_delta_down
        self.price_delta_up = args.price_delta_up
        self.profit_epsilon = args.profit_epsilon
        self.acceptance_price_th = args.acceptance_price_th
        self.acceptance_quantity_th = args.acceptance_quantity_th
        self.current_agreed_price = dict()
        self.next_proposed_price = dict()
        self.started = dict()
        self.finished = []
        self.opposites = -1
        self.cutoff_rate = args.cutoff_rate
        self.cutoff_precentile = args.cutoff_precentile
        self.cutoff_stop_amount = args.cutoff_stop_amount
        self.opposite_price_gap = dict()
        self.success = 0 
        self.failure = 0
        

    def step(self):
        #print("s: " + str(self.success) + ", f: " + str(self.failure))
        self.secured = 0
        self.price_delta_up = float(self.price_delta_up)/2 # may change 2 to be some hyperparameter
        self.price_delta_down = float(self.price_delta_down)/2 # may change 2 to be some hyperparameter

        if self.opposites == self.cutoff_stop_amount:
            return

        # cutoff
        if self.awi.current_step % int(math.ceil(1.0/self.cutoff_rate)) == int(math.ceil(1.0/self.cutoff_rate)):
            precentile_price = np.percentile(np.array(self.current_agreed_price.values()), self.cutoff_precentile)
            for partner in self.current_agreed_price.keys():
                if self.current_agreed_price[partner] < precentile_price:
                    if self.opposites == self.cutoff_stop_amount:
                        return  
                    self.finished.append(partner) 
                    self.opposites -= 1

    # when we succeed
    def on_negotiation_success(self, contract, mechanism):
        self.success += 1
        self.secured += contract.agreement["quantity"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
        else:
            partner = contract.annotation["seller"]

        DEBUG_PRINT("on_negotiation_success " + partner)
        
        # update price
        self.current_agreed_price[partner] = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            self.next_proposed_price[partner] = contract.agreement["unit_price"] + self.price_delta_up*self.opposite_price_gap[partner]
        else: 
            self.next_proposed_price[partner] = contract.agreement["unit_price"] - self.price_delta_down*self.opposite_price_gap[partner]

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

        # update price
        if self._is_selling(mechanism):
            self.next_proposed_price[partner] -= self.price_delta_down*self.opposite_price_gap[partner]
        else:
            self.next_proposed_price[partner] += self.price_delta_up*self.opposite_price_gap[partner]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        DEBUG_PRINT("propose " + negotiator_id)
        my_needs = self._needed(negotiator_id)
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
        if partner in self.finished:
            return None
        DEBUG_PRINT("propose " + partner)
        DEBUG_PRINT("------------------------------")
        self._init_opposites_if_needed(ami)
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        offer = [-1] * 3
        if partner not in self.started.keys():
            # first proposal with this negotiator
            self.started[partner] = True
            if self.start_price_type == START_PRICE_TYPE.SIMPLE_AVG:
                self.opposite_price_gap[partner] = unit_price_issue.max_value-unit_price_issue.min_value
                offer[OFFER_FIELD_IDX.UNIT_PRICE] = float(unit_price_issue.min_value + unit_price_issue.max_value)/2
            elif self.start_price_type == START_PRICE_TYPE.GREEDY_START:
                if self._is_selling(ami):
                    self.opposite_price_gap[partner] = unit_price_issue.max_value-unit_price_issue.min_value
                    offer[OFFER_FIELD_IDX.UNIT_PRICE] = unit_price_issue.max_value
                else:
                    self.opposite_price_gap[partner] = unit_price_issue.max_value-unit_price_issue.min_value
                offer[OFFER_FIELD_IDX.UNIT_PRICE] = unit_price_issue.min_value
            else:
                print("FATAL : unsupported init price method - crash")
                sys.exit(1)
            
        else:
            # not first proposal with this negotiator
            offer[OFFER_FIELD_IDX.UNIT_PRICE] = self.next_proposed_price[partner]

        offer[OFFER_FIELD_IDX.QUANTITY] = float(self._needed())/self.opposites
        offer[OFFER_FIELD_IDX.TIME] = self.awi.current_step

        # init this next_proposed_price for this negotiator_id
        self.next_proposed_price[partner] = offer[OFFER_FIELD_IDX.UNIT_PRICE]

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
        if partner in self.finished:
            return None
        DEBUG_PRINT("respond " + partner)
        DEBUG_PRINT("------------------------------")
        unit_price_issue = ami.issues[OFFER_FIELD_IDX.UNIT_PRICE]
        if self.start_price_type == START_PRICE_TYPE.SIMPLE_AVG:
            start_price = float(unit_price_issue.min_value + unit_price_issue.max_value)/2
        else:
            print("FATAL : unsupported init price method - crash")
            sys.exit(1)
        #print("response to " + str(negotiator_id) + ": " + str(ResponseType.ACCEPT_OFFER
        #    if offer[QUANTITY] <= my_needs
        #    else ResponseType.REJECT_OFFER))

        # TODO : add check of the price. In case we get a better price then reset the learning
        if partner not in self.current_agreed_price:
            cur_price = start_price
        else:
            cur_price = self.current_agreed_price[partner]
        
        if self._is_selling(ami):
            # if offer[OFFER_FIELD_IDX.UNIT_PRICE] < unit_price_issue.min_value:
            #     return ResponseType.REJECT_OFFER
            # else:
            if offer[OFFER_FIELD_IDX.UNIT_PRICE] > cur_price - self.acceptance_price_th and \
                offer[OFFER_FIELD_IDX.QUANTITY] <= float(self._needed()/self.opposites) and \
                offer[OFFER_FIELD_IDX.QUANTITY] >= (float(self._needed())/self.opposites)*(1-self.acceptance_quantity_th):
                    # TODO : check if we need to set the below (not needed in case on_negotiation_success() is called)
                    self.opposite_price_gap[partner] = unit_price_issue.max_value-unit_price_issue.min_value
                    self.current_agreed_price[partner] = offer[OFFER_FIELD_IDX.UNIT_PRICE]
                    #self.next_proposed_price[negotiator_id] = offer[OFFER_FIELD_IDX.UNIT_PRICE] + self.price_delta
                    return ResponseType.ACCEPT_OFFER
            else:
                if partner not in self.next_proposed_price:
                    self.next_proposed_price[partner] = start_price

                return ResponseType.REJECT_OFFER
        else:
            # if offer[OFFER_FIELD_IDX.UNIT_PRICE] > unit_price_issue.max_value:
            #     return ResponseType.REJECT_OFFER
            if offer[OFFER_FIELD_IDX.UNIT_PRICE] < cur_price + self.acceptance_price_th and \
                offer[OFFER_FIELD_IDX.QUANTITY] <= float(self._needed()/self.opposites) and \
                offer[OFFER_FIELD_IDX.QUANTITY] >= (float(self._needed())/self.opposites)*(1-self.acceptance_quantity_th):
                    # TODO : check if we need to set the below (not needed in case on_negotiation_success() is called)
                    self.opposite_price_gap[partner] = unit_price_issue.max_value-unit_price_issue.min_value
                    self.current_agreed_price[partner] = offer[OFFER_FIELD_IDX.UNIT_PRICE]
                    #self.next_proposed_price[negotiator_id] = offer[OFFER_FIELD_IDX.UNIT_PRICE] - self.price_delta
                    return ResponseType.ACCEPT_OFFER
            else:
                if partner not in self.next_proposed_price:
                    self.next_proposed_price[partner] = start_price
                return ResponseType.REJECT_OFFER
            

    def _needed(self, negotiator_id=None):
        return self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity - \
               self.secured

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def _init_opposites_if_needed(self, ami):
        if self.opposites == -1:
            if self._is_selling(ami):
                self.opposites = len(self.awi.my_consumers)
            else:
                self.opposites = len(self.awi.my_suppliers)

    # def _in_range(self, price, quantity, pivot_price, pivot_quanity):
    #     if abs(price-pivot_price) <= self.acceptance_price_th and \
    #         abs(quantity-pivot_quanity) <= self.acceptance_quantity_th:
    #         return True
    #     else:
    #         return False


    # def _update_opposites(self):
    #     pass



from agents_pool import SimpleAgent, BetterAgent, LearningAgent, AdaptiveAgent


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=10,
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
            HunterAgent, 
            #RandomOneShotAgent, 
            #SyncRandomOneShotAgent,
            # SimpleAgent, 
            # BetterAgent,
            # AdaptiveAgent,
            LearningAgent
        ]
    else:
        from scml.scml2020.agents import DecentralizingAgent, BuyCheapSellExpensiveAgent

        competitors = [
            HunterAgent,
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
    scores_df = results.total_scores
    max_score = scores_df['score'].max()
    final_score = scores_df[scores_df['agent_type']=='HunterAgent']['score'].values[0]
    place = scores_df[scores_df['agent_type']=='HunterAgent']['score'].index[0]
    values = [args.start_price_type,args.price_delta_down,args.price_delta_up,args.profit_epsilon,
            args.acceptance_price_th,args.acceptance_quantity_th,args.cutoff_rate,args.cutoff_precentile,
            args.cutoff_stop_amount,final_score-max_score]#place,final_score]
    with open(r'parametrs_scores.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(values)
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")

#if __name__ == "__main__":
import sys

run("oneshot")
#run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
