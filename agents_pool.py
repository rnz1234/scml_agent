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

from collections import defaultdict

import random

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2



class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def init(self):
        self.secured = 0

    def step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state, offer):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_ami(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value),
            quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id=None):
        return self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity - \
               self.secured

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product



class BetterAgent(SimpleAgent):
    """A greedy agent based on OneShotAgent with more sane strategy"""

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = super().propose(negotiator_id, state)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(
            self.get_ami(negotiator_id), state
        )
        return tuple(offer)

    def respond(self, negotiator_id, state, offer):
        response = super().respond(negotiator_id, state, offer)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        ami = self.get_ami(negotiator_id)
        return (
            response if
            self._is_good_price(ami, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

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
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, ami):
        """Finds the minimum and maximum prices"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e



class AdaptiveAgent(BetterAgent):
    """Considers best price offers received when making its decisions"""

    def init(self):
        super().init()
        self._best_selling, self._best_buying = 0.0, float("inf")

    def step(self):
        super().step()
        self._best_selling, self._best_buying = 0.0, float("inf")

    def respond(self, negotiator_id, state, offer):
        """Save the best price received"""
        response = super().respond(negotiator_id, state, offer)
        ami = self.get_ami(negotiator_id)
        if self._is_selling(ami):
            self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
        else:
            self._best_buying = min(offer[UNIT_PRICE], self._best_buying)
        return response

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn, mx = super()._price_range(ami)
        if self._is_selling(ami):
            mn = max(mn, self._best_selling)
        else:
            mx = min(mx, self._best_buying)
        return mn, mx


class LearningAgent(AdaptiveAgent):
    def __init__(
        self,
        *args,
        acc_price_slack=float("inf"),
        step_price_slack=0.0,
        opp_price_slack=0.0,
        opp_acc_price_slack=0.2,
        range_slack = 0.03,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack

    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        super().step()
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(up, self._best_opp_acc_selling[partner])
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(up, self._best_opp_acc_buying[partner])

    def respond(self, negotiator_id, state, offer):
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state, offer)
        # update my current best price to use for limiting concession in other
        # negotiations
        ami = self.get_ami(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)
        return response

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            mn = min(mx * (1 - self._range_slack), max(
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
            ))
        else:
            partner = ami.annotation["seller"]
            mx = max(mn * (1 + self._range_slack),  min(
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
            ))
        return mn, mx


# class DeepSimpleAgent(SimpleAgent):
#     """A greedy agent based on OneShotSyncAgent that does something
#     when in the middle of the production chain"""

#     def init(self):
#         self._sales = self._supplies = 0

#     def step(self):
#         self._sales = self._supplies = 0

#     def on_negotiation_success(self, contract, mechanism):
#         if contract.annotation["product"] == self.awi.my_input_product:
#             self._sales += contract.agreement["quantity"]
#         else:
#             self._supplies += contract.agreement["quantity"]

#     def _needed(self, negotiator_id):
#         summary = self.awi.exogenous_contract_summary
#         secured = (
#             self._sales
#             if self._is_selling(self.get_ami(negotiator_id))
#             else self._supplies
#         )
#         demand = min(summary[0][0], summary[-1][0]) / (self.awi.n_competitors + 1)
#         return  demand - secured


class GreedyOneShotAgent(OneShotAgent):
    """
    A greedy agent based on OneShotAgent
    Args:
        concession_exponent: A real number controlling how fast does the agent
                             concede on price.
        acc_price_slack: The allowed slack in price limits compared with best
                         prices I got so far
        step_price_slack: The allowed slack in price limits compared with best
                         prices I got this step
        opp_price_slack: The allowed slack in price limits compared with best
                         prices I got so far from a given opponent in this step
        opp_acc_price_slack: The allowed slack in price limits compared with best
                         prices I got so far from a given opponent so far
        range_slack: Always consider prices above (1-`range_slack`) of the best
                     possible prices *good enough*.
    Remarks:
        - A `concession_exponent` greater than one makes the agent concede
          super linearly and vice versa
    """

    def __init__(
        self,
        *args,
        concession_exponent=None,
        acc_price_slack=float("inf"),
        step_price_slack=None,
        opp_price_slack=None,
        opp_acc_price_slack=None,
        range_slack=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
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

    def init(self):
        """Initialize the quantities and best prices received so far"""
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))
        self._sales = self._supplies = 0

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._sales = self._supplies = 0

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)

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

    def propose(self, negotiator_id: str, state):
        # find the absolute best offer for me. This will most likely has an
        # unrealistic price
        offer = self.best_offer(negotiator_id)

        # if there are no best offers, just return None to end the negotiation
        if not offer:
            return None

        # over-write the unit price in the best offer with a good-enough price
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(self.get_ami(negotiator_id), state)
        return tuple(offer)

    def respond(self, negotiator_id, state, offer):
        # find the quantity I still need and end negotiation if I need nothing more
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION

        # reject any offers with quantities above my needs
        response = (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )
        if response != ResponseType.ACCEPT_OFFER:
            return response

        # reject offers with prices that are deemed NOT good-enough
        ami = self.get_ami(negotiator_id)
        response = (
            response
            if self._is_good_price(ami, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

        # update my current best price to use for limiting concession in other
        # negotiations
        up = offer[UNIT_PRICE]
        if self._is_selling(ami):
            self._best_selling = max(up, self._best_selling)
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = self._best_selling
        else:
            self._best_buying = min(up, self._best_buying)
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = self._best_buying
        return response

    def best_offer(self, negotiator_id):
        my_needs = int(self._needed(negotiator_id))
        if my_needs <= 0:
            return None
        ami = self.get_ami(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        mx = max(min(my_needs, quantity_issue.max_value), quantity_issue.min_value)
        offer[QUANTITY] = random.randint(
            max(1, int(0.5 + mx * self.awi.current_step / self.awi.n_steps)), mx
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id):
        ami = self.get_ami(negotiator_id)
        if not ami:
            return 0
        summary = self.awi.exogenous_contract_summary
        secured = self._sales if self._is_selling(ami) else self._supplies
        demand = min(summary[0][0], summary[-1][0]) / (self.awi.n_competitors + 1)
        return demand - secured

    def _is_selling(self, ami):
        if not ami:
            return None
        return ami.annotation["product"] == self.awi.my_output_product

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
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
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