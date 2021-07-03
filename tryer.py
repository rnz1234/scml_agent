from collections import defaultdict
import random
from negmas import ResponseType
from scml.oneshot import *
from scml.scml2020 import is_system_agent
from pprint import pprint
from scml.oneshot.world import SCML2020OneShotWorld
from scml.oneshot.world import is_system_agent
from scml.oneshot import RandomOneShotAgent, SingleAgreementAspirationAgent
from qlagent_extended_state import QlAgent
#from qlagent import QlAgent
from agents_pool import *
from hunteragent import HunterAgent

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# RandomOneShotAgent
# SimpleAgent
# HunterAgent
# GreedyOneShotAgent
# LearningAgent
# AdaptiveAgent

def try_agent(agent_type, is_seller=True, n_processes=2):
    """Runs an agent in a world simulation against a randomly behaving agent"""
    if is_seller:
        return try_agents([agent_type, QlAgent], n_processes)
    else:
        return try_agents([QlAgent, agent_type], n_processes) 

def try_agents(agent_types, 
                n_processes=2, 
                n_trials=1, 
                draw=True, 
                n_steps=10000, 
                compact=True,
                n_agents_per_process=1,
                agent_params=None):
    """
    Runs a simulation with the given agent_types, and n_processes n_trial times.
    Optionally also draws a graph showing what happened
    """
    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()
    for _ in range(n_trials):
        p = n_processes if isinstance(n_processes, int) else random.randint(*n_processes)
        world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(agent_types, 
                                        agent_params=agent_params, 
                                        n_steps=n_steps,
                                        n_processes=p, 
                                        random_agent_types=False, 
                                        compact=compact, 
                                        n_agents_per_process=n_agents_per_process),
                                        construct_graphs=True,
                                        disable_agent_printing=False
        )
        world.run()

        all_scores = world.scores()
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            key = aid if n_trials == 1 else f"{aid}@{world.id[:4]}"
            agent_scores[key] = (
                 agent.type_name.split(':')[-1].split('.')[-1],
                 all_scores[aid],
                 '(bankrupt)' if world.is_bankrupt[aid] else ''
                )
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            type_ = agent.type_name.split(':')[-1].split('.')[-1]
            type_scores[type_] += all_scores[aid]
            counts[type_] += 1
    type_scores = {k: v/counts[k] if counts[k] else v for k, v in type_scores.items()}
    if draw:
        world.draw(
            what=["contracts-concluded"],
            steps=(0, world.n_steps),
            together=True, ncols=1, figsize=(20, 20)
        )
        plt.show()

    return world, agent_scores, type_scores

def analyze_contracts(world):
    """
    Analyzes the contracts signed in the given world
    """
    import pandas as pd
    data = pd.DataFrame.from_records(world.saved_contracts)
    return data.groupby(["seller_name", "buyer_name"])[["quantity", "unit_price"]].mean()


def print_agent_scores(agent_scores):
    """
    Prints scores of individiual agent instances
    """
    for aid, (type_, score, bankrupt) in agent_scores.items():
        print(f"Agent {aid} of type {type_} has a final score of {score} {bankrupt}")

def print_type_scores(type_scores):
    """Prints scores of agent types"""
    pprint(sorted(tuple(type_scores.items()), key=lambda x: -x[1]))



world, ascores, tscores = try_agent(QlAgent, is_seller=True)
#world, ascores, tscores = try_agent(HunterAgent)
#world, ascores, tscores = try_agent(GreedyOneShotAgent)

print_agent_scores(ascores)

# world, ascores, tscores = try_agent(QlAgent, is_seller=False)
# #world, ascores, tscores = try_agent(HunterAgent)
# #world, ascores, tscores = try_agent(GreedyOneShotAgent)

# print_agent_scores(ascores)