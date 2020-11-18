import math
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from gym_species_management.envs.shared_env import *

class AbstractSpeciesManagementEnv(gym.Env) {
    metadata = {'render.modes' : ['human']}

    def __init__(self, 
        params = {
            Beta = 0.1 # Cost Discount Coefficient
        }, 
        init_state = (invasive_population, native_population, budget), #Should be an instance of the same type/compatible type with observation space
        TMax = 9, 
        CR_min = 100,
        invasive_growth = np.array([1.5,1.5]),
        native_growth = np.array([1.5,1.5]), 
        annual_budget = 1000,
        eradicate_isolate_cost = 10, 
        eradicate_meadow_cost = 20, 
        restore_isolate_cost = 15, 
        invasive_damage_cost = 10,
        file = None) :

        ## Parameters
        self.Beta = Beta 

        self.eradicate_isolate_cost = eradicate_isolate_cost
        self.eradicate_meadow_cost = eradicate_meadow_cost
        self.restore_isolate_cost = restore_isolate_cost
        self.invasive_damage_cost = invasive_damage_cost

        ## Preserve these for reset
        self.init_state = init_state
        self.years_passed = 0
        self.TMax = TMax
        self.total_penalty = 0

        self.total_budget = self.get_budget()
        self.state = self.init_state

        ## Normalize cts actions / observations
        #TODO CRUCIAL <- look into normalizing these spaces.
        #Look into discrete spaces (multidiscrete)
        self.action_space = spaces.Dict(spaces = {
            eradicate_isolate : spaces.Box(low = 0, high=self.get_invasive_population()[0], dtype = np.int), 
            eradicate_meadow : spaces.Box(low = 0, high=self.get_invasive_population()[1], dtype = np.int),
            restore_isolate : spaces.Box(low=0, high=np.iinfo(np.int).max, dtype = np.int) #Realisitcally bound this
        })

        self.observation_space = spaces.Dict({
            invasive_population : spaces.Box(low=0, high=np.iinfo(np.int).max, dtype=np.int, shape(2,)),
            native_population:  spaces.Box(low=0, high=np.iinfo(np.int).max, dtype=np.int, shape(2,)),
            budget: spaces.Box(low=0, high=self.get_budget() + (self.TMax * self.annual_budget), dtype=np.float32)
        })

        return

    def step(self, action):
        #Validate action against constraints
        assert action in self.action_space
        action = self.get_action(action)

        ## Update State
        #Update Invasive Population
        new_invasive_population = [invasive_growth[0] * (self.state[0][0] - action[eradicate_isolate]), invasive_growth[1] * (self.state[0][1] - action[eradicate_meadow])]
        #Update Native Population
        new_native_population = [native_growth[0] * (self.state[1][0] + action[restore_isolate]), native_growth[1] * self.state[1][1]]
        
        #Update Budget
        fiscal_cost = self.eradicate_isolate_cost*action[eradicate_isolate] + self.eradicate_meadow_cost*action[eradicate_meadow] + self.restore_isolate_cost*action[restore_isolate]
        new_budget = self.total_budget - fiscal_cost
        #Update Penalty (Includes cost of invasive spartina survival)
        new_penalty = self.total_penalty - fiscal_cost - (self.invasive_damage_cost * sum(new_invasive_population))

        # self.state = 

        #Increment Time
        self.years_passed += 1
        #Check if done (TMax)


    ...

    def reset(self):
    ...

    def render(self, mode='human', close=False):
    ...

    def get_action(self, action):
        # Bound each action by state-informed action space.
        #Remaining budget variable? Or proportionality of budget requested by out of bound actions.
        np.clip(action[eradicate_isolate], a_min = 0, a_max = self.state[0][0])
        np.clip(action[eradicate_meadow], a_min = 0, a_max = self.state[0][1])
        np.clip(action[restore_isolate], a_min = 0, a_max = self.budget//self.restore_isolate_cost)

        return action

    def get_invasive_population(self):
        return self.state[0]
    
    def get_native_population(self):
        return self.state[1]

    def get_budget(self):
        return self.state[2]
}