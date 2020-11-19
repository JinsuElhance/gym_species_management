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
            beta = 1 # Cost Discount Coefficient
        }, 
        init_state = (invasive_population, native_population, budget), #Should be an instance of the same type/compatible type with observation space
        total_area = 20
        TMax = 9, 
        CR_min = 0.08,
        invasive_growth = np.array([[1.115,0.00117], [0.256, 1.107]]),
        native_growth = np.array([[1.038,0.0177], [0.085, 1.036]]),
        annual_budget = None,
        eradicate_isolate_cost = 3.3, 
        eradicate_meadow_cost = 3.3, 
        restore_isolate_cost = 18.0, 
        invasive_damage_cost = 1.0,
        file = None) :

        #Set Initial State
        self.state = self.init_state

        ## Parameters
        self.beta = beta 
        self.total_area = total_area
        self.CR_min_prop = CR_min

        #Growth Parameters
        self.invasive_growth = invasive_growth
        self.native_growth = native_growth

        #Cost Parameters
        self.eradicate_isolate_cost = eradicate_isolate_cost
        self.eradicate_meadow_cost = eradicate_meadow_cost
        self.restore_isolate_cost = restore_isolate_cost
        self.invasive_damage_cost = invasive_damage_cost

        #Time / Cost Parameters
        self.years_passed = 0
        self.TMax = TMax
        if annual_budget != None :
            self.annual_budget = 0.2 * self.eradicate_meadow_cost * self.total_area
        else:
            self.annual_budget = annual_budget
        self.total_budget = self.get_budget()

        #Look into discrete spaces (multidiscrete)
        #Look into proportion of budget.
        self.action_space = spaces.Dict(spaces = {
            eradicate_isolate : spaces.Discrete(total_area), 
            eradicate_meadow : spaces.Discrete(total_area),
            restore_isolate : spaces.Discrete(total_area)
        })

        self.observation_space = spaces.Dict({
            invasive_isolates : spaces.Discrete(total_area),
            invasive_meadows : spaces.Discrete(total_area),
            native_isolates : spaces.Discrete(total_area),
            native_meadows : spaces.Discrete(total_area),
            budget: spaces.Box(low=0, high=self.get_budget() + (self.TMax * self.annual_budget), dtype=np.float32)
        })

        return

    def step(self, action):
        #Validate action against constraints
        assert action in self.action_space
        action = self.get_action(action)

        ## Update State
        #Update Invasive Population THESE ARE WRONG CAUSE NOW WE'VE SPLIT UP THE OBSERVATION SPACE INTO 5 VARIABLES DO THE MATRIX THANG
        new_invasive_population = [invasive_growth[0] * (self.state[0][0] - action[eradicate_isolate]), invasive_growth[1] * (self.state[0][1] - action[eradicate_meadow])]
        #Update Native Population
        new_native_population = [native_growth[0] * (self.state[1][0] + action[restore_isolate]), native_growth[1] * self.state[1][1]]
        
        #Update Budget
        fiscal_cost = self.eradicate_isolate_cost*action[eradicate_isolate] + self.eradicate_meadow_cost*action[eradicate_meadow] + self.restore_isolate_cost*action[restore_isolate]
        new_budget = self.total_budget - fiscal_cost

        #Update Penalty (Includes cost of invasive spartina survival)
        step_penalty = fiscal_cost - (self.invasive_damage_cost * sum(new_invasive_population)) * self.beta**self.years_passed

        #Update State
        self.state[invasive_isolates] = new_invasive_isolates
        self.state[invasive_meadows] = new_invasive_meadows
        self.state[native_isolates] = new_native_isolates
        self.state[native_meadows] = new_native_meadows
        self.state[budget] = new_budget

        #Increment Time
        self.years_passed += 1

        #Check if done (TMax)
        done = Bool(
            new_invasive_meadows + new_native_meadows < (self.CR_min * self.total_area) ||
            self.years_passed = self.TMax ||
            self.get_invasive_population = 0
        )

        if new_invasive_meadows + new_native_meadows < (self.CR_min * self.total_area) :
            new_penalty = -1000 #Destroyed Habitat

        return np.array(self.state), new_penalty, done, {}

    ...

    def reset(self):
    ...

    def render(self, mode='human', close=False):
    ...

    def get_action(self, action):
        # Bound each action by state-informed action space.
        #Remaining budget variable? Or proportionality of budget requested by out of bound actions.

        #Proportionality (Current)
        total_action_area = action[eradicate_isolate]  + action[eradicate_meadow] + action[restore_isolate]
        action[eradicate_isolate] = round(action[eradicate_isolate] / total_action_area * self.budget)
        action[eradicate_meadow] = round(action[eradicate_meadow] / total_action_area * self.budget)
        action[restore_isolate] = round(action[restore_isolate] / total_action_area * self.budget)

        #Clip each to budget
        # np.clip(action[eradicate_isolate], a_min = 0, a_max = min(self.state[invasive_isolates], self.budget//self.eradicate_isolate_cost))
        # np.clip(action[eradicate_meadow], a_min = 0, a_max = min(self.state[invasive_meadows], self.budget//self.eradicate_meadow_cost))
        # np.clip(action[restore_isolate], a_min = 0, a_max = min(self.budget//self.restore_isolate_cost, self.get_bare_area()))

        return action

    def get_invasive_population(self):
        return self.state[invasive_isolates], self.state[invasive_meadows]
    
    def get_native_population(self):
        return self.state[native_isolates], self.state[native_meadows]

    def get_budget(self):
        return self.state[budget]

    def get_bare_area(self):
        return self.total_area - self.state[invasive_isolates] - self.state[invasive_meadows] - self.state[native_isolates] - self.state[native_meadows]
}