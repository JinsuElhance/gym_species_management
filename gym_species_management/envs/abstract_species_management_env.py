import math
import numpy as np

import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding

from stable_baselines3.common.env_checker import check_env
class AbstractSpeciesManagementEnv(gym.Env) :
    metadata = {'render.modes' : ['human']}
    
    def __init__(self, 
        beta = 1, 
        init_state = {
            "invasive": np.array([.2, 0.1], dtype=np.float32), 
            "native":  np.array([.3, .1], dtype=np.float32), 
            "budget":  np.array([100.0], dtype=np.float32)},
        total_area = 500, 
        TMax = 10, 
        CR_min = 0.08,
        invasive_growth = np.array([[1.115,0.00117], [0.256, 1.107]], dtype=np.float32),
        native_growth = np.array([[1.038,0.0177], [0.085, 1.036]], dtype=np.float32),
        annual_budget = None,
        eradicate_isolate_cost = 3.3, 
        eradicate_meadow_cost = 3.3, 
        restore_isolate_cost = 18.0, 
        invasive_damage_cost = 1.0,
        file = None) :

        #Set Initial State
        # Struggling to get this to read in, how do we keep a sample of the observation space as our state?
        self.state = init_state
        self.init_state = init_state

        ## Parameters
        self.beta = beta 

        #Constraint Parameters
        self.total_area = total_area
        self.CR_min_prop = CR_min
        self.years_passed = 0
        self.TMax = TMax

        #Growth Parameters
        self.invasive_growth = invasive_growth
        self.native_growth = native_growth

        #Cost Parameters
        self.eradicate_isolate_cost = eradicate_isolate_cost
        self.eradicate_meadow_cost = eradicate_meadow_cost
        self.restore_isolate_cost = restore_isolate_cost
        self.invasive_damage_cost = invasive_damage_cost

        #Cost Parameters
        if annual_budget == None :
            self.annual_budget = 0.2 * self.eradicate_meadow_cost * self.total_area
        else:
            self.annual_budget = annual_budget

        #Use continuous action space, proportion of budget
        self.action_space = spaces.Dict(spaces = {
            "eradicate_isolate" : spaces.Box(low=0, high=1, shape=(1,)),
            "eradicate_meadow" : spaces.Box(low=0, high=1, shape=(1,)),
            "restore_isolate" : spaces.Box(low=0, high=1, shape=(1,))
        })

        #Continuous observation space indicating proportion of land covered by native and invasive Spartina
        self.observation_space = spaces.Dict({
            "invasive" : spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "native" : spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "budget" : spaces.Box(low=0.0, high=self.state["budget"][0] + (self.TMax * self.annual_budget), shape=(1,), dtype=np.float32),
        })

        return

    def step(self, action):
        #Validate action against constraints
        assert action in self.action_space

        #Fetches penalty of action and sets new state
        new_penalty = self.process_action(action)
        
        #Increment Time
        self.years_passed += 1

        #Check terminal constraints.
        insufficient_habitat = bool(self.state["invasive"][1] + self.state["native"][1] < self.CR_min_prop)
        remaining_invasive = bool(self.state["invasive"][0] <= 0 and self.state["invasive"][1] <= 0)
        time_finished = bool(self.years_passed == self.TMax)

        #Check if done
        done = bool(insufficient_habitat or remaining_invasive or time_finished)
        if done:
            print(f"Insufficient Habitat: {insufficient_habitat} -- Remaining Invasive: {remaining_invasive} -- Time Finished: {time_finished}")
        
        if insufficient_habitat or (time_finished and remaining_invasive):
            new_penalty = -1000 #Destroyed Habitat
        
        return self.state, float(new_penalty), done, {}

    def reset(self):
        self.state = self.init_state.copy()
        self.years_passed = 0

        return self.state

    def render(self, mode='human', close=False):
        return None

    def seed(self, seed):
        return [1234567890]

    def close(self):
        super(AbstractSpeciesManagementEnv, self).close()

    def process_action(self, action):
        #Fraction of budget (Current)
        total_action = max(action["eradicate_isolate"] + action["eradicate_meadow"] + action["restore_isolate"], 1)
        
        norm_e_isolates_prop = action["eradicate_isolate"] / total_action * self.state["budget"] / 100 #Fraction of the budget to spend on ~eradication of isolates~
        if self.state["invasive"][0] != 0.0:
            e_isolates = (norm_e_isolates_prop * self.state["budget"] / self.eradicate_isolate_cost) / (self.total_area * self.state["invasive"][0]) * self.state["invasive"][0] # Prop of total area to remove from isolates.
            e_isolates = np.clip(e_isolates, a_min=0, a_max = self.state["invasive"][0]) #Clip by constraints (note: Hides a lot of information from the agent, potentially look into redefining the problem space.)
        else: 
            e_isolates = 0.0
            
        norm_e_meadows_prop = action["eradicate_meadow"] / total_action * self.state["budget"] / 100 #Fraction of the budget to spend on ~eradication of meadows~
        if self.state["invasive"][1] != 0.0:
            e_meadows = (norm_e_meadows_prop * self.state["budget"] / self.eradicate_meadow_cost) / (self.total_area * self.state["invasive"][1]) * self.state["invasive"][1]
            e_meadows = np.clip(e_meadows, a_min=0, a_max = self.state["invasive"][1])
        else:
            e_meadows = 0.0
            
        norm_r_isolates_prop = action["restore_isolate"] / total_action * self.state["budget"] / 100 #Fraction of the budget to spend on ~restoration of isolates~
        if self.state["native"][0] != 0.0:
            r_isolates = (norm_r_isolates_prop * self.state["budget"] / self.restore_isolate_cost) / (self.total_area * self.state["native"][0]) * self.state["native"][0]
            r_isolates = np.clip(r_isolates, a_min=0, a_max = self.state["native"][0])
        else:
            r_isolates = 0.0
            
        fiscal_cost = (self.eradicate_isolate_cost * e_isolates * self.total_area) + (self.eradicate_meadow_cost * e_meadows * self.total_area) + (self.restore_isolate_cost * r_isolates * self.total_area)
        new_budget = self.state["budget"] - fiscal_cost + self.annual_budget

        #Fetch New Invasive Population
        new_invasive = self.get_invasive_population(e_isolates, e_meadows)
        #Fetch New Native Population
        new_native = self.get_native_population(r_isolates)
        
        #Update State
        self.state["invasive"] = new_invasive
        self.state["native"] = new_native
        self.state["budget"] = new_budget
        
        print(self.state)

        #Update Penalty (Includes cost of invasive spartina survival)
        step_penalty = (fiscal_cost - (self.invasive_damage_cost * (sum(self.state["invasive"]) * self.total_area))) * (self.beta**self.years_passed)

        return step_penalty

    def get_invasive_population(self, eradicate_isolates, eradicate_meadows):
        """
        Gets updated invasive population based on units invasive spartina eradicated and growth matrix.
        Input:
        (float) eradicate_isolates : prop of isolates land cover to eradicate
        (float) eradicate_meadows : prop of meadows land cover to eradicate
        Output: 
        (float vector 2x1) new_invasive : vector of new invasive population land cover
        """
        new_invasive = np.matmul(self.invasive_growth, self.state["invasive"] - np.append(eradicate_isolates, eradicate_meadows))
        new_invasive = np.clip(new_invasive, 0.0, 1.0)
        
        return new_invasive
    
    def get_native_population(self, restore_isolates):
        """
        Gets updated native population based on units native spartina restored and growth matrix.
        Input:
        (float) restore_isolates : prop of isolates land cover to restore
        Output: 
        (float vector 2x1) new_native : vector of new native population land cover
        """
        new_native = np.matmul(self.native_growth, self.state["native"] + np.append(restore_isolates, [0.0]))
        new_native = np.clip(new_native, 0.0, 1.0)
        
        return new_native

    def get_bare_area_prop(self):
        return self.total_area - self.state["invasive_isolates"] - self.state["invasive_meadows"] - self.state["native_isolates"] - self.state["native_meadows"]