import math
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
# from gym_species_management.envs.shared_env import *

class AbstractSpeciesManagementEnv(gym.Env) :
    metadata = {'render.modes' : ['human']}
    
    def __init__(self, 
        beta, 
        init_state = {"invasive_isolates": 0, "invasive_meadows": 0, "native_isolates": 0, "native_meadows": 0, "budget": 0}, 
        total_area = 20, 
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
        self.init_state = self.init_state

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

        #Use continuous action space, proportion of either budget or total_area
        self.action_space = spaces.Dict(spaces = {
            "eradicate_isolates" : spaces.Box(low=0, high=1),
            "eradicate_meadows" : spaces.Box(low=0, high=1),
            "restore_isolates" : spaces.Box(low=0, high=1)
        })

        self.observation_space = spaces.Dict({
            "invasive_isolates" : spaces.Discrete(total_area),
            "invasive_meadows" : spaces.Discrete(total_area),
            "native_isolates" : spaces.Discrete(total_area),
            "native_meadows" : spaces.Discrete(total_area),
            "budget": spaces.Box(low=0, high=self.state["budget"] + (self.TMax * self.annual_budget), dtype=np.float32)
        })

        return

    def step(self, action):
        #Validate action against constraints
        assert action in self.action_space
        # action = self.get_action(action)

        e_isolates_units, e_meadows_units, r_isolates_units = self.fraction_into_area(action)

        ## Update State
        #Update Invasive Population 
        new_invasive_isolates, new_invasive_meadows = self.get_invasive_population(e_isolates_units, e_meadows_units)
        #Update Native Population
        new_native_isolates, new_native_meadows = self.get_native_population(r_isolates_units)
        
        #Update Budget
        fiscal_cost = self.eradicate_isolate_cost*e_isolates_units + self.eradicate_meadow_cost*e_meadows_units + self.restore_isolate_cost*r_isolates_units
        new_budget = self.state["budget"] - fiscal_cost + self.annual_budget

        #Update Penalty (Includes cost of invasive spartina survival)
        step_penalty = fiscal_cost - (self.invasive_damage_cost * (new_invasive_isolates + new_invasive_meadows)) * self.beta**self.years_passed

        #Update State
        self.state["invasive_isolates"] = new_invasive_isolates
        self.state["invasive_meadows"] = new_invasive_meadows
        self.state["native_isolates"] = new_native_isolates
        self.state["native_meadows"] = new_native_meadows
        self.state["budget"] = new_budget

        #Increment Time
        self.years_passed += 1

        insufficient_habitat = bool(new_invasive_meadows + new_native_meadows < (self.CR_min * self.total_area))
        remaining_insasive = bool(new_invasive_isolates + new_invasive_meadows == 0)
        time_finished = bool(self.years_passed == self.TMax)

        #Check if done (TMax)
        done = bool(insufficient_habitat or remaining_insasive or time_finished)

        if insufficient_habitat or (time_finished and remaining_insasive):
            new_penalty = -1000 #Destroyed Habitat

        return np.array(self.state), new_penalty, done, {}

    def reset(self):
        self.state = self.init_state
        self.years_passed = 0

        return self.state

    def render(self, mode='human', close=False):
        return None

    def seed(self, seed):
        return [1234567890]

    def close(self):
        super(AbstractSpeciesManagementEnv, self).close() 

    def fraction_into_area(self, action) :
        # total_action_area = action[eradicate_isolate]  + action[eradicate_meadow] + action[restore_isolate]
        #Fraction of budget
        #  = round(action[eradicate_isolate] / total_action_area * self.budget)
        #  = round(action[eradicate_meadow] / total_action_area * self.budget)
        #  = round(action[restore_isolate] / total_action_area * self.budget)

        #Fraction of total area (Current)
        requested_proportion = action["eradicate_isolates"]  + action["eradicate_meadows"] + action["restore_isolates"]

        #Convert each action proportion into area units and then clip by constraints.
        norm_e_isolates_prop = action["eradicate_isolates"] / requested_proportion
        e_isolates_units = round(action["eradicate_isolates"] / (requested_proportion * self.total_area))
        m_e_isolates = min(self.state["invasive_isolates"], self.budget//self.eradicate_isolate_cost)
        np.clip(e_isolates_units, a_min=0, a_max = m_e_isolates)

        norm_e_meadows_prop = action["eradicate_meadows"] / requested_proportion
        e_meadows_units = round(action["eradicate_meadows"] / (requested_proportion * self.total_area))
        m_e_meadows = min(self.state["invasive_meadows"], self.budget//self.eradicate_meadow_cost)
        np.clip(e_meadows_units, a_min=0, a_max = m_e_meadows)

        norm_r_isolates_prop = action["restore_isolates"] / requested_proportion
        r_isolates_units = round(action["restore_isolates"] / (requested_proportion * self.total_area))
        m_r_isolates = min(self.state["native_isolates"], self.budget//self.restore_isolate_cost)
        np.clip(r_isolates_units, a_min=0, a_max = m_r_isolates)

        return e_isolates_units, e_meadows_units, r_isolates_units


    def get_action(self, action):
        # Bound each action by state-informed action space.
        #Remaining budget variable? Or proportionality of budget requested by out of bound actions.

        #Proportionality (Current)
        total_action_area = action["eradicate_isolate"]  + action["eradicate_meadow"] + action["restore_isolate"]
        action[eradicate_isolate] = round(action["eradicate_isolate"] / total_action_area * self.budget)
        action[eradicate_meadow] = round(action["eradicate_meadow"] / total_action_area * self.budget)
        action[restore_isolate] = round(action["restore_isolate"] / total_action_area * self.budget)

        #Clip each to budget
        # np.clip(action[eradicate_isolate], a_min = 0, a_max = min(self.state[invasive_isolates], self.budget//self.eradicate_isolate_cost))
        # np.clip(action[eradicate_meadow], a_min = 0, a_max = min(self.state[invasive_meadows], self.budget//self.eradicate_meadow_cost))
        # np.clip(action[restore_isolate], a_min = 0, a_max = min(self.budget//self.restore_isolate_cost, self.get_bare_area()))

        return action

    def get_invasive_population(self, eradicate_isolates, eradicate_meadows):
        """
        Gets updated invasive population based on units invasive spartina eradicated and growth matrix.
        Input:
        (int) eradicate_isolates : units of isolates to eradicate
        (int) eradicate_meadows : units of meadows to eradicate
        """
        new_invasive_isolates = (self.invasive_growth[0][0] * self.state["invasive_isolates"] - eradicate_isolates) + (self.invasive_growth[0][1] * self.state["invasive_meadows"] - eradicate_meadows)
        new_invasive_meadows = (self.invasive_growth[1][0] * self.state["invasive_isolates"] - eradicate_isolates) + (self.invasive_growth[1][1] * self.state["invasive_meadows"] - eradicate_meadows)

        return new_invasive_isolates, new_invasive_meadows
    
    def get_native_population(self, restore_isolates):
        """
        Gets updated native population based on units native spartina restored and growth matrix.
        Input:
        (int) restore_isolates : units of isolates to restore
        """
        new_native_isolates = (self.native_growth[0][0] * self.state["native_isolates"] + restore_isolates) + (self.native_growth[0][1] * self.state["native_meadows"])
        new_native_meadows = (self.native_growth[1][0] * self.state["native_isolates"] + restore_isolates) + (self.native_growth[1][1] * self.state["native_meadows"])

        return new_native_isolates, new_native_meadows

    def get_bare_area(self):
        return self.total_area - self.state["invasive_isolates"] - self.state["invasive_meadows"] - self.state["native_isolates"] - self.state["native_meadows"]