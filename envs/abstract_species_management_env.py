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
        Beta = 0.1 #Cost Discount Coefficient
    }, 
    init_state = (,) 

    TMax = 9, 
    CR_min = 100, # Minimum Clapper Rail Habitat Constraint
    L_H = [[]], #Invasive Growth Constant Matrix
    L_F = [[]], #Native Growht Constant Matrix
    B = 1000, #Annual Budget
    C_E_I = 10, #Cost of Eradicating Isolates
    C_E_M = 20, #Cost of Eradicating Meadows
    C_R_I = 15, #Cost of Restoring Isolates

    file = None) {
    """
    
    """
    }
}