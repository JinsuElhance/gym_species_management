from gym.envs.registration import register
from gym_species_management.envs.abstract_species_management_env import AbstractSpeciesManagementEnv

register(
    id = 'species_management-v0',
    entry_point = 'gym_species_management.envs:AbstractSpeciesManagementEnv',
)