from gym.envs.registration import register

register(
    id = 'species_management-v0',
    entry_pount = 'gym_species_management.envs:SpeciesManagementEnv',
)