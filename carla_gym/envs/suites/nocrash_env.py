from carla_gym import CARLA_GYM_ROOT_DIR
from carla_gym.carla_multi_agent_env import CarlaMultiAgentEnv
from carla_gym.utils import config_utils
import json


class NoCrashEnv(CarlaMultiAgentEnv):
    def __init__(self, carla_map, host, port, seed, no_rendering, obs_configs, reward_configs, terminal_configs,
                 weather_group, route_description, background_traffic):
        all_tasks = self.build_all_tasks(carla_map, weather_group, route_description, background_traffic)
        super().__init__(carla_map, host, port, seed, no_rendering,
                         obs_configs, reward_configs, terminal_configs, all_tasks)

    @staticmethod
    def build_all_tasks(carla_map, weather_group, route_description, background_traffic):
        assert carla_map in ['Town01', 'Town02']
        assert weather_group in ['new', 'train', 'train_eval']
        assert route_description in ['cexp', 'lbc', 'driving-benchmarks']
        assert background_traffic in ['empty', 'regular', 'dense', 'leaderboard']

        # weather
        if weather_group == 'new':
            weathers = ['SoftRainSunset', 'WetSunset']
        elif weather_group == 'train':
            weathers = ['ClearNoon', 'WetNoon', 'HardRainNoon', 'ClearSunset']
        elif weather_group == 'train_eval':
            weathers = ['WetNoon', 'ClearSunset']

        # zombie vehicles and walkers
        if carla_map == 'Town01':
            if background_traffic == 'empty':
                num_zombie_vehicles = 0
                num_zombie_walkers = 0
            elif background_traffic == 'regular':
                num_zombie_vehicles = 20
                num_zombie_walkers = 50
            elif background_traffic == 'dense':
                num_zombie_vehicles = 100
                num_zombie_walkers = 250
            elif background_traffic == 'leaderboard':
                num_zombie_vehicles = 120
                num_zombie_walkers = 120
        elif carla_map == 'Town02':
            if background_traffic == 'empty':
                num_zombie_vehicles = 0
                num_zombie_walkers = 0
            elif background_traffic == 'regular':
                num_zombie_vehicles = 15
                num_zombie_walkers = 50
            elif background_traffic == 'dense':
                num_zombie_vehicles = 70
                num_zombie_walkers = 150
            elif background_traffic == 'leaderboard':
                num_zombie_vehicles = 70
                num_zombie_walkers = 70

        description_folder = CARLA_GYM_ROOT_DIR / 'envs/scenario_descriptions/NoCrash' / route_description / carla_map
        actor_configs_dict = json.load(open(description_folder / 'actors.json'))
        route_descriptions_dict = config_utils.parse_routes_file(description_folder / 'routes.xml')

        all_tasks = []
        for weather in weathers:
            for route_id, route_description in route_descriptions_dict.items():
                task = {
                    'weather': weather,
                    'description_folder': description_folder,
                    'route_id': route_id,
                    'num_zombie_vehicles': num_zombie_vehicles,
                    'num_zombie_walkers': num_zombie_walkers,
                    'ego_vehicles': {
                        'routes': route_description['ego_vehicles'],
                        'actors': actor_configs_dict['ego_vehicles'],
                    },
                    'scenario_actors': {
                        'routes': route_description['scenario_actors'],
                        'actors': actor_configs_dict['scenario_actors']
                    } if 'scenario_actors' in actor_configs_dict else {}
                }
                all_tasks.append(task)

        return all_tasks
