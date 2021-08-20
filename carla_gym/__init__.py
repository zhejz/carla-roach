from pathlib import Path
from gym.envs.registration import register

# CARLA_GYM_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CARLA_GYM_ROOT_DIR = Path(__file__).resolve().parent

# Declare available environments with a brief description
_AVAILABLE_ENVS = {
    'NoCrash-v0': {
        'entry_point': 'carla_gym.envs:NoCrashEnv',
        'description': 'Empty background traffic.',
        'kwargs': {
            'background_traffic': 'empty'
        }
    },
    'NoCrash-v1': {
        'entry_point': 'carla_gym.envs:NoCrashEnv',
        'description': 'Regular background traffic.',
        'kwargs': {
            'background_traffic': 'regular'
        }
    },
    'NoCrash-v2': {
        'entry_point': 'carla_gym.envs:NoCrashEnv',
        'description': 'Dense background traffic.',
        'kwargs': {
            'background_traffic': 'dense'
        }
    },
    'NoCrash-v3': {
        'entry_point': 'carla_gym.envs:NoCrashEnv',
        'description': 'Moderate background traffic.',
        'kwargs': {
            'background_traffic': 'leaderboard'
        }
    },
    'CoRL2017-v0': {
        'entry_point': 'carla_gym.envs:CoRL2017Env',
        'description': 'straight',
        'kwargs': {
            'task_type': "straight"
        }
    },
    'CoRL2017-v1': {
        'entry_point': 'carla_gym.envs:CoRL2017Env',
        'description': 'one_curve',
        'kwargs': {
            'task_type': "one_curve"
        }
    },
    'CoRL2017-v2': {
        'entry_point': 'carla_gym.envs:CoRL2017Env',
        'description': 'navigation',
        'kwargs': {
            'task_type': "navigation"
        }
    },
    'CoRL2017-v3': {
        'entry_point': 'carla_gym.envs:CoRL2017Env',
        'description': 'navigation_dynamic',
        'kwargs': {
            'task_type': "navigation_dynamic"
        }
    },
    'Endless-v0': {
        'entry_point': 'carla_gym.envs:EndlessEnv',
        'description': 'endless env for rl training and testing',
        'kwargs': {}
    },
    'LeaderBoard-v0': {
        'entry_point': 'carla_gym.envs:LeaderboardEnv',
        'description': 'leaderboard route with no-that-dense backtround traffic',
        'kwargs': {}
    }
}


for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get('entry_point'), kwargs=val.get('kwargs'))


def list_available_envs():
    print('Environment-ID: Short-description')
    import pprint
    available_envs = {}
    for env_id, val in _AVAILABLE_ENVS.items():
        available_envs[env_id] = val.get('description')
    pprint.pprint(available_envs)
