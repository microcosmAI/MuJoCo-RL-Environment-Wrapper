from mujoco_rl import MuJoCo_RL 
import pytest 
import sys 
DIRECTORY = '/home/lisa/Mount/Dateien/StudyProject'
sys.path.insert(0, f'{DIRECTORY}/s.mujoco_environment/Testing')
from Pick_Up_Dynamic import Pick_Up_Dynamic
from Environment_Dynamic import Environment_Dynamic
import mock_dynamics    


@pytest.fixture
def automated_testing_environment():
    experiment_config_dict = {"xmlPath":f"{DIRECTORY}/s.mujoco_environment/Environments/multi_agent/info_example.json", 
                              "renderMode":False}
    return MuJoCo_RL(configDict = experiment_config_dict) 

# checkDynamics
# https://miguendes.me/how-to-check-if-an-exception-is-raised-or-not-with-pytest 
def test_checkDynamics_raises_no_exception(automated_testing_environment):
    try: 
        automated_testing_environment._MuJoCo_RL__checkDynamics([Pick_Up_Dynamic, Environment_Dynamic])
    except Exception as exc:
        assert False, f"__checkDynamics raised an exception for Pick_Up_Dynamics {exc}"


def test_checkDynamics_raises_observaitonSpace_bounds_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info: 
        automated_testing_environment._MuJoCo_RL__checkDynamics([mock_dynamics.Mock_Dynamic_Lower_Bound])

def test_checkDynamics_raises_observaitonSpace_len_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info: 
        automated_testing_environment._MuJoCo_RL__checkDynamics([mock_dynamics.Mock_Dynamic_Length])

def test_checkDynamics_raises_reward_no_float_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info:
        automated_testing_environment._MuJoCo_RL__checkDynamics([mock_dynamics.Mock_Dynamic_Reward])




