from mujoco_rl import MuJoCo_RL 
import pytest 
import sys 
DIRECTORY = '/home/lisa/Mount/Dateien/StudyProject'
sys.path.insert(0, f'{DIRECTORY}/s.mujoco_environment/Testing')
from Pick_Up_Dynamic import Pick_Up_Dynamic
from Environment_Dynamic import Environment_Dynamic
from mock_dynamics import * 
from mock_done_functions import *
from mock_reward_functions import *


@pytest.fixture
def automated_testing_environment():
    experiment_config_dict = {"xmlPath":f"{DIRECTORY}/s.mujoco_environment/Environments/single_agent/ModelVis.xml", 
                              "infoJson":f"{DIRECTORY}/s.mujoco_environment/Environments/single_agent/info_example.json", 
                              "renderMode":False}
    return MuJoCo_RL(configDict = experiment_config_dict) 

# helpful links for pytest
# https://github.com/pixegami/simple-pytest-tutorial
# https://miguendes.me/how-to-check-if-an-exception-is-raised-or-not-with-pytest 

"""
1. __checkDynamics
"""

# accurate environment dynamics, namely Pick_Up_Dynamics, Environment_Dynamics
def test_checkDynamics_raises_no_exception(automated_testing_environment):
    try: 
        automated_testing_environment._MuJoCo_RL__checkDynamics([Environment_Dynamic, Pick_Up_Dynamic])
    except Exception as exc:
        assert False, f"__checkDynamics raised an exception for Dynamics {exc}"

# observation space
def test_checkDynamics_raises_observaitonSpace_bounds_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info: 
        automated_testing_environment._MuJoCo_RL__checkDynamics([Mock_Dynamic_Lower_Bound])

def test_checkDynamics_raises_observaitonSpace_len_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info: 
        automated_testing_environment._MuJoCo_RL__checkDynamics([Mock_Dynamic_Length])

# reward
def test_checkDynamics_raises_reward_no_float_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info:
        automated_testing_environment._MuJoCo_RL__checkDynamics([Mock_Dynamic_Reward])


"""
2. __checkDoneFunctions
"""

# accurate done function
def test_checkDoneFunctions_raises_no_exception(automated_testing_environment):
    try:
        automated_testing_environment._MuJoCo_RL__checkDoneFunctions([mock_doneFunction_correct])
    except Exception as exc:
        assert False, f"__checkDoneFunctions raised an exception for done function {exc}"

# done
def test_checkDoneFunctions_no_boolean_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info: 
        automated_testing_environment._MuJoCo_RL__checkDoneFunctions([mock_doneFunction_boolean_4, mock_doneFunction_boolean_string])

# reward
def test_checkDoneFunctions_no_float_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info: 
        automated_testing_environment._MuJoCo_RL__checkDoneFunctions([mock_doneFunction_reward_int])


"""
3. __checkRewardFunctions
"""

# accurate reward function
def test_checkRewardFunction_raises_no_exception(automated_testing_environment):
    try:
        automated_testing_environment._MuJoCo_RL__checkRewardFunctions([mock_rewardFunction_correct])
    except Exception as exc:
        assert False, f"__checkRewardFunctions raised an exception for reward function {exc}"

# reward
def test_checkRewardFunction_no_float_error(automated_testing_environment):
    with pytest.raises(Exception) as e_info:
        automated_testing_environment._MuJoCo_RL__checkRewardFunction([mock_rewardFunction_int, mock_rewardFunction_string])