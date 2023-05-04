<a name="readme-top"></a>

<br />
<div align="center">
  <h2 align="center">MuJoCo environment</h2>

  <p align="center">
    A python environment for multi agent training in MuJoCo simulations.
    <br />
    <a href="https://github.com/microcosmAI/s.mujoco_environment/wiki"><strong>Explore the wiki docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/microcosmAI/s.mujoco_environment/issues">Report Bug</a>
    ·
    <a href="https://github.com/microcosmAI/s.mujoco_environment/issues">Request Feature</a>
  </p>
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/HedgehockOfMuJoCo.png" alt="Logo" width="80%">
  </a>
</div>

<br/>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Environment Setup](#environment-setup)
  - [Language channel](#language-channel)
  - [Reward and Done function](#reward-and-done-function)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## About The Project

In this repository, we publish a python wrapper that can be used to train a large variety of different environments with reinforcement learning environments.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

Clone this repository, navigate with your terminal into this repository and execute the following steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
  ```sh
  pip install -r requirements.txt
  ```

### Installation

To use the environment, you have to install this repository as a pip package. Alternativly you can open a branch of this repository and implement changes directly in this repo.

1. Navigate to the repository with your terminal.
2. Install the repository as a pip package
   ```sh
   pip install .
   ```
3. Check whether the installation was successful
   ```sh
   python -c "import MuJoCo_Gym"
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage
### Environment Setup

The basic multi agent environment can be imported and used like this:<br/><br/>
First the path for the environment has to be set. Additionaly you need to provide a list of agent names within the environment. Those names correspond to the top level body of your agent within the xml file. The json file containing additional information is optional.
```python
from MuJoCo_Gym.mujoco_rl import MuJoCo_RL

environment_path = "Examples/Environment/MultiEnvs.xml" # File containing the mujoco environment
info_path = "Examples/Environment/info_example.json"     # File containing addtional environment informations
agents = ["agent1", "agent2"]                           # List of agents (body names) within the environment
```
These informations have to be stored in a dictionary. This is necessary to make the environment compatible with Ray.
```python
config_dict = {"xmlPath":environment_path, "infoJson":info_path, "agents":agents}
environment = mujoco_rl(config_dict)
```
Reset the environment to start the simulation.
```python
observation, infos = environment.reset()
```
Store the action of each agent in a dictionary with the agent names as keys. The array has to match the shape of the action space and the single agents have to be part of the action range.
```python
actions = {"agent1":np.array([]), "agent2":np.array([])}
observations, rewards, terminations, truncations, infos = environment.step(actions)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Language channel
To use a language channel, you have to implement it as a environment dynamic. Each environment dynamic has its own observation and action space, which will be forwarded to the agents. Note that at the moment each agent gets all environment dynamics and each dynamic is executed for each agent once during every timestep.<br/><br/>
A basic implementation of a language channel in the environment. Note that every environment dynamic needs to implement a init(self, mujoco_gym) and a dynamic(self, agent, actions).
```python
class Language():

    def __init__(self, mujoco_gym):
        self.mujoco_gym = mujoco_gym
        self.observation_space = {"low":[0], "high":[3]}
        self.action_space = {"low":[0], "high":[3]}
        # The datastore is used to store and preserve data over one or multiple timesteps
        self.dataStore = {}

    def dynamic(self, agent, actions):

        # At timestep 0, the utterance field has to be initialized
        if "utterance" not in self.mujoco_gym.dataStore[agent].keys():
            self.mujoco_gym.dataStore[agent]["utterance"] = 0

        # Extract the utterance from the agents action
        utterance = int(actions[0])

        # Store the utterance in the dataStore for the environment
        self.mujoco_gym.dataStore[agent]["utterance"] = utterance
        otherAgent = [other for other in self.mujoco_gym.agents if other!=agent][0]

        # Check whether the other agent has "spoken" yet (not at timestep 0)
        if "utterance" in self.mujoco_gym.dataStore[otherAgent]:
            utteranceOtherAgent = self.mujoco_gym.dataStore[otherAgent]["utterance"]
            return 0, np.array([utteranceOtherAgent])
        else:
            return 0, np.array([0])
```
The environment dynamic has to be added to the environment config.
```python
config_dict = {"xmlPath":environment_path, "infoJson":info_path, "agents":agents, "environmentDynamics":[Language]}
environment = mujoco_rl(config_dict)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Reward and Done function
A reference implementation of a reward function that gives back a positive reward if the agent gets closer to a target object. All possible target objects are filtered by tags at the beginning. Those tags are set in the info json file, which is handed over in the config dict at the beginning.
```python
def reward_function(mujoco_gym, agent):
    # Creates all the necessary fields to store the needed data within the dataStore at timestep 0 
    if "targets" not in mujoco_gym.dataStore[agent].keys():
        mujoco_gym.dataStore["targets"] = mujoco_gym.filterByTag("target")
        mujoco_gym.dataStore[agent]["current_target"] = mujoco_gym.dataStore["targets"][random.randint(0, len(mujoco_gym.dataStore["targets"]) - 1)]["name"]
        distance = mujoco_gym.distance(agent, mujoco_gym.dataStore[agent]["current_target"])
        mujoco_gym.dataStore[agent]["distance"] = distance
        new_reward = 0
    else: # Calculates the distance between the agent and the current target
        distance = mujoco_gym.distance(agent, mujoco_gym.dataStore[agent]["current_target"])
        new_reward = mujoco_gym.dataStore[agent]["distance"] - distance
        mujoco_gym.dataStore[agent]["distance"] = distance
    reward = new_reward * 10
    return reward
```
The done function end the current training run if the agent gets closer than 1 distance unit to the target.
```python
def done_function(mujoco_gym, agent):
    if mujoco_gym.dataStore[agent]["distance"] <= 1:
        return True
    else:
        return False
```
Both of them have to be included in the config dictionary.
```python
config_dict = {"xmlPath":environment_path, "infoJson":info_path, "agents":agents, "rewardFunctions":[reward_function], "doneFunctions":[done_function]}
environment = mujoco_rl(config_dict)
```
For more examples, please refer to the [Wiki](https://github.com/microcosmAI/s.mujoco_environment/wiki).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact

Cornelius Wolff - cowolff@uos.de

<p align="right">(<a href="#readme-top">back to top</a>)</p>
