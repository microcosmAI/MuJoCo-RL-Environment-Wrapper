def process_sensor(sensor, index):
    """
    Process a sensor and return sensor information and updated index.

    Args:
        sensor (dict): The sensor data.
        index (int): The current index.

    Returns:
        tuple: A tuple containing the sensor information (dict) and the updated index (int).
    """
    sensor_info = {
        "indices": list(range(index, index + len(sensor["data"]))),
        "site": sensor["site"],
        "type": sensor["type"]
    }

    if sensor_info["type"] in ("rangefinder" , "touch" , "accelerometer"):
        sensor_info["cutoff"] = sensor["cutoff"]

    return sensor_info, index + len(sensor["data"])


def extract_agent_indices(new_indices, agent_sites):
    """
    Extracts the indices of agent sensors from a dictionary of new indices.

    Parameters:
    new_indices (dict): A dictionary containing the new indices of sensors.
    agent_sites (list): A list of agent sites.

    Returns:
    agent_indices (list): A list of indices corresponding to agent sensors.
    agent_sensors (list): A list of dictionaries representing agent sensors.
    """
    agent_sensors = [current for current in new_indices.values() if
                     current["site"] in [site["@name"] for site in agent_sites]]
    agent_indices = [item for sublist in (current["indices"] for current in agent_sensors) for item in sublist]
    return agent_indices, agent_sensors


def process_sensors(indices, agent_sites):
    """
    Process the sensor indices and agent sites to extract agent sensors.

    Args:
        indices (dict): A dictionary containing sensor indices.
        agent_sites (list): A list of agent sites.

    Returns:
        list: A list of agent sensors.
    """
    new_indices = {}
    index = 0

    for _, sensor_info in sorted(indices.items()):
        sensor_name = sensor_info["name"]
        new_indices[sensor_name], index = process_sensor(sensor_info, index)

    agent_indices, agent_sensors = extract_agent_indices(new_indices, agent_sites)
    return agent_indices, agent_sensors


def create_observation_space(agent_sensors):
    """
    Creates the observation space based on the given agent sensors.

    Args:
        agent_sensors (list): List of dictionaries representing the agent sensors.

    Returns:
        dict: Dictionary representing the observation space with "low" and "high" values.
    """
    observation_space = {"low": [], "high": []}
    # Creates the observation space from the sensors.
    for sensor_type in agent_sensors:
        if sensor_type["type"] in ["touch", "actuatorpos", "clock"]:
            observation_space["low"].append(0)
            observation_space["high"].append(float(sensor_type["cutoff"]))
        elif sensor_type["type"] in ["accelerometer", "velocimeter", "gyro", "force", "torque", "magnetometer", "framepos", "ballangvel", "framelinvel", "frameangvel", "framelinacc", "frameangacc"]:
            for _ in range(3):
                observation_space["low"].append(-1 * float(sensor_type["cutoff"]))
                observation_space["high"].append(float(sensor_type["cutoff"]))
        elif sensor_type["type"] == "rangefinder":
            observation_space["low"].append(-1)
            observation_space["high"].append(float(sensor_type["cutoff"]))
        elif sensor_type["type"] in ["jointlimitpos", "jointlimitvel", "jointlimitfrc", "tendonlimitpos", "tendonlimitvel", "tendonlimitfrc"]:
            observation_space["low"].append(-1 * float(sensor_type["cutoff"]))
            observation_space["high"].append(0)
        elif sensor_type["type"] == "camprojection":
            for _ in range(2):  # 2D projection
                observation_space["low"].append(0)
                observation_space["high"].append(float(sensor_type["cutoff"]))
        elif sensor_type["type"] in ["ballquat", "framequat"]:
            for _ in range(4):
                observation_space["low"].append(-1 * float(sensor_type["cutoff"]))
                observation_space["high"].append(float(sensor_type["cutoff"]))
        elif sensor_type["type"] in ["framexaxis", "frameyaxis", "framezaxis"]:
            for _ in range(3):
                observation_space["low"].append(-1)
                observation_space["high"].append(1)
        elif sensor_type["type"] in ["subtreecom", "subtreelinvel", "subtreeangmom", "jointpos", "jointvel", "tendonpos", "tendonvel", "actuatorvel", "actuatorfrc", "jointactuatorfrc"]:
            observation_space["low"].append(-1 * float(sensor_type["cutoff"]))
            observation_space["high"].append(float(sensor_type["cutoff"]))
        elif sensor_type["type"] == "user":
            observation_space["low"].append(-1)
            observation_space["high"].append(1)
        elif sensor_type["type"] == "plugin":
            # Handle plugin sensor type
            # The observation space for plugin sensors may vary widely, depending on the plugin
            # need to determine the observation space based on the specific plugin and its functionality
            # with arbitrary values:
            observation_space["low"].append(0)
            observation_space["high"].append(100)

    return observation_space


'''

def process_user_sensor(sensor):
    """
    Process the user sensor and return sensor information.

    Args:
        sensor (dict): The sensor data.

    Returns:
        dict: Sensor information.
    """
    sensor_info = {
        "name": sensor["name"],
        "objtype": sensor.get("objtype", "mjOBJ_UNKNOWN"),
        "objname": sensor.get("objname", None),
        "datatype": sensor.get("datatype", "real"),
        "needstage": sensor.get("needstage", "acc"),
        "dim": int(sensor["dim"]),
        "user": sensor["user"]
    }
    return sensor_info



def process_plugin_sensor(sensor):
    """
    Process the plugin sensor and return sensor information.

    Args:
        sensor (dict): The sensor data.

    Returns:
        dict: Sensor information.
    """
    sensor_info = {
        "name": sensor["name"],
        "plugin": sensor.get("plugin", None),
        "instance": sensor.get("instance", None),
        "objtype": sensor["objtype"],
        "objname": sensor["objname"],
        "reftype": sensor.get("reftype", None),
        "refname": sensor.get("refname", None),
        "user": sensor.get("user", None)
    }
    return sensor_info

'''
