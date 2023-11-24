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

    if sensor_info["type"] == "rangefinder":
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
    agent_sensors = [current for current in new_indices.values() if current["site"] in [site["@name"] for site in agent_sites]]
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
    return agent_sensors


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
        if sensor_type["type"] == "rangefinder":
            observation_space["low"].append(-1)
            observation_space["high"].append(float(sensor_type["cutoff"]))
        elif sensor_type["type"] == "frameyaxis":
            for _ in range(3):
                observation_space["low"].append(-360)
                observation_space["high"].append(360)

    return observation_space