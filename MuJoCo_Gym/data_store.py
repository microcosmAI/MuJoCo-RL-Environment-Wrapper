class DataStore:
    """
    A dictionary-like data structure that restricts access based on agents.

    Attributes:
        data (dict): The main data store containing agent-specific dictionaries.
        buffer (dict): A temporary buffer to store changes before committing.
        current_agent (str): The currently active agent.

    Methods:
        set_agent(agent): Sets the current agent.
        get_agent_subset(agent): Returns the agent-specific dictionary for reading.
        __getitem__(key): Returns the value associated with the key for the current agent.
        __setitem__(key, value): Sets the value associated with the key for the current agent.
        commit(): Commits the changes in the buffer to the main data store.
    """

    def __init__(self, agents):
        """
        Initializes a DataStore object.

        Parameters:
        agents (list): A list of agent names.

        Attributes:
        data (dict): A dictionary to store data for each agent.
        buffer (dict): A dictionary to store buffer data for each agent.
        current_agent (str): The name of the current agent.
        """
        self.data = {agent: {} for agent in agents}
        self.buffer = {agent: {} for agent in agents}
        self.current_agent = None

    def set_agent(self, agent):
        """
        Sets the current agent for writing data to the dictionary.

        Parameters:
            agent (str): The name of the agent.

        Raises:
            ValueError: If the agent is not allowed to write to the dictionary.
        """
        if agent not in self.data and agent != "global":
            raise ValueError(f"Agent {agent} is not allowed to write to this dictionary.")
        self.current_agent = agent

    def get_agent_subset(self, agent):
        """
        Retrieves the subset of data for a specific agent.

        Args:
            agent (str): The name of the agent.

        Returns:
            dict: The subset of data for the specified agent.

        Raises:
            ValueError: If the agent is not allowed to read from this dictionary.
        """
        if agent not in self.data:
            raise ValueError(f"Agent {agent} is not allowed to read from this dictionary.")
        return self.data[agent]

    def __getitem__(self, key):
        """
        Get the value associated with the given key for the current agent.

        Args:
            key: The key to retrieve the value for.

        Returns:
            The value associated with the given key for the current agent.

        Raises:
            ValueError: If no agent is currently set.
        """
        if self.current_agent is None:
            raise ValueError("No agent is currently set.")
        return self.get_agent_subset(self.current_agent).get(key)

    def __setitem__(self, key, value):
        """
        Sets the value for the given key in the data store.

        Args:
            key: The key to set the value for.
            value: The value to set.

        Raises:
            ValueError: If no agent is currently set or if the current agent is not allowed to write to the dictionary.
        """
        if self.current_agent is None:
            raise ValueError("No agent is currently set.")
        if self.current_agent not in self.data:
            raise ValueError(f"Agent {self.current_agent} is not allowed to write to this dictionary.")
        if self.current_agent == "global":
            self.buffer[key] = value
        else:
            self.buffer[self.current_agent][key] = value

    def commit(self):
        """
        Commits the changes in the buffer to the data store.
        """
        for agent, changes in self.buffer.items():
            self.data[agent].update(changes)
        self.buffer = self.data.copy()

    def __repr__(self):
        """
        Returns a string representation of the dictionary.
        """
        return repr(self.data)