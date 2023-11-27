"""
This file contains unit tests for the RestrictedDictionary class in the MuJoCo_Gym.data_store module.

The RestrictedDictionary class is designed to store data in a dictionary-like structure, but with restrictions on which agents can access and modify the data. The class provides methods for setting the current agent, getting agent-specific subsets of the data, setting items in the dictionary, and committing changes to the dictionary.

The unit tests in this file cover various scenarios to ensure the correct functionality of the RestrictedDictionary class and its methods.
"""


import unittest
from MuJoCo_Gym.data_store import DataStore

class TestRestrictedDictionary(unittest.TestCase):
    """
    A test case for the RestrictedDictionary class.
    """
    
    def setUp(self):
        """
        Set up the test case by initializing the agents list and creating a RestrictedDictionary object.
        """
        self.agents_list = ["Agent1", "Agent2", "Agent3"]
        self.restricted_dict = DataStore(self.agents_list)

    def test_set_agent_valid(self):
        """
        Test case to verify the functionality of the 'set_agent' method in the 'restricted_dict' class.
        It sets the current agent to the specified agent name and checks if the current agent is updated correctly.
        """
        self.restricted_dict.set_agent("Agent1")
        self.assertEqual(self.restricted_dict.current_agent, "Agent1")

    def test_set_agent_invalid(self):
        """
        Test case to verify that setting an invalid agent raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.restricted_dict.set_agent("InvalidAgent")

    def test_get_agent_subset_valid(self):
        """
        Test case to verify the behavior of the get_agent_subset method when a valid agent name is provided.
        """
        agent_subset = self.restricted_dict.get_agent_subset("Agent1")
        self.assertEqual(agent_subset, {})

    def test_get_agent_subset_invalid(self):
        """
        Test case to verify that a ValueError is raised when trying to get an invalid agent subset.
        """
        with self.assertRaises(ValueError):
            self.restricted_dict.get_agent_subset("InvalidAgent")

    def test_setitem_valid(self):
        """
        Test case to validate the behavior of setting an item in the restricted dictionary.

        Steps:
        1. Set the agent to "Agent1" in the restricted dictionary.
        2. Set the value "value1" for the key "key1" in the restricted dictionary.
        3. Commit the changes made to the restricted dictionary.
        4. Assert that the value of the key "key1" in the restricted dictionary is "value1".
        """
        self.restricted_dict.set_agent("Agent1")
        self.restricted_dict["key1"] = "value1"
        self.restricted_dict.commit()
        self.assertEqual(self.restricted_dict["key1"], "value1")

    def test_commit(self):
        """
        Test the commit method of the restricted_dict.

        This method makes changes to the restricted_dict, commits the changes,
        and then checks if the changes are reflected in the actual dictionary.
        """
        # Make changes
        self.restricted_dict.set_agent("Agent1")
        self.restricted_dict["key1"] = "value1"
        self.assertNotEqual(self.restricted_dict["key1"], "value1")

        self.restricted_dict.set_agent("Agent2")
        self.restricted_dict["key2"] = "value2"
        self.assertNotEqual(self.restricted_dict["key2"], "value2")

        # Commit changes
        self.restricted_dict.commit()

        # Changes are now in the actual dictionary
        self.restricted_dict.set_agent("Agent1")
        self.assertEqual(self.restricted_dict["key1"], "value1")
        self.restricted_dict.set_agent("Agent2")
        self.assertEqual(self.restricted_dict["key2"], "value2")

    def test_repr(self):
        """
        Test the __repr__ method of the class.
        """
        self.assertEqual(repr(self.restricted_dict), repr({"Agent1": {}, "Agent2": {}, "Agent3": {}}))

if __name__ == '__main__':
    unittest.main()
