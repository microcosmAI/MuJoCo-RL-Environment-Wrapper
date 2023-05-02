class Mock_Dynamic_Lower_Bound():
    def __init__(self, mujoco_gym):
        observation_space = {"low" : [-70, -70, -70], "high" : [70, 70, 70]}

    def dynamic(self):
        return 3.0, [-80, 10, 2]
    
class Mock_Dynamic_Length():
    def __init__(self, mujoco_gym):
        observation_space = {"low" : [-70, -70, -70], "high" : [70, 70, 70]}

    def dynamic(self):
        return 3.0, [-80, 10]
    
class Mock_Dynamic_Reward():
    def __init__(self, mujoco_gym):
        observation_space = {"low" : [-70, -70, -70], "high" : [70, 70, 70]}

    def dynamic(self):
        return 3, [-80, 10]