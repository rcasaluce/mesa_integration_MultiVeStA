import mesa
import random
import math


"""
Mesa Schelling Segregation Model

Adapted from: https://github.com/projectmesa/mesa-examples/blob/main/examples/schelling/model.py

"""

class SchellingAgent(mesa.Agent):
    """
    Schelling segregation agent
    """

    def __init__(self, pos, model, agent_type):
        """
        Create a new Schelling agent.

        Args:
           unique_id: Unique identifier for the agent.
           pos: Agent initial location.
           agent_type: Indicator for the agent's type (minority=1, majority=0)
        """
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type

    def step(self):
        similar = 0
        for neighbor in self.model.grid.iter_neighbors(self.pos, True):
            if neighbor.type == self.type:
                similar += 1

        # If unhappy, move:
        if similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1


class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(self, width=20, height=20, density=0.8, minority_pc=0.2, homophily=3):
        super().__init__()
        self.width = width
        self.height = height
        self.homophily = homophily
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily

        #self.grid = mesa.space.SingleGrid(width, height, torus=True)
        
        self.happy = 0

        

    def set_simulator_for_new_simulation(self, seed):
        random.seed(seed)
        self.random.seed(seed)
        self.reset_randomizer(seed)
        # No need to redefine self.schedule here
        # Create scheduler and assign it to the model
        if(len(self.agents)) > 0:
            for agent in self.agents:
                agent.remove()
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(self.width, self.height, torus=True)

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for _, pos in self.grid.coord_iter():
            if self.random.random() < self.density:
                agent_type = 1 if self.random.random() < self.minority_pc else 0
                agent = SchellingAgent(pos, self, agent_type)
                self.grid.place_agent(agent, pos)
                self.schedule.add(agent)

    def get_distance(self, pos1, pos2):
        """
        Calculate the Euclidean distance between two positions.
        """
        x1, y1 = pos1
        x2, y2 = pos2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def eval(self, obs):
        """
        Evaluate method.
        """
    
        if obs == 'avgHappy':
            return float(self.happy / len(self.agents))
        elif obs == 'dissatisfactionRate':
            return 1.0 - (float(self.happy) / len(self.agents))

         
       
        elif obs == 'schellingProximityIndex':
            # Schelling's proximity index measures the average distance between agents of the same type.
            # It indicates the level of clustering or dispersion of similar agents.
            total_distance = 0
            count = 0
            for agent in self.agents:
                for neighbor in self.grid.iter_neighbors(agent.pos, True):
                    if neighbor.type == agent.type:
                        total_distance += self.get_distance(agent.pos, neighbor.pos)
                        count += 1
            if count == 0:
                return 0  # Avoid division by zero
            else:
                return total_distance / count
        elif obs == 'averageClusteringCoefficient':
            # Calcolo del coefficiente di clustering medio
            clustering_coefficients = []
            for agent in self.agents:
                similar_neighbors = [neighbor for neighbor in self.grid.iter_neighbors(agent.pos, moore=True) if neighbor.type == agent.type]
                num_edges = len(similar_neighbors)
                if num_edges > 1:
                    num_connected_edges = 0
                    for i in range(len(similar_neighbors)):
                        for j in range(i + 1, len(similar_neighbors)):
                            if self.get_distance(similar_neighbors[i].pos, similar_neighbors[j].pos) <= math.sqrt(2):
                                num_connected_edges += 1
                    clustering_coefficients.append(num_connected_edges / (num_edges * (num_edges - 1) / 2))  # Normalizing by the number of possible edges
            if clustering_coefficients:
                return sum(clustering_coefficients) / len(clustering_coefficients)
            else:
                return 0
        
        



    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.
        """
        self.schedule.step()
        self.happy = 0  # Reset counter of happy agents
        self.agents.shuffle().do("step")
        # Must be before data collection.
        self._advance_time()  # Temporary API; will be finalized by Mesa 3.0 release
        

        if self.happy == len(self.agents):
            self.running = False

    def getTime(self):
        return self.schedule.time   