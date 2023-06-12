import datetime
from environment import EnvNode

class Sandbox:
    def __init__(self, start_time=None, end_time=None):

        self.env = EnvNode.init()
        self.agents = []
        self.start_time = start_time if start_time else datetime.time(7, 0)
        self.end_time = end_time if end_time else datetime.time(22, 0)

        # Set simulation start time to today's date plus the specified start time
        self.simulation_time = datetime.datetime.combine(datetime.date.today(), self.start_time)

    def get_time(self):
        return self.simulation_time.strftime("%Y-%m-%d %H:%M") # Convert datetime to string

    def advance(self):
        # Add 5 minutes to the simulation time
        self.simulation_time += datetime.timedelta(minutes=5)

        # If we've passed the end time for the day, skip ahead to the start time on the next day
        if self.simulation_time.time() >= self.end_time:
            self.simulation_time += datetime.timedelta(days=1)
            self.simulation_time = self.simulation_time.replace(hour=self.start_time.hour, minute=self.start_time.minute)

    def add_agent(self, agent):
        self.agents.append(agent)

    def print_agent_info(self):
        for agent in self.agents:
            print(f"{agent.name} is at {agent.location} doing {agent.current_task}")


