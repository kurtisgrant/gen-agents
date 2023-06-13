from environment import EnvNode
from memory import Memory_Stream
from sandbox import Sandbox
import random

class Agent:
    def __init__(self, name, location, bio, sandbox=Sandbox()):
        self.sandbox = sandbox
        self.name = name
        self.location = location
        self.known_env = sandbox.env.get_node(location).to_dict()
        self.bio = bio
        self.current_task = None
        self.memory_stream = Memory_Stream(sandbox)
    
    def init_memory_from_bio(self):
        for chunk in self.bio.split(";"):
            self.memory_stream.add_memory(type="OBSERVATION", content=chunk)

    @staticmethod
    def burt():
        return Agent(
                name="Burt",
                location="Burt's House",
                bio="Burt is a friendly farmer who lives in a small village; Burt enjoys exploring the village and talking to other villagers; Burt likes to keep up with the latest news and gossip; Burt likes to keep his home tidy; Burt likes to read books; Burt is pasionate about horticulture; Burt likes to attend local events; Burt is friends with the shop owner Karen; Burt goes to the pub in town square every day at noon for a drink"
                )
    