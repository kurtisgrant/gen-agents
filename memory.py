from collections import defaultdict
import heapq
from prompts import rate_importance
import numpy as np
import datetime
import random
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

class Memory_Stream:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.mem_count = 0
        self.memories = []
        self.memories_dict = defaultdict(dict)  # New dictionary for quick access

    def add_memory(self, type, content):
        if type not in ["OBSERVATION", "REFLECTION", "ACTION"]:
            raise Exception("Invalid memory type")
        memory = {
            "id": self.mem_count,
            "timestamp": self.sandbox.get_time(),
            "last_accessed": self.sandbox.get_time(),
            "type": type,
            "content": content,
            "importance": rate_importance(content),
            "embedded_content": embeddings.embed_query(content)
        }
        self.mem_count += 1
        self.memories.append(memory)
        self.memories_dict[memory["id"]] = memory  # Store memory in dict as well

    def get_memory(self, id):
        if id not in self.memories_dict:
            raise Exception(f"Memory with id {id} not found")
        return self.memories_dict[id]

    def access(self, id):
        memory = self.get_memory(id)
        memory["last_accessed"] = self.sandbox.get_time()

    def get_recency(self, id):
        memory = self.get_memory(id)
        current_time = datetime.datetime.strptime(self.sandbox.get_time(), "%Y-%m-%d %H:%M")
        last_accessed_time = datetime.datetime.strptime(memory["last_accessed"], "%Y-%m-%d %H:%M")
        hours_since_last_accessed = (current_time - last_accessed_time).total_seconds() / 3600
        recency_score = 0.99 ** hours_since_last_accessed
        return recency_score

    def query(self, query=None, num_results=5):
        recent_memories = self.get_most_recent_memory()
        memory_scores = []

        for memory in recent_memories:
            if query is not None:
                query_embedding = embeddings.embed_query(query)
                relevance = np.dot(memory["embedded_content"], query_embedding)
            else:
                relevance = 0
            importance = float(memory["importance"]) / 10.0  # Cast importance to float for precision
            recency = self.get_recency(memory["id"])
            memory_scores.append((relevance + importance + recency, memory))

        # Use heapq to get the top memories
        top_memories = heapq.nlargest(num_results, memory_scores)
        # Extract the memory from each tuple in top_memories
        memories_to_return = [memory for score, memory in top_memories]
        # Update the last accessed time for each memory
        for memory in memories_to_return:
            self.access(memory["id"])
        return memories_to_return
    
    def get_most_recent_memory(self, number=50):
        return self.memories[-number:]

    def print_memory(self, id):
        memory = self.get_memory(id).copy()  # get a copy so original memory is not affected
        memory.pop('embedded_content', None)  # remove the embedding from the copy
        print(memory)

    def get_memory_content(self, memories_list):
        return [memory["content"] for memory in memories_list]
    
    @staticmethod
    def get_memory_strings(self, memory_list, timestamps=True, type=True, details=False):
        memory_strings = []
        for memory in memory_list:
            memory_string = ""
            if timestamps:
                memory_string += f"{memory['timestamp']} - "
            if type:
                memory_string += f"{memory['type']}: "
            memory_string += f"{memory['content']}"
            if details:
                memory_string += f"\n(id: {memory['id']}, importance: {memory['importance']}, recency: {self.get_recency(memory['id'])}, last accessed: {memory['last_accessed']})"
            memory_strings.append(memory_string)
        return memory_strings

    def load_burt_memories(self):
        temp_memories = [
            {'id': 0, 'timestamp': '2023-06-11 07:00', 'last_accessed': '2023-06-11 07:00', 'type': 'OBSERVATION', 'content': 'Burt rises with the sun, ready to start his day.', 'importance': 6},
            {'id': 1, 'timestamp': '2023-06-11 07:05', 'last_accessed': '2023-06-11 07:05', 'type': 'OBSERVATION', 'content': 'Burt fetches fresh water from the well for his morning coffee.', 'importance': '6'},
            {'id': 2, 'timestamp': '2023-06-11 07:10', 'last_accessed': '2023-06-11 07:10', 'type': 'OBSERVATION', 'content': 'With a cup in hand, Burt admires the morning dew on his crops.', 'importance': '7'},
            {'id': 3, 'timestamp': '2023-06-11 07:25', 'last_accessed': '2023-06-11 07:25', 'type': 'OBSERVATION', 'content': 'Burt heads to the barn to tend to his animals.', 'importance': 5},
            {'id': 4, 'timestamp': '2023-06-11 07:30', 'last_accessed': '2023-06-11 07:30', 'type': 'OBSERVATION', 'content': 'The chickens cluck happily as Burt gathers the eggs for the day.', 'importance': '4'},
            {'id': 5, 'timestamp': '2023-06-11 07:35', 'last_accessed': '2023-06-11 07:35', 'type': 'OBSERVATION', 'content': 'Burt takes a moment to pet his old farm dog, Rover.', 'importance': '8'},
            {'id': 6, 'timestamp': '2023-06-11 07:50', 'last_accessed': '2023-06-11 07:50', 'type': 'OBSERVATION', 'content': 'Burt visits the pig pen, giving each pig a healthy serving of feed.', 'importance': 4},
            {'id': 7, 'timestamp': '2023-06-11 07:55', 'last_accessed': '2023-06-11 07:55', 'type': 'OBSERVATION', 'content': 'Burt spends some time brushing down his horse, Buttercup.', 'importance': '7'},
            {'id': 8, 'timestamp': '2023-06-11 08:10', 'last_accessed': '2023-06-11 08:10', 'type': 'OBSERVATION', 'content': 'Buttercup neighs happily, appreciating the attention.', 'importance': 8},
            {'id': 9, 'timestamp': '2023-06-11 08:15', 'last_accessed': '2023-06-11 08:15', 'type': 'OBSERVATION', 'content': 'Burt sets to work in the fields, tending to his crops.', 'importance': '5'},
            {'id': 10, 'timestamp': '2023-06-11 08:20', 'last_accessed': '2023-06-11 08:20', 'type': 'OBSERVATION', 'content': 'He notices that the corn is growing taller each day.', 'importance': '7'},
            {'id': 11, 'timestamp': '2023-06-11 08:35', 'last_accessed': '2023-06-11 08:35', 'type': 'OBSERVATION', 'content': "Burt makes a note to repair a section of the fence that's starting to sag.", 'importance': '3'},
            {'id': 12, 'timestamp': '2023-06-11 08:50', 'last_accessed': '2023-06-11 08:50', 'type': 'OBSERVATION', 'content': 'He takes a break under an old oak tree, sipping his water from a flask.', 'importance': '7'},
            {'id': 13, 'timestamp': '2023-06-11 09:00', 'last_accessed': '2023-06-11 09:00', 'type': 'OBSERVATION', 'content': 'The postman drives by, waving at Burt, delivering the mail.', 'importance': '4'},
            {'id': 14, 'timestamp': '2023-06-11 09:05', 'last_accessed': '2023-06-11 09:05', 'type': 'OBSERVATION', 'content': 'Burt reads a letter from his sister who lives in the city.', 'importance': '6'},
            {'id': 15, 'timestamp': '2023-06-11 09:15', 'last_accessed': '2023-06-11 09:15', 'type': 'OBSERVATION', 'content': 'A light breeze picks up, and Burt can smell rain on the horizon.', 'importance': '8'},
            {'id': 16, 'timestamp': '2023-06-11 09:30', 'last_accessed': '2023-06-11 09:30', 'type': 'OBSERVATION', 'content': 'Burt decides to repair the fence before the rain comes.', 'importance': '5'},
            {'id': 17, 'timestamp': '2023-06-11 09:45', 'last_accessed': '2023-06-11 09:45', 'type': 'OBSERVATION', 'content': "With the fence mended, Burt is satisfied with his morning's work.", 'importance': 5},
            {'id': 18, 'timestamp': '2023-06-11 09:55', 'last_accessed': '2023-06-11 09:55', 'type': 'OBSERVATION', 'content': 'As he walks back to the house, Burt spots a fox darting into the woods.', 'importance': '8'},
            {'id': 19, 'timestamp': '2023-06-11 10:00', 'last_accessed': '2023-06-11 10:00', 'type': 'OBSERVATION', 'content': "Burt enters his quaint home, it's time for a hearty lunch.", 'importance': 5},
            {'id': 20, 'timestamp': '2023-06-11 10:15', 'last_accessed': '2023-06-11 10:15', 'type': 'OBSERVATION', 'content': 'Over lunch, Burt plans out the rest of his day.', 'importance': '3'},
            {'id': 21, 'timestamp': '2023-06-11 10:25', 'last_accessed': '2023-06-11 10:25', 'type': 'OBSERVATION', 'content': 'Finishing lunch, Burt heads out despite the drizzling rain.', 'importance': 5},
            {'id': 22, 'timestamp': '2023-06-11 10:35', 'last_accessed': '2023-06-11 10:35', 'type': 'OBSERVATION', 'content': 'Burt spends the afternoon fixing a leak in the barn roof.', 'importance': 4},
            {'id': 23, 'timestamp': '2023-06-11 10:45', 'last_accessed': '2023-06-11 10:45', 'type': 'OBSERVATION', 'content': 'He spots a family of birds nesting in the rafters of the barn.', 'importance': '8'},
            {'id': 24, 'timestamp': '2023-06-11 11:00', 'last_accessed': '2023-06-11 11:00', 'type': 'OBSERVATION', 'content': 'Taking a break, Burt enjoys watching the birds flutter around.', 'importance': 8},
            {'id': 25, 'timestamp': '2023-06-11 11:10', 'last_accessed': '2023-06-11 11:10', 'type': 'OBSERVATION', 'content': 'Burt visits his vegetable garden, picking some fresh produce for dinner.', 'importance': '7'},
            {'id': 26, 'timestamp': '2023-06-11 11:25', 'last_accessed': '2023-06-11 11:25', 'type': 'OBSERVATION', 'content': 'The tomatoes are ripe and juicy, perfect for a salad.', 'importance': '5'},
            {'id': 27, 'timestamp': '2023-06-11 11:30', 'last_accessed': '2023-06-11 11:30', 'type': 'OBSERVATION', 'content': 'Burt smiles as he observes a healthy patch of pumpkins growing nicely.', 'importance': '7'},
            {'id': 28, 'timestamp': '2023-06-11 11:35', 'last_accessed': '2023-06-11 11:35', 'type': 'OBSERVATION', 'content': "As the rain stops, a rainbow forms in the sky, much to Burt's delight.", 'importance': '10'},
            {'id': 29, 'timestamp': '2023-06-11 11:40', 'last_accessed': '2023-06-11 11:40', 'type': 'OBSERVATION', 'content': 'Burt takes a moment to appreciate the beauty of his surroundings.', 'importance': 8},
            {'id': 30, 'timestamp': '2023-06-11 11:45', 'last_accessed': '2023-06-11 11:45', 'type': 'OBSERVATION', 'content': 'In the late afternoon, Burt stacks hay bales in the barn.', 'importance': '3'},
            {'id': 31, 'timestamp': '2023-06-11 11:55', 'last_accessed': '2023-06-11 11:55', 'type': 'OBSERVATION', 'content': 'Burt greets a neighbour passing by on the country road.', 'importance': 4},
            {'id': 32, 'timestamp': '2023-06-11 12:05', 'last_accessed': '2023-06-11 12:05', 'type': 'OBSERVATION', 'content': 'He helps his neighbor fix a flat tire on his old truck.', 'importance': 8},
            {'id': 33, 'timestamp': '2023-06-11 12:20', 'last_accessed': '2023-06-11 12:20', 'type': 'OBSERVATION', 'content': 'With the sun setting, Burt begins to wrap up his tasks for the day.', 'importance': 5},
            {'id': 34, 'timestamp': '2023-06-11 12:35', 'last_accessed': '2023-06-11 12:35', 'type': 'OBSERVATION', 'content': 'Burt feeds his animals one last time for the day.', 'importance': '7'},
            {'id': 35, 'timestamp': '2023-06-11 12:45', 'last_accessed': '2023-06-11 12:45', 'type': 'OBSERVATION', 'content': 'Rover follows Burt around, wagging his tail excitedly.', 'importance': '8'},
            {'id': 36, 'timestamp': '2023-06-11 13:00', 'last_accessed': '2023-06-11 13:00', 'type': 'OBSERVATION', 'content': 'Burt takes a walk around his property, checking everything is secure for the night.', 'importance': '3'},
            {'id': 37, 'timestamp': '2023-06-11 13:15', 'last_accessed': '2023-06-11 16:05', 'type': 'OBSERVATION', 'content': 'The fireflies start to come out, dotting the evening with their soft glow.', 'importance': '10'},
            {'id': 38, 'timestamp': '2023-06-11 13:25', 'last_accessed': '2023-06-11 13:25', 'type': 'OBSERVATION', 'content': 'Burt enjoys a simple dinner in his warm, cozy kitchen.', 'importance': '7'},
            {'id': 39, 'timestamp': '2023-06-11 13:40', 'last_accessed': '2023-06-11 13:40', 'type': 'OBSERVATION', 'content': 'Post-dinner, Burt spends some time whittling by the fireplace.', 'importance': '7'},
            {'id': 40, 'timestamp': '2023-06-11 13:50', 'last_accessed': '2023-06-11 13:50', 'type': 'OBSERVATION', 'content': 'Burt decides to turn in early for the night.', 'importance': '1'},
            {'id': 41, 'timestamp': '2023-06-11 13:55', 'last_accessed': '2023-06-11 13:55', 'type': 'OBSERVATION', 'content': 'He makes sure the fire in the fireplace is safely extinguished.', 'importance': 5},
            {'id': 42, 'timestamp': '2023-06-11 14:00', 'last_accessed': '2023-06-11 14:00', 'type': 'OBSERVATION', 'content': 'Burt takes one last look outside, everything is calm and peaceful.', 'importance': '8'},
            {'id': 43, 'timestamp': '2023-06-11 14:05', 'last_accessed': '2023-06-11 14:05', 'type': 'OBSERVATION', 'content': "Rover settles down at the foot of Burt's bed, ready to sleep.", 'importance': 6},
            {'id': 44, 'timestamp': '2023-06-11 14:15', 'last_accessed': '2023-06-11 14:15', 'type': 'OBSERVATION', 'content': 'Burt brushes his teeth, following his nightly routine.', 'importance': 4},
            {'id': 45, 'timestamp': '2023-06-11 14:25', 'last_accessed': '2023-06-11 14:25', 'type': 'OBSERVATION', 'content': 'He listens to the soft hooting of an owl outside his window.', 'importance': '8'},
            {'id': 46, 'timestamp': '2023-06-11 14:40', 'last_accessed': '2023-06-11 14:40', 'type': 'OBSERVATION', 'content': 'Burt checks on Buttercup one last time, who is already dozing in her stable.', 'importance': '8'},
            {'id': 47, 'timestamp': '2023-06-11 14:55', 'last_accessed': '2023-06-11 14:55', 'type': 'OBSERVATION', 'content': 'The farm is quiet, the animals are all settling down for the night.', 'importance': '7'},
            {'id': 48, 'timestamp': '2023-06-11 15:05', 'last_accessed': '2023-06-11 16:05', 'type': 'OBSERVATION', 'content': 'Burt takes a moment to admire the clear, starry night sky.', 'importance': '10'},
            {'id': 49, 'timestamp': '2023-06-11 15:10', 'last_accessed': '2023-06-11 16:05', 'type': 'OBSERVATION', 'content': 'The moon is full and bright, casting long shadows across the farm.', 'importance': '10'},
            {'id': 50, 'timestamp': '2023-06-11 15:20', 'last_accessed': '2023-06-11 15:20', 'type': 'OBSERVATION', 'content': 'Burt climbs into bed, pulling the quilt up to his chin.', 'importance': '5'},
            {'id': 51, 'timestamp': '2023-06-11 15:35', 'last_accessed': '2023-06-11 15:35', 'type': 'OBSERVATION', 'content': 'As he closes his eyes, he listens to the faint sounds of the farm at night.', 'importance': 8},
            {'id': 52, 'timestamp': '2023-06-11 15:40', 'last_accessed': '2023-06-11 15:40', 'type': 'OBSERVATION', 'content': 'The day has been long and fulfilling, and Burt quickly drifts off to sleep.', 'importance': 3},
            {'id': 53, 'timestamp': '2023-06-11 15:55', 'last_accessed': '2023-06-11 15:55', 'type': 'OBSERVATION', 'content': "Even in sleep, Burt's mind is filled with plans for the next day's work.", 'importance': '7'},
        ]
        for mem in temp_memories:
            mem["embedded_content"] = embeddings.embed_query(mem["content"])
            self.memories.append(mem)
            self.memories_dict[mem["id"]] = mem
            # Advance time by 1-3 steps between each observation
            for _ in range(random.randint(1, 3)):
                self.sandbox.advance()
        print("Burt's memories loaded.")


