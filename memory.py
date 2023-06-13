import os
from apikey import apikey

from collections import defaultdict
import heapq
import numpy as np
import datetime
import random
import json

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from text import importance_examples, importance_advice

os.environ['OPENAI_API_KEY'] = apikey

llm = OpenAI(temperature=0.9)

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
            "importance": self.get_importance(content),
            "embedded_content": embeddings.embed_query(content)
        }
        self.mem_count += 1
        self.memories.append(memory)
        self.memories_dict[memory["id"]] = memory  # Store memory in dict as well
        return memory

    def get_importance(self, memory_str):

        importance_schema = ResponseSchema(name='rating', description=f'The score given to the memory. A number between 1 and 10. {importance_advice}', type='number')
        output_parser = StructuredOutputParser.from_response_schemas([importance_schema])
        format_instructions = output_parser.get_format_instructions()

        importance_template = PromptTemplate(
            input_variables = ['query', 'format_instructions'],
            template="Please rate the importance of the following memory on a scale of 1-10: {query} \n\n{format_instructions}\n\n"
        )

        importance_chain = LLMChain(llm=llm, prompt=importance_template, verbose=False)
        format_instructions = output_parser.get_format_instructions()
        # On error, log and try again up to 5 times
        for i in range(5):
            try:
                response = output_parser.parse(importance_chain.run(query=memory_str, format_instructions=format_instructions))
                break
            except Exception as e:
                print(e)
                continue
        return response['rating']
    
    def recalculate_importances(self, verbose=False):
        for memory in self.memories:
            if verbose:
                print(f"Calculating importance of memory {memory['id']}")
                print(f"Memory: {memory['content']}")
                print(f"Importance: {memory['importance']}")
            memory["importance"] = self.get_importance(memory["content"])
            if verbose:
                print(f"New importance: {memory['importance']}")


    def get_memory(self, id):
        if id not in self.memories_dict:
            raise Exception(f"Memory with id {id} not found")
        return self.memories_dict[id]
    
    # Export memories to JSON file
    def export(self, filename):
        filename = filename + ".mems"
        with open(filename, "w") as f:
            json.dump(self.memories, f)
    
    # Import memories from JSON file
    def import_memories(self, filename):
        filename = filename + ".mems"
        with open(filename, "r") as f:
            self.memories = json.load(f)
        self.memories_dict = {memory["id"]: memory for memory in self.memories}
        self.mem_count = len(self.memories)
        # Update sandbox time to the last time a memory was added
        self.sandbox.set_time(self.memories[-1]["timestamp"])

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
    
    def query(self, query=None, num_results=5, weights=(0.5, 0.3, 0.2), verbose=False):
        recent_memories = self.get_recent_memories()
        memory_scores = []

        for memory in recent_memories:
            if query is not None:
                query_embedding = embeddings.embed_query(query)
                relevance = np.dot(memory["embedded_content"], query_embedding)
            else:
                relevance = 0
            importance = float(memory["importance"]) / 10.0  # Normalize importance
            recency = self.get_recency(memory["id"])

            # Weight the scores
            weighted_score = weights[0] * relevance + weights[1] * importance + weights[2] * recency

            if verbose:
                print(f"\nMemory: [node_{memory['id']}] {memory['timestamp']}: {memory['content']}")
                print(f"Relevance: {relevance}")
                print(f"Weighted relevance: {weights[0] * relevance}")
                print(f"Importance: {importance}")
                print(f"Weighted importance: {weights[1] * importance}")
                print(f"Recency: {recency}")
                print(f"Weighted recency: {weights[2] * recency}")
                print(f"Weighted score: {weighted_score}")

            memory_scores.append((weighted_score, memory))

        # Use heapq to get the top memories
        top_memories = heapq.nlargest(num_results, memory_scores)
        # Extract the memory from each tuple in top_memories
        memories_to_return = [memory for score, memory in top_memories]
        # Update the last accessed time for each memory
        for score, memory in top_memories:
            if verbose:
                print(f"\nMemory: [node_{memory['id']}] {memory['timestamp']}: {memory['content']}")
                print(f"Weighted score: {score}")
            self.access(memory["id"])
        return memories_to_return

    
    def get_recent_memories(self, number=100):
        return self.memories[-number:]

    def print_memory(self, id):
        memory = self.get_memory(id).copy()  # get a copy so original memory is not affected
        memory.pop('embedded_content', None)  # remove the embedding from the copy
        print(memory)

    def get_memory_content(self, memories_list):
        return [memory["content"] for memory in memories_list]
    
    def extract_insights(self, memories_list):
        insights_template = PromptTemplate(
            input_variables = ['memories', 'format_instructions'],
            template = 'What 5 high-level insights can you infer from the memories provided  \n\nMEMORIES: {memories}\n\n{format_instructions}\n\n'
        )
        insights_chain = LLMChain(llm=llm, prompt=insights_template, verbose=True)

        schema1 = ResponseSchema(name='insight1', description='The first insight', type='string')
        schema2 = ResponseSchema(name='insight2', description='The second insight', type='string')
        schema3 = ResponseSchema(name='insight3', description='The third insight', type='string')
        schema4 = ResponseSchema(name='insight4', description='The fourth insight', type='string')
        schema5 = ResponseSchema(name='insight5', description='The fifth insight', type='string')
        output_parser = StructuredOutputParser.from_response_schemas([schema1, schema2, schema3, schema4, schema5])
        format_instructions = output_parser.get_format_instructions()

        insights = output_parser.parse(insights_chain.run(memories="\n\n".join(memories_list), format_instructions=format_instructions))
        return insights
    
    def reflect(self, verbose=False):
        memory_list = self.get_recent_memories(50)
        prompt_template = PromptTemplate(
            input_variables = ['memories', 'format_instructions'],
            template = 'Given only the memories above, what are 3 most salient high-level questions we can answer about the subjects in the statements? MEMORIES: {memories}\n\n{format_instructions}\n\n'
        )
        generate_questions_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

        question1_schema = ResponseSchema(name='question1', description='The first generated question', type='string')
        question2_schema = ResponseSchema(name='question2', description='The second generated question', type='string')
        question3_schema = ResponseSchema(name='question3', description='The third generated question', type='string')
        output_parser = StructuredOutputParser.from_response_schemas([question1_schema, question2_schema, question3_schema])
        format_instructions = output_parser.get_format_instructions()

        memories_string = "\n\n".join(fstring_memories(memory_list))
        questions = output_parser.parse(generate_questions_chain.run(memories=memories_string, format_instructions=format_instructions))

        # For each question, use question as query on memory stream
        for question in questions.values():
            if verbose:
                print(f"\n\nQuestion: {question}")
            memories = self.query(query=question, num_results=5, verbose=verbose)
            memory_strings = fstring_memories(memories)
            memories_string = '\n\n'.join(memory_strings)
            if verbose:
                print(f"\n\nRelated Memories: {memories_string}")
            reflections = self.extract_insights(memory_strings)
            if verbose:
                print(f"\n\nReflections: {reflections}")
            for reflection in reflections.values():
                self.add_memory(content=reflection, type='REFLECTION')

    def load_observations_through_time(self, observations):
        for ob in observations:
            # Advance time 1-3 times
            for _ in range(random.randint(1, 3)):
                self.sandbox.advance()
            self.add_memory(type='OBSERVATION', content=ob)



def fstring_memories(memories, verbose=False):
    memory_strings = []
    for m in memories:
        memory_string = f"[node_{m['id']}] {m['timestamp']}: {m['content']}"
        if verbose:
            memory_string = f"[node_{m['id']}] {m['timestamp']} (i:{m['importance']}): {m['content']}"
        memory_strings.append(memory_string)
    return memory_strings

observation_examples = [
    "common room sofa is idle",
    "common room sofa is being sat on by Francisco Lopez",
    "Francisco Lopez is watching a comedy show",
    "common room table is idle",
    "cooking area is idle",
    "kitchen sink is idle",
    "toaster is idle",
    "refrigerator is idle",
    "closet is idle",
    "bed is idle",
    "desk is idle",
    "Abigail Chen is playing a game",
    "desk is cluttered with books and papers",
    "Abigail Chen is browsing the internet",
    "bed is occupied",
    "Abigail Chen is reading a book",
    "desk is unoccupied",
    "Abigail Chen is checking her emails",
    "closet is idle",
    "bed is idle",
    "desk is idle",
    "cooking area is idle",
    "kitchen sink is idle",
    "toaster is idle",
    "refrigerator is idle",
    "common room table is strewn with snacks and drinks",
    "Rajiv Patel is discussing the show with friends",
    "Abigail Chen is checking her emails",
    "common room table is idle",
    "Abigail Chen is taking a few deep breaths",
    "Hailey Johnson is brainstorming ideas for her novel",
    "common room sofa is idle",
    "common room sofa is being sat on",
    "common room sofa is in use",
    "Rajiv Patel is watching the show",
    "common room sofa is being sat on by Hailey Johnson",
    "Hailey Johnson is watching the new show",
    "common room table is idle",
    "Abigail Chen is taking a few deep breaths",
    "common room table is being used as a platform for a laptop and books",
    "Abigail Chen is browsing the web for inspiration",
    "common room sofa is idle",
    "common room sofa is in use",
    "Rajiv Patel is watching the show",
    "common room sofa is being sat on by Hailey Johnson",
    "Hailey Johnson is watching the new show",
    "common room table is idle",
    "cooking area is idle",
    "kitchen sink is idle",
    "toaster is idle",
    "refrigerator is idle",
    "Abigail Chen is browsing the web for inspiration",
    "desk is empty and unused",
    "Abigail Chen is checking her emails",
    "bed is unoccupied",
    "closet is idle",
    "bed is idle",
    "desk is idle",
    "Abigail Chen is checking her phone for notifications",
    "cooking area is idle",
    "kitchen sink is idle",
    "toaster is idle",
    "refrigerator is idle",
    "common room sofa is in use",
    "Rajiv Patel is watching the show",
    "common room sofa is being sat on by Hailey Johnson",
    "Hailey Johnson is watching the new show",
    "Abigail Chen is checking her phone for notifications",
    "common room sofa is idle",
    "common room table is idle",
    "common room sofa is being used by Abigail Chen",
    "Abigail Chen is taking a break to stretch",
    "common room sofa is in use",
    "Rajiv Patel is watching the show",
    "common room sofa is being sat on by Hailey Johnson",
    "Hailey Johnson is watching the new show",
]
