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
        self.verbosity = 2

    def cast_importances_to_floats(self):
        for memory in self.memories:
            memory["importance"] = float(memory["importance"])

    def add_memory(self, type, content, references=None):
        if type not in ["OBSERVATION", "REFLECTION", "ACTION"]:
            raise Exception("Invalid memory type")
        if type == "REFLECTION" and references is None:
            raise Exception("Reflections must have references")
        memory = {
            "id": self.mem_count,
            "timestamp": self.sandbox.get_time(),
            "last_accessed": self.sandbox.get_time(),
            "type": type,
            "content": content,
            "importance": self.get_importance(content),
            "embedded_content": embeddings.embed_query(content)
        }
        if references:
            memory["references"] = references
        self.mem_count += 1
        self.memories.append(memory)
        self.memories_dict[memory["id"]] = memory  # Store memory in dict as well
        return memory
    
    def add_reflection(self, content, references):
        return self.add_memory(type="REFLECTION", content=content, references=references)

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
        return float(response['rating'])
    
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
    
    def query(self, query=None, num_results=5, weights=(0.5, 0.3, 0.2), num_memories=50):
        if self.verbosity > 0:
            print(f"\nQuerying: \"{query}\"...")
        recent_memories = self.get_recent_memories(num_memories)
        memory_scores = {}

        for memory in recent_memories:
            scores = {}
            if query is not None:
                query_embedding = embeddings.embed_query(query)
                scores['relevance'] = np.dot(memory["embedded_content"], query_embedding)
            else:
                scores['relevance'] = 0.0
            scores['importance'] = memory['importance']
            scores['recency'] = self.get_recency(memory['id'])
            scores['id'] = memory['id']
            memory_scores[memory['id']] = scores
        
        # Min-max normalize scores
        for score_type in ['relevance', 'importance', 'recency']:
            scores = [memory_scores[memory_id][score_type] for memory_id in memory_scores]
            min_score = min(scores)
            max_score = max(scores)
            for memory_id in memory_scores:
                # If all scores are the same, set all scores to 1
                if min_score == max_score:
                    memory_scores[memory_id][score_type] = 1.0
                memory_scores[memory_id][score_type] = (memory_scores[memory_id][score_type] - min_score) / (max_score - min_score)

        # Calculate final scores
        for memory_id, scores in memory_scores.items():
            memory_object = self.get_memory(memory_id)
            scores['score'] = weights[0] * scores['relevance'] + weights[1] * scores['importance'] + weights[2] * scores['recency']

            if self.verbosity > 2:
                print(f"\nMemory: [node_{memory_object['id']}] {memory_object['timestamp']}: {memory_object['content']}")
                print(f"Relevance: {scores['relevance']}")
                print(f"Importance: {scores['importance']}")
                print(f"Recency: {scores['recency']}")
                print(f"Score: {scores['score']}")

        # Sort memories by score
        top_memory_scores = sorted(memory_scores.values(), key=lambda x: x['score'], reverse=True)[:num_results]

        if self.verbosity > 1:
            print("Query results:")

        # Update the last accessed time for each memory
        for memory_scores in top_memory_scores:
            memory = self.get_memory(memory_scores['id'])
            if self.verbosity > 1:
                score = memory_scores['score']
                # Fix score to 3 decimal places
                score = "{:.3f}".format(score)
                print(f"Scoring {score}) [node_{memory['id']}] {memory['timestamp']}: {memory['content']}")
            self.access(memory["id"])
        
        # Get the actual unaltered memory objects
        memories_to_return = [self.get_memory(memory_scores['id']) for memory_scores in top_memory_scores]

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
        if self.verbosity > 0:
            print("\nExtracting insights...")
        insights_template = PromptTemplate(
            input_variables = ['memories', 'format_instructions'],
            template = 'What 3 high-level useful insights can you infer from the memories provided?\n\nMEMORIES: {memories}\n\n{format_instructions}\n\n'
        )
        insights_chain = LLMChain(llm=llm, prompt=insights_template, verbose=False)

        schema1 = ResponseSchema(name='insight1', description='The first insight', type='string')
        references1 = ResponseSchema(name='references1', description='List of ids for referenced memories for first insight. eg. "2, 5, 12"', type='string')
        schema2 = ResponseSchema(name='insight2', description='The second insight', type='string')
        references2 = ResponseSchema(name='references2', description='List of ids for referenced memories for second insight. eg. "2, 5, 12"', type='string')
        schema3 = ResponseSchema(name='insight3', description='The third insight', type='string')
        references3 = ResponseSchema(name='references3', description='List of ids for referenced memories for third insight. eg. "2, 5, 12"', type='string')
        output_parser = StructuredOutputParser.from_response_schemas([schema1, references1, schema2, references2, schema3, references3])
        format_instructions = output_parser.get_format_instructions()

        insights = output_parser.parse(insights_chain.run(memories="\n\n".join(memories_list), format_instructions=format_instructions))
        # Put pairs of insights and their references into a list of dicts
        insights = [{'insight': insights[f'insight{i}'], 'references': insights[f'references{i}']} for i in range(1, 4)]

        return insights
    
    def reflect(self, insight_memories=4, num_memories=25):
        if self.verbosity > 0:
            print("\n\nReflecting...")
        if self.verbosity > 1:
            print("\nIdentifying topics...")
        memory_list = self.get_recent_memories(num_memories)
        prompt_template = PromptTemplate(
            input_variables = ['memories', 'format_instructions'],
            template = 'Given only the memories provided, what are 3 most salient high-level topics we should reflect on regarding the subjects in the memories? MEMORIES: {memories}\n\n{format_instructions}\n\n'
        )
        generate_topics_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=self.verbosity > 3)

        topic1_schema = ResponseSchema(name='topic1', description='The first generated topic', type='string')
        topic2_schema = ResponseSchema(name='topic2', description='The second generated topic', type='string')
        topic3_schema = ResponseSchema(name='topic3', description='The third generated topic', type='string')
        output_parser = StructuredOutputParser.from_response_schemas([topic1_schema, topic2_schema, topic3_schema])
        format_instructions = output_parser.get_format_instructions()

        memories_string = "\n\n".join(fstring_memories(memory_list))
        topics = output_parser.parse(generate_topics_chain.run(memories=memories_string, format_instructions=format_instructions))

        topics_string = ', '.join(topics.values())
        
        if self.verbosity > 1:
            print(f"\nTopics: {topics_string}")

        insights = []

        # For each topic, use topic as query on memory stream
        for topic in topics.values():
            if self.verbosity > 2:
                print(f"\n\nTopic: {topic}")
            memories = self.query(query=topic, num_results=insight_memories, num_memories=num_memories)
            memory_strings = fstring_memories(memories)
            memories_string = '\n\n'.join(memory_strings)
            topic_insights = self.extract_insights(memory_strings)
            for insight in topic_insights:
                # If references contain one or more "node_" prefix, remove them
                references = insight['references']
                # Check if references string contains "node_" prefix
                if "node_" in references:
                    # Remove "node_" prefix from each reference
                    references = [reference.replace("node_", "") for reference in references.split(',')]
                    # Join references back into a string
                    references = ','.join(references)
                    # Update references in insight dict
                    insight['references'] = references
                insights.append(insight)
            topic_insights_string = '\n'.join([f"\n{insight['insight']}\nReferences: {insight['references']}" for insight in topic_insights])
            if self.verbosity > 2:
                print(f"\nNew reflections from insights: \n{topic_insights_string}")

        if self.verbosity > 0:
            print("\n\nGenerating memories from insights...")

        reflection_memories = []

        for insight_object in insights:
            insight = insight_object['insight']
            references = insight_object['references']
            memory = self.add_reflection(content=insight, references=references)
            reflection_memories.append(memory)
        
        reflection_memory_strings = fstring_memories(reflection_memories, verbose=True)
        reflection_memories_string = '\n'.join(reflection_memory_strings)
        if self.verbosity > 1:
            print(f"\nReflections: \n{reflection_memories_string}")

        if 0 < self.verbosity < 2:
            print(f"\n\nDone reflecting. {len(reflection_memories)} new reflections added to memory stream.")

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
        if 'references' in m and m['references'] is not None:
            memory_string += f" (because of {m['references']})"
        memory_strings.append(memory_string)
    return memory_strings
