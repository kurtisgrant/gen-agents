import os
from apikey import apikey

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.memory import ConversationBufferMemory
# from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

flatten_graph_template = PromptTemplate(
    input_variables = ['subgraph'],
    template = 'Here is a JSON representation of your environment as you know it. Please provide a natural language description of it. ENVIRONMENT: {subgraph}'
)

llm = OpenAI(temperature=0.9)
flatten_graph = LLMChain(llm=llm, prompt=flatten_graph_template, verbose=True)

importance_schema = ResponseSchema(name='rating', description='The importance rating of the memory. A number between 1 and 10.', type='number')
output_parser = StructuredOutputParser.from_response_schemas([importance_schema])
format_instructions = output_parser.get_format_instructions()

rate_importance_template = PromptTemplate(
    input_variables = ['query', 'format_instructions'],
    template="Please rate the importance of the following memory on a scale of 1-10, where 1 is 'mundane' and 10 is 'very interesting': {query} \n\n{format_instructions}\n\n"
)


rate_importance_chain = LLMChain(llm=llm, prompt=rate_importance_template, verbose=False)

def rate_importance(memory_str):
    format_instructions = output_parser.get_format_instructions()
    response = output_parser.parse(rate_importance_chain.run(query=memory_str, format_instructions=format_instructions))
    return response['rating']


generate_questions_template = PromptTemplate(
    input_variables = ['memories'],
    template = 'Given only the memories above, what are 3 most salient high-level questions we can answer about the subjects in the statements? MEMORIES: {memories}'
)

question1_schema = ResponseSchema(name='question1', description='The first generated question', type='string')
question2_schema = ResponseSchema(name='question2', description='The second generated question', type='string')
question3_schema = ResponseSchema(name='question3', description='The third generated question', type='string')
output_parser_questions = StructuredOutputParser.from_response_schemas([question1_schema, question2_schema, question3_schema])
format_instructions_questions = output_parser_questions.get_format_instructions()

generate_questions_chain = LLMChain(llm=llm, prompt=generate_questions_template, verbose=True)

def generate_questions(memory_list):
    format_instructions = output_parser_questions.get_format_instructions()
    memory_str = '\n\n'.join(memory_list)
    questions = output_parser_questions.parse(generate_questions_chain.run(memories=memory_str, format_instructions=format_instructions))
    return questions

def answer_question(question, memories_list):
    answer_template = PromptTemplate(
        input_variables = ['question', 'memories'],
        template = 'QUESTION: {question}\n\nCONTEXT_MEMORIES: {memories}\n\nANSWER: '
    )
    memories_str = '\n\n'.join(memories_list)
    answer_chain = LLMChain(llm=llm, prompt=answer_template, verbose=True)
    answer = answer_chain.run(question=question, memories=memories_str)
    return answer