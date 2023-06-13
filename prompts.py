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


def rate_importance(memory_str):
    importance_schema = ResponseSchema(name='rating', description='The importance rating of the memory. A number between 1 and 10.', type='number')
    output_parser = StructuredOutputParser.from_response_schemas([importance_schema])
    format_instructions = output_parser.get_format_instructions()

    rate_importance_template = PromptTemplate(
        input_variables = ['query', 'format_instructions'],
        template="Please rate the importance of the following memory on a scale of 1-10, where 1 is 'mundane' and 10 is 'very interesting': {query} \n\n{format_instructions}\n\n"
    )

    rate_importance_chain = LLMChain(llm=llm, prompt=rate_importance_template, verbose=False)
    format_instructions = output_parser.get_format_instructions()
    response = output_parser.parse(rate_importance_chain.run(query=memory_str, format_instructions=format_instructions))
    return response['rating']

def answer_question(question, memories_list):
    answer_template = PromptTemplate(
        input_variables = ['question', 'memories', 'format_instructions'],
        template = 'Using the context provided, answer the following questions. QUESTION: {question}\n\nCONTEXT_MEMORIES: {memories}\n\n{format_instructions}\n\n'
    )
    answer_chain = LLMChain(llm=llm, prompt=answer_template, verbose=True)

    answer_schema = ResponseSchema(name='answer', description='The answer to the question', type='string')
    output_parser = StructuredOutputParser.from_response_schemas([answer_schema])
    format_instructions = output_parser.get_format_instructions()

    memories_str = '\n\n'.join(memories_list)
    answer = output_parser.parse(answer_chain.run(question=question, memories=memories_str, format_instructions=format_instructions))
    return answer

def get_reflections(memory_list):
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

    memories_string = '\n\n'.join(memory_list)
    questions = output_parser.parse(generate_questions_chain.run(memories=memories_string, format_instructions=format_instructions))

    print(questions)
    # Return the answers to the questions as a list

    return [answer_question(question, memory_list) for question in questions.values()]

