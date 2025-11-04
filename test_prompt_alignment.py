#RUN:deepeval test run test_prompt_alignment.py
from deepeval import evaluate
from deepeval.metrics import PromptAlignmentMetric
from deepeval.test_case import LLMTestCase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

template = """Question: {question}
Answer: Answer in Upper case."""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model
query = "What is capital of India?"
input_data = {"question": query}
# Invoke the chain with input data and display the response
actual_output = chain.invoke(input_data).content
print("actual_output:", actual_output)

# Measuring prompt alignment
metric = PromptAlignmentMetric(
    prompt_instructions=["Reply in all uppercase"],
    model="gpt-4o-mini",
    include_reason=True
)
test_case = LLMTestCase(
    input=query,
    actual_output=actual_output
)

metric.measure(test_case)
print("metric.score:", metric.score)
print("metric.reason:", metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])