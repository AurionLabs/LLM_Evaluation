from deepeval import evaluate
from deepeval.metrics import JsonCorrectnessMetric
from deepeval.test_case import LLMTestCase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

class ExampleSchema(BaseModel):
    name: str

# Querying the model
template = """Question: {question}
Answer:  Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model
query ="Output me a random Json with the 'name' key"
input_data = {"question": query}
# Invoke the chain with input data and display the response
actual_output = chain.invoke(input_data).content
print("actual_output:", actual_output)

# Measuring Json correctness
metric = JsonCorrectnessMetric(
    expected_schema=ExampleSchema,
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