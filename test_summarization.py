from deepeval import evaluate
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

class ExampleSchema(BaseModel):
    name: str

# This is the original text to be summarized
text = """
Rice is the staple food of Bengal. Bhortas (lit-"mashed") are a really common type of food used as an additive too rice. there are several types of Bhortas such as Ilish bhorta shutki bhorta, begoon bhorta and more. Fish and other seafood are also important because Bengal is a reverrine region.
Some fishes like puti (Puntius species) are fermented. Fish curry is prepared with fish alone or in combination with vegetables.Shutki maach is made using the age-old method of preservation where the food item is dried in the sun and air, thus removing the water content. This allows for preservation that can make the fish last for months, even years in Bangladesh
"""

template = """Question: {question}
Answer:  Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model
query ="Summarize the text for me %s"%(text)
input_data = {"question": query}
# Invoke the chain with input data and display the response in Markdown format
actual_output = chain.invoke(input_data).content
print("actual_output:", actual_output)

test_case = LLMTestCase(input=text, actual_output=actual_output)
metric = SummarizationMetric(
    threshold=0.7,
    model="gpt-4o-mini",
)

metric.measure(test_case)
print("metric.score:", metric.score)
print("metric.reason:", metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])