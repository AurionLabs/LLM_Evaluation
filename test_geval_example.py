#RUN:deepeval test run test_geval_example.py

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from dotenv import load_dotenv
load_dotenv()

correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o-mini",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also lightly penalize omission of detail, and focus on the main idea",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
)

first_test_case = LLMTestCase(input="What are the main causes of deforestation?",
                              actual_output="The main causes of deforestation include agricultural expansion, logging, infrastructure development, and urbanization.",
                              expected_output="The main causes of deforestation include agricultural expansion, logging, infrastructure development, and urbanization.")


second_test_case = LLMTestCase(input="Define the term 'artificial intelligence'.",
                               actual_output="Artificial intelligence is the simulation of human intelligence by machines.",
                               expected_output="Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think and learn like humans, including tasks such as problem-solving, decision-making, and language understanding.")


third_test_case = LLMTestCase(input="List the primary colors.",
                              actual_output="The primary colors are green, orange, and purple.",
                              expected_output="The primary colors are red, blue, and yellow.")

test_cases = [first_test_case, second_test_case, third_test_case]
for test_case in test_cases:
    assert_test(test_case, [correctness_metric])