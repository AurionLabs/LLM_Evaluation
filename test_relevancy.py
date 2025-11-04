#RUN:
#deepeval test run test_relevancy.py

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from dotenv import load_dotenv

load_dotenv()


def test_relevancy():
    # Define the metric with a threshold
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")
    
    # Case 1: Partially relevant answer
    test_case_1 = LLMTestCase(
        input="Can I return these shoes after 30 days?",
        actual_output="Yes, you can return them. We offer a 30-day full refund. Do you have your original receipt?",
        retrieval_context=[
            "All customers are eligible for a 30-day full refund at no extra cost.",
            "Returns are only accepted within 30 days of purchase.",
        ],
    )
    
    # Case 2: Fully relevant answer
    test_case_2 = LLMTestCase(
        input="Can I return these shoes after 30 days?",
        actual_output="Unfortunately, returns are only accepted within 30 days of purchase.",
        retrieval_context=[
            "All customers are eligible for a 30-day full refund at no extra cost.",
            "Returns are only accepted within 30 days of purchase.",
        ],
    )
    
    # Run evaluation
    assert_test(test_case_1, [relevancy_metric])
    assert_test(test_case_2, [relevancy_metric])