from loguru import logger

JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of an AI-generated answer to a financial question.

Question: {question}
Generated Answer: {answer}
Retrieved Context: {context}
Expected Answer (ground truth, if available): {ground_truth}

Rate the answer on these dimensions, each from 1 (worst) to 5 (best):
1. Faithfulness: Is the answer fully supported by the retrieved context (no hallucinated facts)?
2. Relevancy: Does the answer directly address the question asked?

Respond ONLY in this exact format (no other text):
FAITHFULNESS: <score>
RELEVANCY: <score>
REASONING: <one sentence explanation>"""


def llm_judge_evaluation(question: str, answer: str, context: str, ground_truth: str, judge_gateway) -> dict:
    """
    Uses a separate judge LLM (Gemini, per the plan) to score faithfulness
    and relevancy of a generated answer — a lightweight replacement for
    RAGAS metrics, avoiding RAGAS's fragile dependency chain.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, answer=answer, context=context[:3000], ground_truth=ground_truth
    )

    try:
        chat_model = judge_gateway.get_chat_model("gemini")
        response = chat_model.invoke(prompt)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)

        faithfulness_score = None
        relevancy_score = None
        reasoning = ""

        for line in raw_text.split("\n"):
            if line.startswith("FAITHFULNESS:"):
                faithfulness_score = int(line.split(":")[1].strip())
            elif line.startswith("RELEVANCY:"):
                relevancy_score = int(line.split(":")[1].strip())
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return {
            "faithfulness": faithfulness_score,
            "relevancy": relevancy_score,
            "reasoning": reasoning,
        }
    except Exception as e:
        logger.error(f"LLM judge evaluation failed: {e}")
        return {"faithfulness": None, "relevancy": None, "reasoning": f"Judge error: {e}"}


def hallucination_spot_check(question: str, answer: str, ground_truth: str) -> dict:
    """
    Simple substring-based spot check: does the expected number/answer
    actually appear in the generated answer?
    """
    ground_truth_clean = ground_truth.replace(",", "").strip()
    answer_clean = answer.replace(",", "")

    found = ground_truth_clean in answer_clean if ground_truth_clean else None

    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer_snippet": answer[:200],
        "expected_value_found": found,
    }