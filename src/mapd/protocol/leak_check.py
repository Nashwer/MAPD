from mapd.reward.exact_match import normalize_answer


def contains_answer(text: str, answers: list[str]) -> bool:
    text_tokens = normalize_answer(text).split()
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens or len(answer_tokens) > len(text_tokens):
            continue
        width = len(answer_tokens)
        if any(text_tokens[index : index + width] == answer_tokens for index in range(len(text_tokens) - width + 1)):
            return True
    return False


def hidden_answers(question: str, answers: list[str]) -> list[str]:
    """Return answer aliases that are not already observable in the question.

    Some HotpotQA comparison questions name both candidates verbatim.  Treating
    the correct candidate as an oracle leak in that case rejects every useful
    query containing the publicly supplied entity.  Only answer strings absent
    from the model-visible question can constitute label injection.
    """
    return [answer for answer in answers if not contains_answer(question, [answer])]


def protocol_has_leak(reasoning_plan: list[str], queries: list[str], answers: list[str]) -> bool:
    return any(contains_answer(step, answers) for step in reasoning_plan) or any(
        contains_answer(query, answers) for query in queries
    )
