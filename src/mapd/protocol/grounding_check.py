def facts_are_extractive(facts: list[str], passages: list[str]) -> bool:
    normalized_passages = [passage.replace("\r\n", "\n") for passage in passages]
    return all(
        any(fact.replace("\r\n", "\n") in passage for passage in normalized_passages)
        for fact in facts
    )

