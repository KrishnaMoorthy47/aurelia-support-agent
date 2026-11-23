import pytest


@pytest.mark.parametrize("query,doc_fragment", [
    ("how much is the withdrawal fee", "Withdrawal fees"),
    ("what happens at stop-out", "stop-out"),
    ("how long does verification take", "How long verification takes"),
    ("do you charge inactivity fees", "Inactivity fee"),
    ("what is the minimum deposit", "Deposit methods"),
])
def test_in_domain_questions_find_the_right_section_in_top_3(retriever, query, doc_fragment):
    res = retriever.search(query)
    assert res.confident
    # top-3, not top-1: the offline keyword embedder is deliberately crude; the real model is judged by the eval suite
    assert any(doc_fragment.lower() in h.chunk.citation.lower() for h in res.hits)


@pytest.mark.parametrize("query", [
    "what is the weather in paris",
    "tell me a joke about cats",
    "write me a python script",
    "what is the capital of france",
])
def test_off_topic_questions_are_not_confident(retriever, query):
    assert not retriever.search(query).confident


def test_every_hit_has_a_citation(retriever):
    for h in retriever.search("how are margin calls handled").hits:
        assert " > " in h.chunk.citation


def test_rewrite_brings_in_the_right_section(retriever):
    plain = retriever.search("I want to leave for good")
    fused = retriever.search("I want to leave for good", "account closure")
    assert not any("Account closure" in h.chunk.citation for h in plain.hits)
    assert any("Account closure" in h.chunk.citation for h in fused.hits)


def test_identical_rewrite_is_ignored(retriever):
    a = retriever.search("what is the withdrawal fee")
    b = retriever.search("what is the withdrawal fee", "What is the withdrawal fee")
    assert [h.chunk.id for h in a.hits] == [h.chunk.id for h in b.hits]
