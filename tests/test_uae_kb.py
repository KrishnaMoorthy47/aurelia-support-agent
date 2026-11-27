"""UAE knowledge-base documents and incremental knowledge-base sync."""
import pytest

from app.rag.chunking import load_chunks
from app.rag.embedders import LexicalEmbedder
from app.rag.ingest import sync
from app.rag.stores import MemoryStore


@pytest.mark.parametrize("query,section", [
    ("can I verify my account with my Emirates ID", "Documents for UAE residents"),
    ("who regulates Aurelia in the UAE", "Who regulates Aurelia in the UAE"),
    ("is my client money segregated from Aurelia's own money", "How your money is protected"),
    ("how long do AED bank transfer deposits take", "Deposits and withdrawals in AED"),
    ("why do you ask about my source of funds", "Why we ask about your source of funds"),
    ("how long are my records kept after closing the account", "How long records are kept"),
    ("can I fund my trading account with a credit card", "Credit cards cannot be used to fund trading"),
    ("do you offer a deposit bonus or trading credits", "No deposit bonuses or trading incentives"),
    ("when are all my positions closed automatically under the DFSA rules", "Automatic close-out at 50%"),
    ("an account manager on WhatsApp promised me guaranteed profits", "Fake account managers on WhatsApp and social media"),
    ("how do I check that a broker website is not a clone", "Clone websites and fake apps"),
    ("they asked me to pay a release fee before I can withdraw", "Requests to pay fees or send money"),
    ("do you have Islamic swap-free accounts", "What a swap-free account is"),
])
def test_uae_questions_find_the_uae_section(retriever, query, section):
    res = retriever.search(query)
    assert res.confident
    assert any(section.lower() in h.chunk.citation.lower() for h in res.hits)


def test_uae_docs_use_the_current_regulator_name():
    text = " ".join(c.content for c in load_chunks() if c.doc.startswith(("09_", "10_", "11_")))
    assert "Capital Market Authority (CMA)" in text
    assert "renamed on 1 January 2026" in text


def test_retail_protection_figures_match_the_margin_document():
    text = " ".join(c.content for c in load_chunks() if c.doc.startswith("12_"))
    for figure in ("1:30", "3.3%", "50%", "1:2"):
        assert figure in text
    assert "debit cards" in text and "credit card" in text


def test_sync_embeds_only_new_or_changed_sections_and_removes_stale_ones():
    store, emb = MemoryStore(), LexicalEmbedder()
    current = load_chunks()
    first = sync(store, emb)
    assert first == (len(current), 0)
    assert sync(store, emb) == (0, 0)  # nothing changed, nothing re-embedded

    # simulate an old deployment: one section edited, one section that no longer exists
    edited = current[0].__class__(**{**current[0].__dict__, "content": "old wording"})
    store.upsert([edited], emb.embed([edited.text_for_embedding]), emb.name)
    ghost = current[0].__class__(id="99_removed#0", doc="99_removed", title="Removed", section="Gone", content="gone")
    store.upsert([ghost], emb.embed([ghost.text_for_embedding]), emb.name)

    assert sync(store, emb) == (1, 1)
    ids = {c.id for c in store.load_chunks(emb.name)}
    assert "99_removed#0" not in ids and len(ids) == len(current)
