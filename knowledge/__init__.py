"""
knowledge/ — Everything the agent knows and how it recalls it

Four sub-packages, in the order data flows through them:

    ingestion/    Raw text in.  clean -> chunk -> embed -> store.
    retrieval/    Text back out. Embeddings, the vector store, and the
                  semantic retriever that ranks what comes back.
    memory/       What survives across runs. episodic.py is the one in
                  active use; base/short_term/long_term are scaffolding
                  that nothing currently calls.
    reliability/  How much to trust a piece of evidence. Source tiers,
                  staleness decay, and conflict detection between sources.

The guiding idea (from Phase 2): evidence is not truth. Everything that
comes out of retrieval carries a reliability score and a staleness check,
and conflicts are reported rather than silently resolved.

Sub-packages are imported directly — `from knowledge.retrieval.retriever
import SemanticRetriever` — rather than re-exported here, so that importing
one part does not drag in chromadb and the embedding stack.
"""
