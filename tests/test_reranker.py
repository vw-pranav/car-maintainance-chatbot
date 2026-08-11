import sys
import types
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sentence_transformers = types.ModuleType("sentence_transformers")


class CrossEncoder:
    def __init__(self, *args, **kwargs):
        pass


sentence_transformers.CrossEncoder = CrossEncoder
sys.modules["sentence_transformers"] = sentence_transformers

import reranker as reranker_module


class RerankerHeuristicTests(unittest.TestCase):
    def setUp(self):
        reranker_module.reranker = None

    def test_equipment_heading_and_phrase_rank_first(self):
        question = "What equipment is required?"
        docs = [
            SimpleNamespace(
                page_content="Diagnostic test only\nRead fault memory and clear codes.",
                metadata={"section_hints": ["Fault finding / diagnosis"]},
            ),
            SimpleNamespace(
                page_content="Special tools and workshop equipment required\nVAS 6931 Engine and Gearbox Jack\nT40257 Engine Support Bridge",
                metadata={"section_hints": ["Special tools and workshop equipment required"]},
            ),
            SimpleNamespace(
                page_content="1. Remove the cover.\n2. Disconnect the connector.",
                metadata={"section_hints": ["Removing and installing"]},
            ),
        ]

        ranked = reranker_module.rerank(question, docs, top_k=3)

        self.assertIs(ranked[0], docs[1])
        self.assertIs(ranked[1], docs[2])
        self.assertIs(ranked[2], docs[0])

    def test_diagnostic_only_chunk_is_penalized(self):
        question = "What equipment is required?"
        diagnostic_doc = SimpleNamespace(
            page_content="Diagnostic information\nRead fault codes using the tester.",
            metadata={"section_hints": ["Diagnostics"]},
        )
        specific_doc = SimpleNamespace(
            page_content="Special tools and workshop equipment required\nVAS 6095A Engine support fixture",
            metadata={"section_hints": ["Special tools and workshop equipment required"]},
        )

        ranked = reranker_module.rerank(question, [diagnostic_doc, specific_doc], top_k=2)

        self.assertIs(ranked[0], specific_doc)
        self.assertIs(ranked[1], diagnostic_doc)


if __name__ == "__main__":
    unittest.main()
