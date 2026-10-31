import unittest
from unittest.mock import patch

from app.tools.literature_synthesis import synthesize_literature
from app.tools.paper_comparison import compare_papers
from app.tools.results_extraction import extract_paper_results
from app.tools.summarisation import summarize_paper


SAMPLE_CHUNK = {
    "text": "The model reaches 92.4% accuracy on CIFAR-10.",
    "paper_id": 1,
    "project_id": 1,
    "arxiv_id": "1234.5678",
    "page_number": 4,
    "score": 0.9,
}


class ToolWorkflowTests(unittest.TestCase):
    def test_summarize_returns_error_without_chunks(self):
        with patch(
            "app.tools.summarisation.retrieve_paper_chunks",
            return_value=[],
        ):
            result = summarize_paper(paper_id=1, project_id=1)

        self.assertFalse(result["grounded"])
        self.assertIsNone(result["summary"])
        self.assertIn("error", result)

    def test_summarize_returns_grounded_summary(self):
        with patch(
            "app.tools.summarisation.retrieve_paper_chunks",
            return_value=[SAMPLE_CHUNK],
        ), patch(
            "app.tools.summarisation.generate_from_excerpts",
            return_value="1. Problem and Motivation\nA classification task.",
        ):
            result = summarize_paper(paper_id=1, project_id=1)

        self.assertTrue(result["grounded"])
        self.assertIn("Problem and Motivation", result["summary"])
        self.assertEqual(result["sources"][0]["page_number"], 4)

    def test_compare_requires_two_papers(self):
        result = compare_papers(paper_ids=[1], project_id=1)

        self.assertFalse(result["grounded"])
        self.assertIn("two papers", result["error"])

    def test_compare_fails_closed_if_one_paper_has_no_evidence(self):
        def fake_gather(paper_ids, project_id, question, top_k):
            return {
                "contexts": [],
                "chunks": [],
                "error": "No relevant content was found for paper ID 2. Ensure the paper has been indexed.",
            }

        with patch(
            "app.tools.paper_comparison.gather_multi_paper_evidence",
            side_effect=fake_gather,
        ):
            result = compare_papers(paper_ids=[1, 2], project_id=1)

        self.assertFalse(result["grounded"])
        self.assertIn("paper ID 2", result["error"])

    def test_extract_results_includes_focus(self):
        with patch(
            "app.tools.results_extraction.retrieve_paper_chunks",
            return_value=[SAMPLE_CHUNK],
        ), patch(
            "app.tools.results_extraction.generate_from_excerpts",
            return_value="4. Main Quantitative Results\n92.4% on CIFAR-10 [Page 4]",
        ):
            result = extract_paper_results(
                paper_id=1,
                project_id=1,
                focus="CIFAR-10",
            )

        self.assertTrue(result["grounded"])
        self.assertEqual(result["focus"], "CIFAR-10")
        self.assertIn("92.4%", result["results"])

    def test_synthesis_requires_topic(self):
        result = synthesize_literature(
            paper_ids=[1, 2],
            project_id=1,
            topic="   ",
        )

        self.assertFalse(result["grounded"])
        self.assertIn("topic", result["error"].lower())

    def test_synthesis_returns_grounded_output(self):
        with patch(
            "app.tools.literature_synthesis.gather_multi_paper_evidence",
            return_value={
                "contexts": ["=== PAPER ID: 1 ===", "=== PAPER ID: 2 ==="],
                "chunks": [
                    SAMPLE_CHUNK,
                    {**SAMPLE_CHUNK, "paper_id": 2, "page_number": 7},
                ],
                "error": None,
            },
        ), patch(
            "app.tools.literature_synthesis.generate_from_excerpts",
            return_value="2. Points of Agreement\nBoth use CIFAR-10.",
        ):
            result = synthesize_literature(
                paper_ids=[1, 2],
                project_id=1,
                topic="image classification",
            )

        self.assertTrue(result["grounded"])
        self.assertEqual(result["topic"], "image classification")
        self.assertEqual(len(result["sources"]), 2)


if __name__ == "__main__":
    unittest.main()
