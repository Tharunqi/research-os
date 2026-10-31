import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.tools.search_and_save import search_and_save_papers


def fake_paper(**overrides):
    data = {
        "id": 1,
        "arxiv_id": "1234.5678",
        "title": "A Test Paper",
        "abstract": "An abstract.",
        "authors": ["Ada"],
        "published": None,
        "paper_url": "https://arxiv.org/abs/1234.5678",
        "pdf_url": "https://arxiv.org/pdf/1234.5678",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class SearchAndSaveIngestionTests(unittest.TestCase):
    def test_ingests_new_paper(self):
        paper = fake_paper()

        with patch("app.tools.search_and_save.search_papers", return_value=[{}]), patch(
            "app.tools.search_and_save.save_papers", return_value=[paper]
        ), patch(
            "app.tools.search_and_save.is_paper_indexed", side_effect=[False, True]
        ), patch(
            "app.tools.search_and_save.ingest_pdf"
        ) as ingest:
            results = search_and_save_papers(project_id=1, query="1234.5678")

        ingest.assert_called_once()
        self.assertEqual(results[0]["ingestion_status"], "indexed")
        self.assertIsNone(results[0]["ingestion_error"])

    def test_skips_already_indexed_paper(self):
        paper = fake_paper()

        with patch("app.tools.search_and_save.search_papers", return_value=[{}]), patch(
            "app.tools.search_and_save.save_papers", return_value=[paper]
        ), patch("app.tools.search_and_save.is_paper_indexed", return_value=True), patch(
            "app.tools.search_and_save.ingest_pdf"
        ) as ingest:
            results = search_and_save_papers(project_id=1, query="test query")

        ingest.assert_not_called()
        self.assertEqual(results[0]["ingestion_status"], "already_indexed")

    def test_records_missing_pdf_url(self):
        paper = fake_paper(pdf_url=None)

        with patch("app.tools.search_and_save.search_papers", return_value=[{}]), patch(
            "app.tools.search_and_save.save_papers", return_value=[paper]
        ), patch("app.tools.search_and_save.ingest_pdf") as ingest:
            results = search_and_save_papers(project_id=1, query="test query")

        ingest.assert_not_called()
        self.assertEqual(results[0]["ingestion_status"], "no_pdf_url")

    def test_records_failed_ingestion(self):
        paper = fake_paper()

        with patch("app.tools.search_and_save.search_papers", return_value=[{}]), patch(
            "app.tools.search_and_save.save_papers", return_value=[paper]
        ), patch("app.tools.search_and_save.is_paper_indexed", return_value=False), patch(
            "app.tools.search_and_save.ingest_pdf", side_effect=RuntimeError("download failed")
        ):
            results = search_and_save_papers(project_id=1, query="test query")

        self.assertEqual(results[0]["ingestion_status"], "failed")
        self.assertIn("download failed", results[0]["ingestion_error"])

    def test_one_failure_does_not_drop_other_papers(self):
        ready = fake_paper(id=1, arxiv_id="1111.1111", title="Ready")
        broken = fake_paper(
            id=2,
            arxiv_id="2222.2222",
            title="Broken",
            pdf_url=None,
        )

        with patch("app.tools.search_and_save.search_papers", return_value=[{}, {}]), patch(
            "app.tools.search_and_save.save_papers", return_value=[ready, broken]
        ), patch(
            "app.tools.search_and_save.is_paper_indexed", side_effect=[False, True]
        ), patch(
            "app.tools.search_and_save.ingest_pdf"
        ):
            results = search_and_save_papers(project_id=1, query="attention")

        self.assertEqual(results[0]["ingestion_status"], "indexed")
        self.assertEqual(results[1]["ingestion_status"], "no_pdf_url")


if __name__ == "__main__":
    unittest.main()
