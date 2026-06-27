import unittest

from src.past_papers import extract_picture_number, matches_selected_paper, paper_name_from_filename


class PastPaperSelectionTests(unittest.TestCase):
    def test_paper_name_from_filename_strips_question_suffix(self):
        self.assertEqual(paper_name_from_filename("Girraween 2023_Picture 8.png"), "Girraween 2023")
        self.assertEqual(paper_name_from_filename("Girraween 2023 AT1_Picture 8.png"), "Girraween 2023 AT1")

    def test_matches_selected_paper_requires_exact_paper_label(self):
        self.assertTrue(matches_selected_paper("Girraween 2023_Picture 8.png", "Girraween 2023"))
        self.assertFalse(matches_selected_paper("Girraween 2023 AT1_Picture 8.png", "Girraween 2023"))

    def test_extract_picture_number_orders_questions(self):
        self.assertEqual(extract_picture_number("Girraween 2023_Picture 8.png"), 8)
        self.assertEqual(extract_picture_number("Girraween 2023_Group 12.png"), 12)


if __name__ == "__main__":
    unittest.main()
