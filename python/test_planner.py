import unittest

from asl.planner import plan_asl


def unit_by_gloss(plan, gloss):
    return next(unit for unit in plan["units"] if unit.get("gloss") == gloss)


class PlannerHandPatternTests(unittest.TestCase):
    def test_symmetrical_two_handed_signs(self):
        plan = plan_asl("work school sign")
        for gloss in ("WORK", "SCHOOL", "SIGN"):
            unit = unit_by_gloss(plan, gloss)
            self.assertEqual(unit["hands"]["pattern"], "symmetrical")
            self.assertEqual(unit["hands"]["active"], ["dominant", "non_dominant"])

        self.assertTrue(unit_by_gloss(plan, "SIGN")["hands"]["alternating"])

    def test_asymmetrical_two_handed_signs(self):
        plan = plan_asl("help name learn")
        for gloss in ("HELP", "NAME", "LEARN"):
            unit = unit_by_gloss(plan, gloss)
            self.assertEqual(unit["hands"]["pattern"], "asymmetrical")
            self.assertEqual(unit["hands"]["active"], ["dominant", "non_dominant"])
            self.assertIn(unit["hands"]["nonDominant"]["role"], {"support", "target"})

    def test_fingerspelling_uses_non_dominant_reference_hand(self):
        plan = plan_asl("codex")
        unit = plan["units"][0]
        self.assertEqual(unit["type"], "fingerspell")
        self.assertEqual(unit["hands"]["pattern"], "asymmetrical")
        self.assertEqual(unit["hands"]["dominant"]["role"], "fingerspell")
        self.assertEqual(unit["hands"]["nonDominant"]["role"], "reference")

    def test_common_sentence_uses_signs_instead_of_fingerspelling(self):
        plan = plan_asl("testing how are you doing today")
        glosses = [unit.get("gloss") for unit in plan["units"]]
        self.assertEqual(glosses, ["TEST", "HOW", "YOU", "DO", "TODAY"])
        self.assertTrue(all(unit["type"] == "sign" for unit in plan["units"]))
        self.assertEqual(unit_by_gloss(plan, "HOW")["hands"]["pattern"], "symmetrical")
        self.assertEqual(unit_by_gloss(plan, "DO")["hands"]["pattern"], "symmetrical")
        self.assertEqual(unit_by_gloss(plan, "TODAY")["hands"]["pattern"], "symmetrical")


if __name__ == "__main__":
    unittest.main()
