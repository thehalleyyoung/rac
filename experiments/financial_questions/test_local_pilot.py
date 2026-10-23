from pathlib import Path
import tempfile
import unittest

from case_manifest import parse_json
from local_pilot import run_pilot


class LocalPilotTests(unittest.TestCase):
    def test_composed_workflow_blind_binding_and_control_negative_reach_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            result = run_pilot(out)
            self.assertTrue(result["protocol_ready"])
            self.assertEqual((result["answers"], result["judge_records"]), (6, 12))
            report = parse_json((out / "workflow-report.json").read_text())
            self.assertEqual(len(report["steps"]), 6)
            self.assertFalse(parse_json((out / "flat-high-negative.json").read_text())["passed"])
            self.assertTrue(parse_json((out / "control-report.json").read_text())["passed"])
            request = parse_json(next(out.glob("scripted-good-step3-*/request.json")).read_text())
            self.assertEqual(len(request["payload"]["prior_context"]), 4)
            with self.assertRaises(FileExistsError):
                run_pilot(out)


if __name__ == "__main__":
    unittest.main()
