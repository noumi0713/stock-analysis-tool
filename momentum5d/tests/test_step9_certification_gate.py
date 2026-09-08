"""No market-data computation is performed by these provenance tests."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/step9_certification_gate.py"
SPEC = importlib.util.spec_from_file_location("gate", SOURCE)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class CertificationTests(unittest.TestCase):
    def test_missing_inputs_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = GATE.inspect_inputs(Path(directory), True)
        self.assertEqual(result["quality"], "FAIL")
        self.assertIn("certified_step1_parquet_missing", result["errors"])
        self.assertFalse(result["analysis_executed"])

    def test_skipped_reproducibility_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            result = GATE.inspect_inputs(Path(directory), False)
        self.assertIn("two_independent_runs_not_requested", result["errors"])
        self.assertIsNone(result["reproducibility_passed"])

    def test_forged_pass_report_does_not_certify_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "quality").mkdir()
            for name in ("step1", "step2", "step7_v2", "step8_v2"):
                (root / "quality" / f"{name}_report.json").write_text(
                    json.dumps({"quality": "PASS"}), encoding="utf-8")
            result = GATE.inspect_inputs(root, True)
        self.assertEqual(result["quality"], "FAIL")
        self.assertIn("step7_report_bytes_mismatch", result["errors"])

    def test_target_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "targets/equity_daily_forward_targets/part.parquet"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"not certified market data")
            result = GATE.inspect_inputs(root, True)
        self.assertIn("step2_bytes_do_not_match_pinned_artifact", result["errors"])


if __name__ == "__main__":
    unittest.main()
