"""Offline contract tests for the read-only MeG Phi-2 source preflight."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "meg_phi2_source_preflight.py"
spec = importlib.util.spec_from_file_location("meg_preflight", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)
        subprocess.check_call(["git", "-C", str(self.source), "init", "-q"])
        subprocess.check_call(["git", "-C", str(self.source), "config", "user.email", "audit@example.invalid"])
        subprocess.check_call(["git", "-C", str(self.source), "config", "user.name", "Audit Test"])
        for rel, tokens in module.REQUIRED.items():
            path = self.source / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(self.source), "add", "."])
        subprocess.check_call(["git", "-C", str(self.source), "commit", "-qm", "synthetic fixture"])
        self.commit = subprocess.check_output(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True
        ).strip()

    def test_reports_absent_artifacts_without_claiming_reproduction(self):
        report = module.audit(self.source, expected_commit=self.commit)
        self.assertEqual(report["source_integrity"], "PASS")
        self.assertEqual(report["state"], "BLOCKED_MISSING_RESEARCH_ASSETS")
        self.assertFalse(report["training_or_evaluation_executed"])
        self.assertEqual(
            set(report["derived_artifacts_missing"]), set(module.GENERATED_OR_EXTERNAL)
        )

    def test_commit_mismatch_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "upstream commit mismatch"):
            module.audit(self.source)

    def test_missing_required_file_is_fatal(self):
        (self.source / "train.py").unlink()
        with self.assertRaises(FileNotFoundError):
            module.audit(self.source, expected_commit=self.commit)

if __name__ == "__main__":
    unittest.main()
