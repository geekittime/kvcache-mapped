from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "improved_protocol_validator", ROOT / "validate_protocol.py"
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


class ProtocolValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
        self.data_policy = json.loads((ROOT / "data_policy.json").read_text(encoding="utf-8"))
        self.refs = json.loads((ROOT / "baseline_refs.json").read_text(encoding="utf-8"))

    def test_full_read_only_validation_passes(self) -> None:
        self.assertEqual(validator.validate_protocol(ROOT), [])

    def test_every_baseline_digest_is_checked_and_tamper_detected(self) -> None:
        repo_root = validator.repository_root(self.refs, ROOT)
        baseline_root = (repo_root / self.refs["baseline_root"]).resolve()
        for reference in self.refs["references"]:
            with self.subTest(reference=reference["id"]):
                self.assertEqual(
                    validator.verify_reference(reference, repo_root, baseline_root), []
                )

        tampered = copy.deepcopy(self.refs["references"][0])
        tampered["sha256"] = "0" * 64
        errors = validator.verify_reference(tampered, repo_root, baseline_root)
        self.assertTrue(any("SHA-256 mismatch" in error for error in errors))

    def test_baseline_references_cannot_escape_existing_experiment(self) -> None:
        repo_root = validator.repository_root(self.refs, ROOT)
        baseline_root = (repo_root / self.refs["baseline_root"]).resolve()
        escaped = copy.deepcopy(self.refs["references"][0])
        escaped["path"] = "README.md"
        errors = validator.verify_reference(escaped, repo_root, baseline_root)
        self.assertTrue(any("escapes read-only baseline root" in error for error in errors))

    def test_data_tiers_are_distinct_ordered_and_test_is_sealed(self) -> None:
        tiers = self.data_policy["tiers"]
        self.assertEqual([tier["id"] for tier in tiers], ["mapper-fit", "method-dev", "locked-test"])
        self.assertEqual(len({tier["namespace"] for tier in tiers}), 3)
        self.assertTrue(self.data_policy["token_level_random_split_forbidden"])
        self.assertFalse(self.data_policy["cross_tier_exact_id_overlap_allowed"])
        self.assertEqual(tiers[-1]["state"], "sealed_pending_candidate_freeze")
        self.assertIn("training", tiers[-1]["forbidden_actions"])

    def test_prior_test_exposure_is_not_misrepresented_as_blind(self) -> None:
        exposure = self.data_policy["historical_exposure"]
        self.assertTrue(exposure["aggregate_results_previously_observed"])
        self.assertIn("not historically blind", exposure["required_report_language"])

    def test_ablation_order_blocks_early_locked_test_access(self) -> None:
        candidates = self.protocol["candidates"]
        self.assertEqual([item["order"] for item in candidates], list(range(len(candidates))))
        for candidate in candidates[:9]:
            self.assertNotIn("locked-test", candidate["data_access"])
            self.assertFalse(candidate["locked_test_allowed"])
        self.assertEqual(candidates[9]["id"], "A9")

    def test_pure_and_hybrid_claims_are_separate(self) -> None:
        modes = self.protocol["system_classes"]
        self.assertEqual([mode["target_replay_tokens"] for mode in modes], [1, 8, 32])
        self.assertNotEqual(modes[0]["claim_class"], modes[1]["claim_class"])
        self.assertFalse(self.protocol["claim_policy"]["pure_and_hybrid_results_may_be_merged"])

    def test_primary_method_is_not_prematurely_claimed(self) -> None:
        self.assertEqual(self.protocol["primary_method"]["status"], "unvalidated_hypothesis")
        self.assertFalse(self.protocol["claim_policy"]["paper_bit_exact_claim_allowed"])

    def test_protocol_python_is_standard_library_only(self) -> None:
        for path in [ROOT / "validate_protocol.py", ROOT / "tests" / "test_protocol.py"]:
            with self.subTest(path=path.name):
                self.assertEqual(validator.imported_roots(path) - validator.ALLOWED_IMPORT_ROOTS, set())

    def test_json_pointer_handles_lists_and_escapes(self) -> None:
        document = {"a/b": [{"~key": 7}]}
        self.assertEqual(validator.json_pointer(document, "/a~1b/0/~0key"), 7)


if __name__ == "__main__":
    unittest.main()
