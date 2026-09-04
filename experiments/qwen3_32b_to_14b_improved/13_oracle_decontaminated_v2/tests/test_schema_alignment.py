from __future__ import annotations

from src.contracts import EXPECTED_ANSWER_ACCESS_POLICY, ROOT, read_json


FROZEN_DIR12_G0_CONTRACT = (
    ROOT.parent / "12_prospective_screen_v2" / "contracts" / "g0_consumer_contract.json"
)


def test_dir13_g0_contract_matches_frozen_dir12_09_machine_schema() -> None:
    ours = read_json(ROOT / "g0_consumer_contract.json")
    frozen = read_json(FROZEN_DIR12_G0_CONTRACT)
    assert ours["contract_schema_version"] == frozen["contract_schema_version"] == 1
    assert ours["required_top_level_values"] == frozen["required_top_level_values"]
    assert ours["candidate_registry_binding"] == frozen["candidate_registry_binding"]
    assert ours["comparisons"] == frozen["comparisons"]
    assert ours["fail_closed"] is frozen["fail_closed"] is True


def test_native_contract_template_and_code_freeze_one_exact_answer_policy() -> None:
    native = read_json(ROOT / "native_registry_consumer_contract.json")
    template = read_json(ROOT / "config.template.json")
    assert native["contract_id"] == "dir10_native_registry_train_oracle_v2"
    assert native["contract_revision"] == 2
    requirement = native["receipt_requirements"]["answer_access_policy"]
    assert requirement["exact_keys"] == list(EXPECTED_ANSWER_ACCESS_POLICY)
    assert requirement["exact_typed_value"] == EXPECTED_ANSWER_ACCESS_POLICY
    assert requirement["additional_keys_allowed"] is False
    assert requirement["typed_values_must_match_exactly"] is True
    assert requirement["formal_config_must_embed_exact_typed_value"] is True
    assert (
        template["native_registry"]["required_answer_access_policy"]
        == EXPECTED_ANSWER_ACCESS_POLICY
    )
