import json

from trust_bench.core.provenance import EnvProvenance, capture


def test_capture_returns_a_populated_env_provenance():
    provenance = capture()
    assert provenance.backend_name
    assert provenance.backend_version
    assert provenance.language_runtime
    assert provenance.blas_lapack
    assert provenance.os
    assert provenance.cpu_model
    assert provenance.cpu_count > 0
    assert provenance.machine_fingerprint


def test_env_provenance_round_trips_through_json_serialisation():
    provenance = capture()
    restored = EnvProvenance.from_dict(json.loads(json.dumps(provenance.to_dict())))
    assert restored == provenance


def test_machine_fingerprint_is_stable_across_calls():
    assert capture().machine_fingerprint == capture().machine_fingerprint
