import json

from trust_bench.core.config import RunConfig


def test_run_config_defaults_to_no_constraints_and_linear_loss():
    config = RunConfig()

    assert config.max_iter is None
    assert config.tolerance is None
    assert config.bounds is None
    assert config.derivative_mode is None
    assert config.loss == "linear"
    assert config.x_scale is None


def test_run_config_round_trips_through_json_serialisation():
    # bounds is a JSON-native list-of-lists here, not a tuple: `from_dict`
    # follows the same trivial cls(**data) pattern as TimingStats and
    # EnvProvenance, with no field-specific reconstruction, so a
    # tuple-shaped bounds value would not survive the JSON hop.
    config = RunConfig(
        max_iter=100,
        tolerance=1e-6,
        bounds=[[0.0, 1.0]],
        derivative_mode="analytic",
        x_scale="jac",
    )

    restored = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
