"""Config loading and CLI override tests.

Overrides are the easiest place in the project to introduce a silent bug: a
mistyped value does not crash, it just trains with the wrong hyperparameter.
``test_scientific_notation_becomes_a_float`` exists because PyYAML implements
YAML 1.1, which does not accept "3e-4" as a float -- so the natural
implementation quietly hands the trainer a string learning rate.
"""

import pytest
import yaml

from training.train import apply_overrides, coerce, load_config, set_deep

# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12", 12),
        ("-3", -3),
        ("0.5", 0.5),
        ("3.0e-4", 3.0e-4),
        ("true", True),
        ("false", False),
        ("checkpoints", "checkpoints"),
        ("sdpa", "sdpa"),
        ("bfloat16", "bfloat16"),
    ],
)
def test_coerce_matches_yaml_types(raw, expected):
    value = coerce(raw)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize("raw", ["3e-4", "1e-4", "1E5"])
def test_scientific_notation_becomes_a_float(raw):
    value = coerce(raw)
    assert isinstance(value, float), f"{raw!r} must be a float, got {type(value).__name__}"
    assert value == float(raw)


def test_override_type_matches_the_config_it_replaces():
    """An override must not change a key's type -- that is the whole risk."""
    cfg = yaml.safe_load("training:\n  lr: 3.0e-4\n  amp: true\n  steps: 2000\n")
    apply_overrides(cfg, ["training.lr=1e-4", "training.amp=false", "training.steps=10"])
    assert isinstance(cfg["training"]["lr"], float)
    assert isinstance(cfg["training"]["amp"], bool)
    assert isinstance(cfg["training"]["steps"], int)
    assert cfg["training"] == {"lr": 1e-4, "amp": False, "steps": 10}


# ---------------------------------------------------------------------------
# Dotted-key assignment
# ---------------------------------------------------------------------------


def test_set_deep_creates_missing_levels():
    cfg: dict = {}
    set_deep(cfg, "a.b.c", 1)
    assert cfg == {"a": {"b": {"c": 1}}}


def test_set_deep_overwrites_without_clobbering_siblings():
    cfg = {"model": {"d_model": 512, "n_layers": 8}}
    set_deep(cfg, "model.n_layers", 12)
    assert cfg == {"model": {"d_model": 512, "n_layers": 12}}


def test_override_without_equals_is_rejected():
    with pytest.raises(ValueError, match="dotted.key=value"):
        apply_overrides({}, ["training.lr"])


def test_empty_override_list_is_a_noop():
    cfg = {"seed": 42}
    assert apply_overrides(cfg, []) == {"seed": 42}


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seed: 7\nmodel:\n  d_model: 64\n", encoding="utf-8")
    assert load_config(str(path)) == {"seed": 7, "model": {"d_model": 64}}


def test_load_config_tolerates_an_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(str(path)) == {}


def test_shipped_default_config_has_every_key_the_code_reads():
    """Guards against the config and the code drifting apart."""
    cfg = load_config("configs/default.yaml")
    assert cfg["model"]["n_heads"] % cfg["model"]["n_kv_heads"] == 0
    assert cfg["model"]["d_model"] % cfg["model"]["n_heads"] == 0
    assert cfg["dataset"]["seq_len"] <= cfg["model"]["max_seq_len"]
    for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "vocab_size"):
        assert key in cfg["model"], f"configs/default.yaml is missing model.{key}"
    for key in ("steps", "lr", "warmup_steps", "out_dir", "amp_dtype"):
        assert key in cfg["training"], f"configs/default.yaml is missing training.{key}"
