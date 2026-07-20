"""Consistency checks on the stat registry itself, so adding a new
entry with a missing/misspelled key fails a test instead of blowing up
at runtime for whoever first plots that stat."""

import pytest

from mlb_stats.stats import STAT_CONFIGS, get_stat_config


def test_every_config_is_complete() -> None:
    for key, config in STAT_CONFIGS.items():
        assert config["label"], key
        assert config["group"] in ("pitching", "batting", "team"), key
        assert "cumulative_field" in config, key

        if "composite_of" in config:
            # Composite stats delegate entirely to their components.
            for component in config["composite_of"]:
                assert component in STAT_CONFIGS, f"{key} references unknown stat {component}"
        elif "computation" in config:
            # Bespoke-computation stats bypass the rate machinery; the
            # marker must be one the plotting code actually handles.
            assert config["computation"] == "war_approx", key
        else:
            assert config["numerator_fields"], key
            assert config["denominator_fields"], key
            assert config["multiplier"], key


def test_composite_components_are_not_themselves_composite() -> None:
    # The plotting code supports recursion, but the registry currently
    # promises single-level composites; flag it if that changes silently.
    for key, config in STAT_CONFIGS.items():
        for component in config.get("composite_of", []):
            assert "composite_of" not in STAT_CONFIGS[component], (
                f"{key} -> {component} is a composite of a composite"
            )


def test_get_stat_config_unknown_stat_raises_with_choices() -> None:
    with pytest.raises(ValueError, match="Unknown stat 'nope'"):
        get_stat_config("nope")
