"""The severity rules: what the model SAW in, minor / major / emergency out."""
import itertools

import pytest

from app.core.policy import assess
from app.core.schemas import Evidence

WATER = ["none", "stain_or_damp", "dripping", "steady_flow_or_spray", "flooding", "contained_in_fixture"]
DAMAGE = ["none", "cosmetic", "localized", "extensive"]


def ev(**kw) -> Evidence:
    base = dict(
        water="none", damage="none", electrical_hazard=False, gas_or_fire=False,
        structural_risk=False, sewage=False, simple_fix=False,
    )
    return Evidence(**{**base, **kw})


@pytest.mark.parametrize(
    "name, evidence, severity, diy",
    [
        ("slow drip from a tap", ev(water="dripping", simple_fix=True), "minor", True),
        ("sink to unclog", ev(simple_fix=True), "minor", True),
        # water that stays inside a fixture is a nuisance, not a leak or a flood
        ("blocked sink, full of dirty water", ev(water="contained_in_fixture", simple_fix=True), "minor", True),
        ("running toilet", ev(water="contained_in_fixture", simple_fix=True), "minor", True),
        ("fixture problem the model thinks is hard", ev(water="contained_in_fixture"), "minor", False),
        # ...but if the sink is overflowing onto the floor, that is flooding
        ("sink overflowing across the floor", ev(water="flooding", simple_fix=True), "emergency", False),
        ("a stream down a wall", ev(water="steady_flow_or_spray", damage="localized"), "major", False),
        # a drip from a tap is a DIY job; a drip from a wall, ceiling or pipe is an active leak
        ("water dripping out of a wall crack", ev(water="dripping", damage="localized"), "major", False),
        ("water dripping from a ceiling", ev(water="dripping"), "major", False),
        ("a stain with no moving water stays low urgency", ev(water="stain_or_damp", damage="localized"), "minor", False),
        ("scuffed paint", ev(damage="cosmetic", simple_fix=True), "minor", True),
        # urgency and DIY are separate: low urgency, but not a job for the tenant
        ("old stain on a ceiling", ev(water="stain_or_damp", damage="localized"), "minor", False),
        ("stain but the model thinks it's easy", ev(water="stain_or_damp", damage="localized", simple_fix=True), "minor", False),
        ("small spray from a garden pipe", ev(water="steady_flow_or_spray"), "major", False),
        ("heavy mould and damage", ev(damage="extensive"), "major", False),
        ("sewage present but contained", ev(sewage=True, water="stain_or_damp"), "major", False),
        ("sewage flowing", ev(sewage=True, water="steady_flow_or_spray"), "emergency", False),
        ("flooded floor", ev(water="flooding"), "emergency", False),
        ("sparking socket", ev(electrical_hazard=True), "emergency", False),
        ("smoke or a damaged gas fitting", ev(gas_or_fire=True), "emergency", False),
        ("sagging ceiling", ev(structural_risk=True), "emergency", False),
    ],
)
def test_known_situations(name, evidence, severity, diy):
    got_severity, got_diy, reasons = assess(evidence)
    assert (got_severity, got_diy) == (severity, diy), name
    assert reasons, "every decision has a reason a person can read"


def test_a_hazard_always_wins_over_everything_else():
    # even a model that says "simple fix, cosmetic" can't hide a visible hazard
    for hazard in ("electrical_hazard", "gas_or_fire", "structural_risk"):
        assert assess(ev(**{hazard: True}, simple_fix=True, damage="cosmetic"))[0] == "emergency"


def test_a_small_leak_is_not_an_emergency():
    """The reported bug: a thin spray from a pipe was being called an emergency."""
    for water in ("stain_or_damp", "dripping", "steady_flow_or_spray"):
        assert assess(ev(water=water))[0] != "emergency"


def test_invariants_hold_for_every_combination():
    flags = list(itertools.product([False, True], repeat=5))  # 4 hazards + simple_fix
    for water, damage, (elec, gas, struct, sewage, simple) in itertools.product(WATER, DAMAGE, flags):
        e = ev(
            water=water, damage=damage, electrical_hazard=elec, gas_or_fire=gas,
            structural_risk=struct, sewage=sewage, simple_fix=simple,
        )
        severity, diy, reasons = assess(e)
        assert reasons
        if elec or gas or struct or water == "flooding":
            assert severity == "emergency", e
        if diy:  # DIY is only ever offered for a plain, low-urgency, simple job
            assert severity == "minor" and simple and damage in ("none", "cosmetic")
            assert not (elec or gas or struct or sewage), e
        if severity == "emergency":
            assert not diy
        if water == "contained_in_fixture" and not (elec or gas or struct or sewage) and damage != "extensive":
            assert severity == "minor", e  # water inside a sink or toilet never escalates by itself
