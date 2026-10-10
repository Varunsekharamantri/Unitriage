"""The severity rules: turns what the model SAW into minor / major / emergency.

Severity here means URGENCY, and is separate from whether a tenant can fix it
(is_safe_for_diy). A ceiling stain is low urgency but not a DIY job: it goes to a
technician without the alarm. Edit this table to change how cautious the system is.

    emergency  immediate danger to people or property
    major      needs a professional soon (within about a day)
    minor      everything else
"""
from typing import List, Tuple

from app.core.schemas import Evidence, Severity


def assess(e: Evidence) -> Tuple[Severity, bool, List[str]]:
    """Returns (severity, is_safe_for_diy, reasons)."""
    emergency: List[str] = []
    if e.electrical_hazard:
        emergency.append("exposed or sparking electrics, or water near electrics")
    if e.gas_or_fire:
        emergency.append("fire, smoke or a damaged gas fitting")
    if e.structural_risk:
        emergency.append("structural risk (sagging ceiling or wall, large cracks)")
    if e.water == "flooding":
        emergency.append("flooding: water spreading over a floor or large area")
    if e.sewage and e.water in ("steady_flow_or_spray", "flooding"):
        emergency.append("sewage flowing or overflowing")
    if emergency:
        return "emergency", False, emergency

    major: List[str] = []
    if e.water == "steady_flow_or_spray":
        major.append("water flowing or spraying continuously")
    if e.water == "dripping" and not e.simple_fix:
        # A dripping tap is a DIY job. A drip from a wall, ceiling or pipe is an active leak:
        # the model often calls a thin stream "dripping", so don't treat it as low urgency.
        major.append("water dripping from the building itself (not a simple fixture fix)")
    if e.damage == "extensive":
        major.append("extensive damage")
    if e.sewage and e.water != "contained_in_fixture":
        # Dirty water that stays inside a sink is a blockage, not a sewage escape.
        major.append("sewage present")
    if major:
        return "major", False, major

    # Water that stays inside a sink, bath or toilet (a blocked sink, a running toilet) is a
    # nuisance, not a leak: "contained_in_fixture" is treated like no water at all.
    reasons = ["no hazard visible and no continuous water flow"]
    diy = e.simple_fix and e.damage in ("none", "cosmetic") and not e.sewage
    if not diy:
        reasons.append("not a simple fix, so a technician should look at it")
    return "minor", diy, reasons
