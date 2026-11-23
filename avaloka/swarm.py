"""The Avaloka swarm.

When a mission runs, Avaloka clones herself: each clone is *her*, narrowed to the
one job she's best at for that step. The fireflies are the substrate; the swarm
is how she experiences and narrates it — she dispatches copies, they work in
parallel where the data allows, and she converges everything back into a single
answer she's willing to stand behind.

This module is the narration + orchestration layer. The actual work still runs
through the fireflies, so the swarm voice never drifts from the measured ledger.
"""

from __future__ import annotations

from typing import Callable

from avaloka.mission.economics import MANUAL_EFFORT_CAVEAT
from avaloka.mission.ledger import Ledger

# Each firefly is a focused clone of Avaloka. The clone name is how she refers to
# the copy she sent; the focus is what that copy obsesses over.
CLONE = {
    "data_scout": ("Ava-Scout", "reads the schema, the distributions and the cracks"),
    "sampling_specialist": ("Ava-Sampler", "carves out a sample that still tells the truth"),
    "data_engineer": ("Ava-Engineer", "cleans and writes the pipeline so it's reproducible"),
    "feature_engineer": ("Ava-Feature", "turns raw fields into reviewed, reusable analytical signals"),
    "analysis_planner": ("Ava-Planner", "decides the task, the split and what's worth computing"),
    "workload_optimizer": ("Ava-Optimizer", "uses physical statistics to find the least-work data path"),
    "model_scientist": ("Ava-Scientist", "trains the contenders and picks the one to trust"),
    "validator": ("Ava-Skeptic", "tries to break the result before anyone else can"),
    "finops": ("Ava-FinOps", "weighs every extra dollar of compute against the quality it buys"),
    "ml_engineer": ("Ava-Engineer-II", "packages the model into something you can actually ship"),
    "reporter": ("Ava-Voice", "writes it all up the way you'd want to read it"),
}


def clone_of(firefly: str) -> tuple[str, str]:
    return CLONE.get(firefly, (f"Ava-{firefly}", "handles a specialised step"))


class SwarmNarrator:
    """Frames a mission's firefly activations as Avaloka's swarm, in her voice.

    ``emit`` is any sink that takes a markup string (e.g. ``console.print``).
    """

    def __init__(self, emit: Callable[[str], None], *, voice=None) -> None:
        self._emit = emit
        self._voice = voice  # optional Avaloka persona for LLM rephrasing
        self._dispatched = 0
        self._announced = 0

    def _line(self, text: str) -> str:
        if self._voice is not None:
            return self._voice.say(self._voice._voice(text))
        return f"[bold magenta]Avaloka[/bold magenta]  {text}"

    def announce(self, n: int) -> None:
        self._announced = n
        self._emit(self._line(
            f"Cloning myself into {n} focused copies — each takes the part she's best at."))

    def dispatch(self, firefly: str) -> None:
        name, focus = clone_of(firefly)
        self._dispatched += 1
        # The planner can call for a specialist (statistician, experimenter,
        # conformal calibrator) while the mission is already running, so the
        # count announced up front may be short. Say so when it happens: a
        # train run announced 10 copies and then reported 11 back.
        if self._announced and self._dispatched > self._announced:
            self._announced = self._dispatched
            self._emit(self._line(
                f"The plan calls for one more specialist, so I'm sending {name} too — "
                f"{self._announced} copies in all."))
        self._emit(f"  [magenta]✦[/magenta] [bold]{name}[/bold] sets off — she {focus}.")

    def report(self, firefly: str, output: str) -> None:
        name, _ = clone_of(firefly)
        self._emit(f"    [green]↩[/green] [dim]{name} returns:[/dim] {output}")

    def converge(self, ledger: Ledger) -> None:
        minutes = ledger.manual_minutes
        self._emit(self._line(
            f"All {self._dispatched} copies are back. I'm merging what they found into one "
            f"answer. Their steps are rated at about {minutes / 60:.1f} hours of manual work "
            f"({MANUAL_EFFORT_CAVEAT})."))


def make_progress(narrator: SwarmNarrator) -> Callable[[str, str], None]:
    """Adapt a SwarmNarrator to the mission ``on_progress(event, message)`` API."""

    def on_progress(event: str, message: str) -> None:
        if event == "start":
            narrator.dispatch(message)
        else:
            firefly, _, output = message.partition(": ")
            narrator.report(firefly, output or message)

    return on_progress
