"""Curriculum scheduler for GRubrics-Transfer training.

Manages the progressive shift from verifiable-heavy to open-heavy data:
  Phase 1: 80% verifiable, 20% open  (rubric format + basic discrimination)
  Phase 2: 50% verifiable, 50% open  (transition to open domain)
  Phase 3: 20% verifiable, 80% open  (specialisation on target domain)

Each phase uses a pre-generated parquet file. The scheduler tells the
training orchestrator when to switch data files and optionally adjusts
learning rate between phases.

Usage:
    scheduler = CurriculumScheduler(
        total_steps=2000,
        phases=[
            CurriculumPhase(verif_ratio=0.8, open_ratio=0.2, fraction=0.4),
            CurriculumPhase(verif_ratio=0.5, open_ratio=0.5, fraction=0.3),
            CurriculumPhase(verif_ratio=0.2, open_ratio=0.8, fraction=0.3),
        ],
    )

    phase_idx = scheduler.get_phase(step=500)
    data_file = scheduler.get_data_file(step=500)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class CurriculumPhase:
    """A single phase of the curriculum.

    Attributes:
        verif_ratio: Fraction of verifiable data in this phase (0-1).
        open_ratio: Fraction of open-domain data in this phase (0-1).
        fraction: What fraction of total training steps this phase uses.
        lr_scale: Optional learning rate multiplier for this phase.
        data_file: Path to the parquet file for this phase (set by scheduler).
    """
    verif_ratio: float
    open_ratio: float
    fraction: float
    lr_scale: float = 1.0
    data_file: str = ""


# Default 3-phase curriculum
DEFAULT_PHASES = [
    CurriculumPhase(verif_ratio=0.8, open_ratio=0.2, fraction=0.4),
    CurriculumPhase(verif_ratio=0.5, open_ratio=0.5, fraction=0.3),
    CurriculumPhase(verif_ratio=0.2, open_ratio=0.8, fraction=0.3),
]


@dataclass
class CurriculumScheduler:
    """Manages curriculum phase transitions during training.

    Attributes:
        total_steps: Total training steps across all phases.
        phases: List of CurriculumPhase objects.
        data_dir: Directory containing phase parquet files.
    """
    total_steps: int
    phases: List[CurriculumPhase] = field(default_factory=lambda: list(DEFAULT_PHASES))
    data_dir: str = "data/processed/curriculum"

    def __post_init__(self):
        # Normalise fractions to sum to 1
        total_frac = sum(p.fraction for p in self.phases)
        if abs(total_frac - 1.0) > 0.01:
            for p in self.phases:
                p.fraction /= total_frac

        # Compute step boundaries
        self._boundaries: List[int] = []
        cumulative = 0
        for p in self.phases:
            cumulative += int(self.total_steps * p.fraction)
            self._boundaries.append(cumulative)
        # Ensure last boundary covers all steps
        self._boundaries[-1] = self.total_steps

        # Set default data file paths if not already set
        for i, p in enumerate(self.phases):
            if not p.data_file:
                p.data_file = str(
                    Path(self.data_dir) / f"mixed_curriculum_phase{i + 1}.parquet"
                )

    def get_phase_index(self, step: int) -> int:
        """Return the 0-based phase index for the given training step."""
        for i, boundary in enumerate(self._boundaries):
            if step < boundary:
                return i
        return len(self.phases) - 1

    def get_phase(self, step: int) -> CurriculumPhase:
        """Return the CurriculumPhase for the given training step."""
        return self.phases[self.get_phase_index(step)]

    def get_data_file(self, step: int) -> str:
        """Return the parquet data file path for the given training step."""
        return self.get_phase(step).data_file

    def get_lr_scale(self, step: int) -> float:
        """Return the learning rate scale for the given training step."""
        return self.get_phase(step).lr_scale

    def get_phase_boundaries(self) -> List[int]:
        """Return list of step boundaries (exclusive end of each phase)."""
        return list(self._boundaries)

    def get_ratios(self) -> List[Tuple[float, float]]:
        """Return (verif_ratio, open_ratio) for each phase."""
        return [(p.verif_ratio, p.open_ratio) for p in self.phases]

    def summary(self) -> str:
        """Human-readable summary of the curriculum schedule."""
        lines = [f"Curriculum: {len(self.phases)} phases, {self.total_steps} total steps"]
        prev = 0
        for i, (phase, boundary) in enumerate(zip(self.phases, self._boundaries)):
            lines.append(
                f"  Phase {i + 1}: steps {prev}-{boundary - 1} "
                f"({boundary - prev} steps) | "
                f"verif={phase.verif_ratio:.0%} open={phase.open_ratio:.0%} "
                f"lr_scale={phase.lr_scale}"
            )
            lines.append(f"    data: {phase.data_file}")
            prev = boundary
        return "\n".join(lines)

    def needs_data_switch(self, prev_step: int, curr_step: int) -> bool:
        """Check if a data file switch is needed between two steps."""
        return self.get_phase_index(prev_step) != self.get_phase_index(curr_step)

    def generate_parquets(
        self,
        verif_adapters: Sequence[Tuple[str, float]],
        open_adapters: Sequence[Tuple[str, float]],
        tokenizer=None,
        total_items_per_phase: Optional[int] = None,
        cache_paths: Optional[Dict[str, str]] = None,
        seed: int = 42,
    ) -> List[Path]:
        """Generate curriculum parquet files using prepare.py.

        Args:
            verif_adapters: List of (adapter_name, weight) for verifiable.
            open_adapters: List of (adapter_name, weight) for open.
            tokenizer: Optional HF tokenizer.
            total_items_per_phase: Rows per parquet. None = use all available.
            cache_paths: Dict mapping adapter name to precompute cache path.
            seed: Random seed.

        Returns:
            List of paths to generated parquet files.
        """
        from ..data.prepare import prepare_mixed_with_cache

        paths = []
        for i, phase in enumerate(self.phases):
            # Build adapter list with phase ratios
            v_total = sum(w for _, w in verif_adapters) or 1.0
            o_total = sum(w for _, w in open_adapters) or 1.0

            mixed = []
            for name, w in verif_adapters:
                mixed.append((name, phase.verif_ratio * w / v_total))
            for name, w in open_adapters:
                mixed.append((name, phase.open_ratio * w / o_total))

            p = prepare_mixed_with_cache(
                adapters_with_ratios=mixed,
                output_dir=self.data_dir,
                tokenizer=tokenizer,
                total_items=total_items_per_phase,
                split=f"curriculum_phase{i + 1}",
                cache_paths=cache_paths or {},
                seed=seed + i,
            )
            phase.data_file = str(p)
            paths.append(p)

        return paths


def parse_phases(phase_strs: List[str]) -> List[CurriculumPhase]:
    """Parse phase strings like '0.8:0.2:0.4' into CurriculumPhase objects.

    Format: ``verif_ratio:open_ratio:fraction[:lr_scale]``

    Examples:
        ``["0.8:0.2:0.4", "0.5:0.5:0.3", "0.2:0.8:0.3"]``
        ``["1.0:0.0:0.5:1.0", "0.0:1.0:0.5:0.3"]``
    """
    phases = []
    for s in phase_strs:
        parts = s.split(":")
        if len(parts) == 3:
            v, o, f = float(parts[0]), float(parts[1]), float(parts[2])
            phases.append(CurriculumPhase(verif_ratio=v, open_ratio=o, fraction=f))
        elif len(parts) == 4:
            v, o, f, lr = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            phases.append(CurriculumPhase(verif_ratio=v, open_ratio=o, fraction=f, lr_scale=lr))
        else:
            raise ValueError(f"Invalid phase format: {s}. Expected v:o:frac or v:o:frac:lr")
    return phases
