"""The firefly fleet.

Each firefly maps to an expensive human responsibility and activates only when
its expertise is needed:

    Data Scout          -> Data analyst          (schema, distributions, quality)
    Sampling Specialist -> Data scientist        (defensible working sample)
    Data Engineer       -> Data engineer         (transformations + reproducible pipeline)
    Analysis Planner    -> Senior data scientist (hypotheses, task & resource plan)
    Model Scientist     -> ML scientist          (baselines, experiments, selected model)
    Validator           -> Model-risk specialist (leakage, stability, validity)
    ML Engineer         -> ML platform engineer  (container, API, deployment manifest)
    FinOps              -> Infrastructure engineer (cost & compute optimisation)
    Reporter            -> Analyst / consultant  (executive & technical reports)
"""

from avaloka.fireflies.base import Firefly
from avaloka.fireflies.data_scout import DataScout
from avaloka.fireflies.sampling_specialist import SamplingSpecialist
from avaloka.fireflies.data_engineer import DataEngineer
from avaloka.fireflies.analysis_planner import AnalysisPlanner
from avaloka.fireflies.model_scientist import ModelScientist
from avaloka.fireflies.validator import Validator
from avaloka.fireflies.ml_engineer import MLEngineer
from avaloka.fireflies.finops import FinOps
from avaloka.fireflies.reporter import Reporter

__all__ = [
    "Firefly",
    "DataScout",
    "SamplingSpecialist",
    "DataEngineer",
    "AnalysisPlanner",
    "ModelScientist",
    "Validator",
    "MLEngineer",
    "FinOps",
    "Reporter",
]
