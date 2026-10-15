"""Stable pydantic v2 data contracts for gelpy.

This package is intentionally free of numpy/scipy/matplotlib imports so it can be
imported standalone (e.g. by a future MCP server) purely for schema discovery and
validation. Every model subclasses ``CanonicalModel`` (``extra="forbid"``), so an
unrecognized field or model name is a hard ``ValidationError`` rather than a silent
bug.
"""

from .common import CanonicalModel, ErrorResponse
from .image import ImageMeta, ImageRef
from .lanes import (
    LaneDefinition,
    LaneDetectionResult,
    LaneLayout,
    LaneSelector,
    LaneSelectorTerm,
)
from .contrast import ContrastConfig, GammaCorrection, IntensityRange
from .background import BackgroundStageConfig, PlaneRansacParams, RobustPlaneParams
from .profiles import LineProfileConfig, NormalizationType
from .baseline import BaselineConfig, BaselineFit
from .peaks import (
    BandFit,
    BiGaussianPeakParams,
    EmgPeakParams,
    FitQuality,
    GaussianPeakParams,
    LaneFitResult,
    PeakParams,
    PseudoVoigtPeakParams,
)
from .fit_config import FitConfig, FitConfigStore, PerLaneFitConfig, PriorConfig
from .recipe import Recipe
from .requests import (
    DetectLanesRequest,
    ExtractProfilesRequest,
    FitConfigOverrides,
    FitRunRequest,
    GetResultsRequest,
    HtmlReportRequest,
    ImageKind,
    InitRunRequest,
    LineProfileConfigOverrides,
    RefitLaneRequest,
    RemoveBackgroundRequest,
    RenderImageRequest,
    RequestModel,
    SetContrastRequest,
    SetImageRequest,
    SetLanesRequest,
    StatusRequest,
)
from .report import ReportMarker, ReportStatus
from .run import LaneStatus, RunManifest, RunStatus, RunUpdateSummary, StageRecord
from .results import AnalysisResult
from .tools import (
    HtmlReportResult,
    ImageRenderResult,
    PathRole,
    ToolAnnotations,
    ToolList,
    ToolSpec,
)

__all__ = [
    "CanonicalModel",
    "ErrorResponse",
    "ImageMeta",
    "ImageRef",
    "LaneDefinition",
    "LaneDetectionResult",
    "LaneLayout",
    "LaneSelector",
    "LaneSelectorTerm",
    "ContrastConfig",
    "GammaCorrection",
    "IntensityRange",
    "BackgroundStageConfig",
    "PlaneRansacParams",
    "RobustPlaneParams",
    "LineProfileConfig",
    "NormalizationType",
    "BaselineConfig",
    "BaselineFit",
    "BandFit",
    "BiGaussianPeakParams",
    "EmgPeakParams",
    "FitQuality",
    "GaussianPeakParams",
    "LaneFitResult",
    "PeakParams",
    "PseudoVoigtPeakParams",
    "FitConfig",
    "FitConfigStore",
    "PerLaneFitConfig",
    "PriorConfig",
    "Recipe",
    "DetectLanesRequest",
    "ExtractProfilesRequest",
    "FitConfigOverrides",
    "FitRunRequest",
    "GetResultsRequest",
    "HtmlReportRequest",
    "ImageKind",
    "InitRunRequest",
    "LineProfileConfigOverrides",
    "RefitLaneRequest",
    "RemoveBackgroundRequest",
    "RenderImageRequest",
    "RequestModel",
    "SetContrastRequest",
    "SetImageRequest",
    "SetLanesRequest",
    "StatusRequest",
    "LaneStatus",
    "ReportMarker",
    "ReportStatus",
    "RunManifest",
    "RunStatus",
    "RunUpdateSummary",
    "StageRecord",
    "AnalysisResult",
    "HtmlReportResult",
    "ImageRenderResult",
    "PathRole",
    "ToolAnnotations",
    "ToolList",
    "ToolSpec",
]
