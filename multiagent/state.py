"""
State Management for Multi-Agent Log Parsing System
Defines the shared state across all agents in the LangGraph workflow
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Merge two dictionaries, combining their keys"""
    return {**left, **right}


class LogParsingState(TypedDict):
    """
    Comprehensive state object shared across all agents in the multi-agent system.
    Uses TypedDict for type safety and LangGraph compatibility.
    """
    # Input data
    raw_logs: List[str]
    total_logs: int
    
    # Processing metadata
    current_stage: str
    processing_status: str
    
    # Pattern Analysis results
    identified_patterns: List[Dict[str, any]]
    pattern_confidence: float
    pattern_metadata: Dict[str, any]
    
    # Template Synthesis results
    generated_templates: List[str]
    template_quality_scores: List[float]
    template_generation_method: str
    
    # Variable Extraction results
    extracted_variables: List[Dict[str, str]]
    variable_types: List[List[str]]
    extraction_confidence: List[float]
    
    # Quality Assurance results
    quality_validated: bool
    quality_score: float
    validation_errors: Annotated[List[str], add]  # Accumulate errors
    quality_recommendations: List[str]
    
    # Final outputs
    parsed_results: List[Dict[str, any]]
    
    # Classification/Anomaly Detection results
    classifications: List[str]  # "normal" or "abnormal" for each log
    classification_confidences: List[float]
    classification_explanations: List[str]
    classification_method: str
    
    # Metrics and evaluation
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Classification metrics
    classification_accuracy: float
    classification_precision: float
    classification_recall: float
    classification_f1_score: float
    normal_count: int
    abnormal_count: int
    
    # Agent execution tracking
    agents_executed: Annotated[List[str], add]  # Track agent execution order
    execution_times: Annotated[Dict[str, float], merge_dicts]  # Merge execution times from all agents
    
    # Decision routing
    requires_reprocessing: bool
    routing_decision: Optional[str]

