"""
LangGraph Multi-Agent Workflow for Log Parsing
Orchestrates 6 specialized agents in a sophisticated workflow with parallel execution
"""

import os
from typing import Literal
from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI

from state import LogParsingState
from agents import (
    LogIngestionValidator,
    SemanticPatternAnalyzer,
    TemplateSynthesizer,
    VariableExtractor,
    QualityAssuranceAgent,
    MetricsOrchestrator,
    ClassificationAgent
)


class LogParsingWorkflow:
    """
    Multi-Agent Log Parsing Workflow using LangGraph
    
    Architecture:
    =============
    START
      ↓
    [Log Ingestion & Validator] (Sequential)
      ↓
    [Semantic Pattern Analyzer] (Sequential)
      ↓
    ┌─────────────────────┬──────────────────────┐
    │                     │                      │
    [Template Synthesizer] [Variable Extractor]  (Parallel)
    │                     │                      │
    └─────────────────────┴──────────────────────┘
      ↓
    [Quality Assurance Agent] (Sequential)
      ↓
    [Metrics Orchestrator] (Sequential)
      ↓
    [Classification Agent] (Sequential) ← NEW: Anomaly Detection
      ↓
    END
    
    Features:
    - 7 specialized agents (including Classification Agent)
    - Parallel execution for Template Synthesis & Variable Extraction
    - Sequential coordination for validation, metrics, and classification
    - Conditional routing based on quality checks
    - Comprehensive state management
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-flash-latest", temperature: float = 0.0):
        """
        Initialize the multi-agent workflow
        
        Args:
            api_key: Google Gemini API key
            model_name: Model to use for LLM-powered agents
            temperature: Temperature for generation (0.0 for deterministic)
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )
        
        # Initialize all agents
        self.ingestion_agent = LogIngestionValidator(self.llm)
        self.pattern_agent = SemanticPatternAnalyzer(self.llm)
        self.template_agent = TemplateSynthesizer(self.llm)
        self.variable_agent = VariableExtractor(self.llm)
        self.qa_agent = QualityAssuranceAgent(self.llm)
        self.metrics_agent = MetricsOrchestrator(self.llm)
        self.classification_agent = ClassificationAgent(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Constructs the LangGraph StateGraph with all agents and edges
        """
        # Create the state graph
        workflow = StateGraph(LogParsingState)
        
        # Add nodes (agents)
        workflow.add_node("log_ingestion", self._ingestion_node)
        workflow.add_node("pattern_analysis", self._pattern_node)
        workflow.add_node("template_synthesis", self._template_node)
        workflow.add_node("variable_extraction", self._variable_node)
        workflow.add_node("quality_assurance", self._qa_node)
        workflow.add_node("metrics_orchestration", self._metrics_node)
        workflow.add_node("classification", self._classification_node)
        
        # Define edges (workflow flow)
        
        # Sequential: START → Log Ingestion
        workflow.add_edge(START, "log_ingestion")
        
        # Sequential: Log Ingestion → Pattern Analysis
        workflow.add_edge("log_ingestion", "pattern_analysis")
        
        # Parallel: Pattern Analysis → Template Synthesis & Variable Extraction
        workflow.add_edge("pattern_analysis", "template_synthesis")
        workflow.add_edge("pattern_analysis", "variable_extraction")
        
        # Convergence: Both parallel agents → Quality Assurance
        workflow.add_edge("template_synthesis", "quality_assurance")
        workflow.add_edge("variable_extraction", "quality_assurance")
        
        # Sequential: Quality Assurance → Metrics Orchestration
        workflow.add_edge("quality_assurance", "metrics_orchestration")
        
        # Sequential: Metrics Orchestration → Classification
        workflow.add_edge("metrics_orchestration", "classification")
        
        # Conditional: Classification → END or reprocess
        workflow.add_conditional_edges(
            "classification",
            self._routing_decision,
            {
                "complete": END,
                "reprocess": "pattern_analysis"  # Loop back if needed
            }
        )
        
        return workflow.compile()
    
    # Agent wrapper functions for graph nodes
    
    def _ingestion_node(self, state: LogParsingState) -> LogParsingState:
        """Log Ingestion & Validation node"""
        return self.ingestion_agent.process(state)
    
    def _pattern_node(self, state: LogParsingState) -> LogParsingState:
        """Semantic Pattern Analyzer node"""
        return self.pattern_agent.process(state)
    
    def _template_node(self, state: LogParsingState) -> LogParsingState:
        """Template Synthesizer node (parallel)"""
        return self.template_agent.process(state)
    
    def _variable_node(self, state: LogParsingState) -> LogParsingState:
        """Variable Extractor node (parallel)"""
        return self.variable_agent.process(state)
    
    def _qa_node(self, state: LogParsingState) -> LogParsingState:
        """Quality Assurance node"""
        return self.qa_agent.process(state)
    
    def _metrics_node(self, state: LogParsingState) -> LogParsingState:
        """Metrics Orchestrator node"""
        return self.metrics_agent.process(state)
    
    def _classification_node(self, state: LogParsingState) -> LogParsingState:
        """Classification/Anomaly Detection node"""
        return self.classification_agent.process(state)
    
    def _routing_decision(self, state: LogParsingState) -> Literal["complete", "reprocess"]:
        """
        Conditional routing based on quality validation
        Determines if processing is complete or requires reprocessing
        """
        # Check if reprocessing is needed
        if state.get("requires_reprocessing", False) and state.get("quality_score", 1.0) < 0.5:
            return "reprocess"
        return "complete"
    
    def execute(self, logs: list[str]) -> dict:
        """
        Execute the multi-agent workflow on input logs
        
        Args:
            logs: List of log messages to parse
            
        Returns:
            Final state with all results and metrics
        """
        # Initialize state
        initial_state: LogParsingState = {
            "raw_logs": logs,
            "total_logs": len(logs),
            "current_stage": "initialized",
            "processing_status": "starting",
            "identified_patterns": [],
            "pattern_confidence": 0.0,
            "pattern_metadata": {},
            "generated_templates": [],
            "template_quality_scores": [],
            "template_generation_method": "",
            "extracted_variables": [],
            "variable_types": [],
            "extraction_confidence": [],
            "quality_validated": False,
            "quality_score": 0.0,
            "validation_errors": [],
            "quality_recommendations": [],
            "parsed_results": [],
            "classifications": [],
            "classification_confidences": [],
            "classification_explanations": [],
            "classification_method": "",
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "classification_accuracy": 0.0,
            "classification_precision": 0.0,
            "classification_recall": 0.0,
            "classification_f1_score": 0.0,
            "normal_count": 0,
            "abnormal_count": 0,
            "agents_executed": [],
            "execution_times": {},
            "requires_reprocessing": False,
            "routing_decision": None
        }
        
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
    
    def visualize_workflow(self, save_png: bool = True, output_dir: str = "results"):
        """
        Generate and display Mermaid diagram of the workflow
        Also saves as PNG if possible
        
        Args:
            save_png: Whether to attempt saving as PNG
            output_dir: Directory to save PNG file
        """
        try:
            # Generate Mermaid diagram using LangGraph's built-in method
            mermaid_code = self.graph.get_graph().draw_mermaid()
            
            print("\n" + "="*80)
            print("Workflow Architecture (Mermaid Diagram)")
            print("="*80)
            print("\nCopy this code to https://mermaid.live to visualize:\n")
            print(mermaid_code)
            print("\n" + "="*80)
            
            # Save Mermaid code to file
            os.makedirs(output_dir, exist_ok=True)
            mmd_path = os.path.join(output_dir, "workflow_diagram.mmd")
            with open(mmd_path, "w") as f:
                f.write(mermaid_code)
            print(f"✓ Mermaid diagram saved to: {mmd_path}")
            
            # Try to save as PNG
            if save_png:
                png_path = os.path.join(output_dir, "workflow_diagram.png")
                success = self._save_mermaid_as_png(mermaid_code, png_path)
                if success:
                    print(f"✓ Workflow diagram saved as PNG to: {png_path}")
                else:
                    print(f"⚠️  Could not save PNG. Install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
                    print(f"   Or use the Mermaid code at: {mmd_path}")
            
            return mermaid_code
        except Exception as e:
            print(f"⚠️  Could not generate Mermaid diagram: {e}")
            print("\nManual workflow structure:")
            print("  START → Ingestion → Pattern Analysis → [Template + Variable] → QA → Metrics → Classification → END")
            return None
    
    def _save_mermaid_as_png(self, mermaid_code: str, output_path: str) -> bool:
        """
        Save Mermaid diagram as PNG using mermaid-cli or alternative methods
        
        Args:
            mermaid_code: Mermaid diagram code
            output_path: Path to save PNG file
            
        Returns:
            True if successful, False otherwise
        """
        import subprocess
        import tempfile
        
        # Method 1: Try using mermaid-cli (mmdc) if available
        try:
            # Create temporary file with mermaid code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as tmp_file:
                tmp_file.write(mermaid_code)
                tmp_mmd_path = tmp_file.name
            
            # Try to use mermaid-cli
            result = subprocess.run(
                ['mmdc', '-i', tmp_mmd_path, '-o', output_path, '-b', 'transparent', '-w', '1920', '-H', '1080'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up temp file
            try:
                os.unlink(tmp_mmd_path)
            except:
                pass
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # mermaid-cli not available, try alternative
            pass
        
        # Method 2: Try using playwright with mermaid (if available)
        try:
            import subprocess
            import json
            import base64
            
            # Use mermaid.ink API (online service)
            import urllib.request
            import urllib.parse
            
            # Encode mermaid code
            encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()
            url = f"https://mermaid.ink/img/{encoded}"
            
            # Download the image
            urllib.request.urlretrieve(url, output_path)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
        except Exception as e:
            pass
        
        # Method 3: Try using requests with mermaid.ink
        try:
            import requests
            import base64
            
            encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()
            url = f"https://mermaid.ink/img/{encoded}"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return True
        except Exception:
            pass
        
        return False


def create_workflow(api_key: str = None) -> LogParsingWorkflow:
    """
    Factory function to create the multi-agent workflow
    
    Args:
        api_key: Google Gemini API key (or uses GOOGLE_API_KEY env var)
        
    Returns:
        Configured LogParsingWorkflow instance
    """
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set in GOOGLE_API_KEY environment variable")
    
    return LogParsingWorkflow(api_key=api_key)

