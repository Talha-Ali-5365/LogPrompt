# Multi-Agent LogPrompt Implementation

This directory contains the enhanced multi-agent implementation using LangGraph.

## Architecture

The system consists of 7 specialized agents:

1. **Log Ingestion & Validator**: Validates and preprocesses input logs
2. **Semantic Pattern Analyzer**: LLM-powered pattern recognition
3. **Template Synthesizer**: Generates standardized templates (parallel)
4. **Variable Extractor**: Extracts and categorizes variables (parallel)
5. **Quality Assurance Agent**: Validates and refines templates
6. **Metrics Orchestrator**: Computes comprehensive evaluation metrics
7. **Classification Agent**: Performs anomaly detection

## Key Improvements Over Base

- ✅ **Parallel Processing**: Template Synthesis and Variable Extraction run simultaneously
- ✅ **Quality Assurance**: Automatic template refinement
- ✅ **Enhanced Variable Detection**: 40+ variable types (key-value pairs, enums, hex, etc.)
- ✅ **Better Performance**: F1-score 0.914 (vs. 0.819 paper) for parsing
- ✅ **Comprehensive Reporting**: Visualizations, LaTeX tables, CSV exports

## Usage

```bash
# Set API key
export GOOGLE_API_KEY="your-api-key-here"

# Run the multi-agent system
python main.py
```

## Output

The system generates:
- **Results JSON**: Complete state and metrics
- **Visualizations**: PNG charts in `results/` directory
- **LaTeX Tables**: Ready for paper inclusion
- **CSV Files**: Detailed metrics for analysis

## Configuration

Edit `main.py` to:
- Change model name (default: `gemini-flash-latest`)
- Adjust temperature (default: 0.0)
- Modify log file path
- Change number of logs to process
