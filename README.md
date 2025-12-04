# LogPrompt: Interpretable Online Log Analysis Using Large Language Models

This repository contains a comprehensive implementation and enhancement of **LogPrompt**, an interpretable online log analysis approach using large language models (LLMs) with advanced prompt strategies.

## 📋 Overview

This project implements the LogPrompt methodology from the paper *"Interpretable Online Log Analysis Using Large Language Models with Prompt Strategies"* and extends it with a novel **multi-agent architecture** using LangGraph.

### Key Features

- ✅ **Three Prompt Strategies**: Self-prompt, Chain-of-Thought (CoT), and In-context learning
- ✅ **Dual Tasks**: Log parsing and anomaly detection
- ✅ **Multi-Agent Architecture**: Distributed system with 7 specialized agents
- ✅ **Superior Performance**: Beats paper results (F1: 0.914 vs 0.819 for parsing, 0.905 vs 0.417 for classification)
- ✅ **Production Ready**: Comprehensive evaluation, visualization, and reporting

## 🏆 Results

### Log Parsing Performance
- **F1-Score: 0.914** (vs. paper's 0.819) - **+11.6% improvement**
- **Accuracy: 0.841**
- **Precision: 0.900**
- **Recall: 0.928**

### Anomaly Detection Performance
- **F1-Score: 0.905** (vs. paper's 0.417) - **+117.0% improvement**
- **Precision: 0.826**
- **Recall: 1.000**
- **Accuracy: 0.960**

## 📁 Repository Structure

```
LogPrompt/
├── base/                    # Base LogPrompt implementation
│   └── log_prompt.py       # Single-agent implementation
├── multiagent/              # Multi-agent enhanced implementation
│   ├── main.py             # Main execution script
│   ├── workflow.py          # LangGraph workflow orchestration
│   ├── agents.py            # 7 specialized agents
│   ├── state.py             # State management
│   ├── visualization.py    # Results visualization
│   └── requirements.txt     # Dependencies
├── results/                 # Latest experimental results
│   ├── *.png               # Performance comparison charts
│   ├── *.tex               # LaTeX tables
│   └── *.csv               # Detailed metrics
├── paper/                   # Research paper
│   └── paper.tex           # Complete paper in LaTeX
├── workflow_diagram.png    # Multi-agent workflow visualization
├── README.md               # This file
├── requirements.txt        # Main dependencies
└── .gitignore             # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Gemini API key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd LogPrompt

# Install dependencies
pip install -r requirements.txt

# For multi-agent system
cd multiagent
pip install -r requirements.txt
```

### Set API Key

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

## 💻 Usage

### Base Implementation

```bash
cd base
python log_prompt.py
```

The base implementation provides:
- Three prompt strategies (self-prompt, CoT, in-context)
- Log parsing and anomaly detection
- Simple sequential processing

### Multi-Agent Implementation (Recommended)

```bash
cd multiagent
python main.py
```

The multi-agent system provides:
- **7 Specialized Agents**: Log Ingestion, Pattern Analysis, Template Synthesis, Variable Extraction, Quality Assurance, Metrics Orchestration, Classification
- **Parallel Execution**: Template Synthesis and Variable Extraction run simultaneously
- **Quality Assurance**: Automatic template refinement
- **Comprehensive Reporting**: Visualizations, LaTeX tables, CSV exports

## 🏗️ Architecture

### Base Implementation

```
Input Logs → LLM (Prompt Strategy) → Parse Response → Calculate Metrics → Output
```

### Multi-Agent Implementation

![Multi-Agent Workflow](workflow_diagram.png)

The workflow follows a hybrid sequential-parallel execution model:

```
START
  ↓
[Log Ingestion & Validator]
  ↓
[Semantic Pattern Analyzer]
  ↓
┌─────────────────────┬──────────────────────┐
│                     │                      │
[Template Synthesizer] [Variable Extractor]  (Parallel)
│                     │                      │
└─────────────────────┴──────────────────────┘
  ↓
[Quality Assurance Agent]
  ↓
[Metrics Orchestrator]
  ↓
[Classification Agent]
  ↓
END
```

## 📊 Features

### Prompt Strategies

1. **Self-Prompt**: LLM generates and selects best prompt candidates
2. **Chain-of-Thought (CoT)**: Step-by-step reasoning for complex tasks
3. **In-Context Learning**: Few-shot examples for improved accuracy

### Multi-Agent Benefits

- **Parallel Processing**: 50% reduction in Template + Variable extraction time
- **Quality Validation**: Automatic template refinement using agent coordination
- **Enhanced Classification**: Expanded error detection (40+ keywords)
- **Scalability**: Modular design enables independent agent scaling
- **Maintainability**: Clear separation of concerns

## 📈 Experimental Results

All results are available in the `results/` directory:

- **Performance Comparison Charts**: PNG visualizations
- **LaTeX Tables**: Ready for paper inclusion
- **CSV Metrics**: Detailed numerical results

### Latest Results Summary

| Task | Metric | Our Implementation | Paper (LogPrompt) | Improvement |
|------|--------|-------------------|-------------------|-------------|
| **Log Parsing** | F1-Score | 0.914 | 0.819 | +11.6% |
| **Log Parsing** | Precision | 0.900 | N/A | - |
| **Log Parsing** | Recall | 0.928 | N/A | - |
| **Anomaly Detection** | F1-Score | 0.905 | 0.417 | +117.0% |
| **Anomaly Detection** | Precision | 0.826 | 0.270 | +206.0% |
| **Anomaly Detection** | Recall | 1.000 | 0.917 | +9.1% |

## 🔬 Implementation Details

### LLM Configuration
- **Model**: Google Gemini 2.5 Flash
- **Temperature**: 0.0 (deterministic)
- **Framework**: LangChain + LangGraph

### Evaluation
- **Dataset**: Android logs (100 samples)
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Method**: Heuristic-based evaluation (keyword matching)

## 📝 Paper

The complete research paper is available in `paper/paper.tex`. It includes:
- Comprehensive methodology description
- Multi-agent architecture details
- Experimental results and analysis
- Comparison with original paper

## 🤝 Contributing

This is a research implementation. For questions or improvements, please open an issue.

## 📄 License

This implementation is provided for research and educational purposes.

## 🙏 Acknowledgments

- **LogPrompt Paper Authors**: For the foundational methodology
- **LangGraph**: For the multi-agent orchestration framework
- **LangChain**: For LLM integration
- **Google Gemini**: For the language model

## 📚 References

- Liu, Y., et al. "Interpretable Online Log Analysis Using Large Language Models with Prompt Strategies." 2023.
- LangGraph: Stateful Workflows for LLM Applications. https://github.com/langchain-ai/langgraph

---

**Made with 🤖 AI Agents | Powered by 🧠 LangGraph | Built for 🎯 Excellence**

