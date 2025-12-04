# Base LogPrompt Implementation

This directory contains the base implementation of LogPrompt with three prompt strategies.

## Features

- **Self-Prompt Strategy**: LLM generates and selects best prompt candidates
- **Chain-of-Thought (CoT) Prompt**: Step-by-step reasoning
- **In-Context Learning**: Few-shot examples for improved accuracy
- **Dual Tasks**: Log parsing and anomaly detection
- **Format Control Functions**: Input and answer format regulation

## Usage

```bash
# Set API key
export GOOGLE_API_KEY="your-api-key-here"

# Run the implementation
python log_prompt.py
```

## Configuration

Edit `log_prompt.py` to:
- Change model name (default: `gemini-flash-latest`)
- Adjust temperature (default: 0.0)
- Modify batch size (default: 100 logs)

## Output

The script generates:
- Parsing results with templates and variables
- Classification results with explanations
- Evaluation metrics (accuracy, precision, recall, F1-score)

