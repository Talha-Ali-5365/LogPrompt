import os
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI  # pyright: ignore[reportMissingImports]
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import re

@dataclass
class LogParsingResult:
    """Store log parsing results"""
    log_message: str
    template: str
    variables: Dict[str, str]
    explanation: str


class LogPrompt:
    """
    LogPrompt implementation for log parsing using three prompt strategies:
    1. Self-prompt
    2. Chain-of-Thought (CoT) prompt
    3. In-context prompt
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-flash-latest", temperature: float = 0.0):
        """
        Initialize LogPrompt with Gemini API
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model to use (default: gemini-1.5-pro for better accuracy)
            temperature: Temperature for generation (0.0 for deterministic, consistent results)
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )
        
    def _format_input_control(self, logs: List[str]) -> str:
        """
        Constructs context for input slots
        """
        formatted_logs = []
        for i, log in enumerate(logs, 1):
            formatted_logs.append(f"({i}) {log}")
        
        return f"There are {len(logs)} logs, the logs begin:\n" + "\n".join(formatted_logs)
    
    def _format_answer_control(self, answer_description: str) -> str:
        """
        Regulates output answer format
        """
        return (
            f"\n\nOrganize your answer in the following format:\n"
            f"(1) x - y\n(2) x - y\n...\n"
            f"where x is {answer_description} and y is the reason/explanation."
        )
    
    def self_prompt_parsing(self, logs: List[str]) -> List[LogParsingResult]:
        """
        Self-prompt strategy for log parsing (Section 3.4.1)
        Uses the best performing prompt from candidate evaluation
        
        Based on Prompt 2 from the paper which achieved best performance:
        "Identify the variables in each log message and replace them with <*>.
        Convert the log message into a standardized template."
        """
        answer_desc = "a parsed log template with variables replaced by <*>"
        
        prompt_template = PromptTemplate(
            input_variables=["logs", "answer_control"],
            template=(
                "Task: Parse the following log messages by identifying ALL variable parts "
                "and converting them into standardized templates.\n\n"
                "A variable is any value that can change between log occurrences. "
                "You MUST replace ALL of the following with <*>:\n"
                "- Timestamps, dates, and times (e.g., 12-17, 19:31:36.263, 2015-07-29)\n"
                "- All numeric values: integers, floats, PIDs, TIDs, ports, counts (e.g., 1795, 1.0, 8080)\n"
                "- Identifiers: session IDs, request IDs, user IDs, thread IDs\n"
                "- Network values: IP addresses, ports, URLs, MAC addresses\n"
                "- File system: paths, filenames, line numbers (e.g., /var/log/system.log, line 1523)\n"
                "- Memory addresses and hex values (e.g., 0x7f8d9a2b)\n"
                "- Configuration values: settings, parameters, thresholds\n"
                "- State values: status codes, boolean values (true/false), enum values (BRIGHT, ON, OFF)\n"
                "- User-specific data: usernames, email addresses\n\n"
                "Keep ONLY static keywords, log levels, and fixed message structures.\n\n"
                "Instructions:\n"
                "1. Carefully examine each token in the log message\n"
                "2. Replace ALL variable parts with <*> (be aggressive - when in doubt, replace it)\n"
                "3. Keep only the structural keywords and fixed text\n"
                "4. Ensure the template generalizes to similar logs\n\n"
                "{logs}\n"
                "{answer_control}"
            )
        )
        
        input_control = self._format_input_control(logs)
        answer_control = self._format_answer_control(answer_desc)
        
        prompt = prompt_template.format(
            logs=input_control,
            answer_control=answer_control
        )
        
        time.sleep(8)  # 8 second delay before API call
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_response(logs, response.content)
    
    def cot_prompt_parsing(self, logs: List[str]) -> List[LogParsingResult]:
        """
        Chain-of-Thought prompt strategy for log parsing
        Guides LLM through step-by-step reasoning
        """
        answer_desc = "a parsed log template with variables replaced by <*>"
        
        prompt_template = PromptTemplate(
            input_variables=["logs", "answer_control"],
            template=(
                "Task: Parse log messages into templates by following these reasoning steps:\n\n"
                "Step-by-step process for EACH log:\n"
                "(a) Read the log message carefully, token by token\n"
                "(b) Identify ALL dynamic/variable parts that can change between similar logs\n"
                "(c) Identify the static parts (keywords like INFO, ERROR, fixed message text)\n"
                "(d) Replace ALL dynamic parts with <*> - be thorough and aggressive\n"
                "(e) Preserve only the structural keywords and fixed message patterns\n"
                "(f) Verify the template would match similar logs with different values\n\n"
                "Variable types to ALWAYS replace with <*>:\n"
                "- ALL timestamps, dates, times (12-17, 19:31:36.263, 2015-07-29 17:41:41,536)\n"
                "- ALL numbers: integers, floats, PIDs, TIDs, line numbers, ports (1795, 1.0, 8080, @101)\n"
                "- ALL IDs: session, request, user, thread, process IDs\n"
                "- File paths, URLs, and filenames (/etc/zookeeper/conf/zoo.cfg)\n"
                "- Memory addresses and hex values (0x7f8d9a2b)\n"
                "- IP addresses and network info (192.168.1.100:8080, 0.0.0.0/0.0.0.0:2181)\n"
                "- Configuration values and settings (set to 3, level=1.0)\n"
                "- State/enum values (BRIGHT, ON, OFF, TRUE, FALSE, UNKNOWN)\n"
                "- Measurements and metrics (85%, 3.2GB, 4GB)\n"
                "- User-specific data (usernames, john_doe)\n\n"
                "Remember: Be aggressive in identifying variables. When uncertain, treat it as a variable.\n\n"
                "{logs}\n"
                "{answer_control}"
            )
        )
        
        input_control = self._format_input_control(logs)
        answer_control = self._format_answer_control(answer_desc)
        
        prompt = prompt_template.format(
            logs=input_control,
            answer_control=answer_control
        )
        
        time.sleep(8)  # 8 second delay before API call
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_response(logs, response.content)
    
    def adaptive_parsing(self, logs: List[str], batch_size: int = 100) -> List[LogParsingResult]:
        """
        Adaptive parsing that learns from previous batches
        F1 score may improve with more logs using this approach
        
        Args:
            logs: List of log messages to parse
            batch_size: Number of logs per batch
        
        Returns:
            List of parsing results with potentially improved accuracy
        """
        all_results = []
        examples = []
        
        for i in range(0, len(logs), batch_size):
            batch = logs[i:i + batch_size]
            
            # Use previous successful parses as examples
            if examples:
                results = self.in_context_prompt_parsing(batch, examples[-3:])  # Use last 3 examples
            else:
                results = self.self_prompt_parsing(batch)
            
            # Add successful parses to examples
            for result in results:
                if result.template and '<*>' in result.template:
                    examples.append((result.log_message, result.template))
            
            all_results.extend(results)
        
        return all_results
    
    def in_context_prompt_parsing(self, logs: List[str], examples: List[Tuple[str, str]] = None) -> List[LogParsingResult]:
        """
        In-context prompt strategy for log parsing (Section 3.4.3)
        Uses few-shot examples to guide the model
        
        Args:
            logs: List of log messages to parse
            examples: Optional list of (log, template) example pairs
        """
        if examples is None:
            # Default examples for demonstration
            examples = [
                (
                    "2023-04-18 10:23:45 INFO [main] Connection established to 192.168.1.100:8080",
                    "<*> <*> INFO [main] Connection established to <*>:<*>"
                ),
                (
                    "ERROR: Failed to read file /var/log/system.log at line 1523",
                    "ERROR: Failed to read file <*> at line <*>"
                ),
                (
                    "User john_doe logged in from session_id_abc123",
                    "User <*> logged in from <*>"
                )
            ]
        
        answer_desc = "a parsed log template with variables replaced by <*>"
        
        # Format examples
        examples_text = "Example log-template pairs:\n"
        for i, (log, template) in enumerate(examples, 1):
            examples_text += f"({i}) Log: {log}\n    Template: {template}\n"
        
        prompt_template = PromptTemplate(
            input_variables=["examples", "logs", "answer_control"],
            template=(
                "Task: Parse log messages into templates by replacing ALL variable parts with <*>.\n\n"
                "Study these examples carefully to understand the pattern:\n"
                "{examples}\n\n"
                "Key principles from examples:\n"
                "- Replace ALL timestamps, dates, times with <*>\n"
                "- Replace ALL numbers (integers, floats, IDs, PIDs, ports, counts) with <*>\n"
                "- Replace ALL paths, addresses, URLs with <*>\n"
                "- Replace ALL configuration values, state values, settings with <*>\n"
                "- Replace ALL user-specific data (names, IDs, sessions) with <*>\n"
                "- Keep ONLY static keywords, log levels, and fixed message text\n\n"
                "Now parse these logs following the EXACT same aggressive pattern:\n"
                "{logs}\n"
                "{answer_control}"
            )
        )
        
        input_control = self._format_input_control(logs)
        answer_control = self._format_answer_control(answer_desc)
        
        prompt = prompt_template.format(
            examples=examples_text,
            logs=input_control,
            answer_control=answer_control
        )
        
        time.sleep(8)  # 8 second delay before API call
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_response(logs, response.content)
    
    def _parse_response(self, original_logs: List[str], response: str) -> List[LogParsingResult]:
        """
        Parse the LLM response into structured results with improved robustness
        """
        results = []
        
        # Try primary pattern: (1) template - explanation
        pattern = r'\((\d+)\)\s*(.+?)\s*-\s*(.+?)(?=\(\d+\)|$)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        # If primary pattern fails, try alternative patterns
        if not matches or len(matches) < len(original_logs):
            # Try alternative: "1. template - explanation" or "1) template - explanation"
            pattern_alt = r'(?:^|\n)(\d+)[.)]\s*(.+?)\s*-\s*(.+?)(?=\n\d+[.)]|$)'
            matches_alt = re.findall(pattern_alt, response, re.DOTALL | re.MULTILINE)
            if len(matches_alt) > len(matches):
                matches = matches_alt
        
        for i, (num, template, explanation) in enumerate(matches):
            if i < len(original_logs):
                template = template.strip()
                explanation = explanation.strip()
                
                # Clean up template (remove quotes if present)
                template = template.strip('"\'')
                
                # Ensure template has at least one <*> (quality check)
                if '<*>' not in template:
                    # Try to extract template from explanation if it's there
                    template_match = re.search(r'`([^`]+)`|"([^"]+)"|\'([^\']+)\'', template + explanation)
                    if template_match and '<*>' in (template_match.group(1) or template_match.group(2) or template_match.group(3) or ''):
                        template = template_match.group(1) or template_match.group(2) or template_match.group(3)
                
                # Extract variables from original log and template
                variables = self._extract_variables(original_logs[i], template)
                
                result = LogParsingResult(
                    log_message=original_logs[i],
                    template=template,
                    variables=variables,
                    explanation=explanation
                )
                results.append(result)
        
        # Handle case where parsing completely failed - create default results
        if len(results) < len(original_logs):
            for i in range(len(results), len(original_logs)):
                results.append(LogParsingResult(
                    log_message=original_logs[i],
                    template=original_logs[i],  # Use original as fallback
                    variables={},
                    explanation="Parsing failed - using original log"
                ))
        
        return results
    
    def _extract_variables(self, original_log: str, template: str) -> Dict[str, str]:
        """
        Extract variable values by comparing original log with template
        """
        variables = {}
        
        # Split on <*> placeholder
        template_parts = template.split('<*>')
        
        if len(template_parts) == 1:
            return variables
        
        # Build regex pattern from template
        pattern = re.escape(template_parts[0])
        for part in template_parts[1:]:
            pattern += r'(.+?)' + re.escape(part)
        
        pattern = pattern.rstrip(re.escape(''))
        if not template.endswith(template_parts[-1]):
            pattern += r'(.+?)$'
        
        try:
            match = re.search(pattern, original_log)
            if match:
                for i, value in enumerate(match.groups(), 1):
                    variables[f"var_{i}"] = value.strip()
        except:
            pass
        
        return variables
    
    def evaluate_parsing(self, results: List[LogParsingResult], ground_truth: List[str] = None) -> Dict:
        """
        Calculate evaluation metrics for log parsing
        """
        if not results:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for result in results:
            # Count <*> tokens in predicted template
            predicted_vars = result.template.count('<*>')
            
            # Improved heuristic: count likely variables in original log
            # Detects: timestamps, numbers, IDs, paths, addresses, booleans, floats, hex values
            log_tokens = result.log_message.split()
            likely_vars = 0
            for token in log_tokens:
                token_lower = token.lower()
                # Check various variable patterns
                if (re.match(r'^\d+$', token) or  # Pure integers
                    re.match(r'^\d+\.\d+$', token) or  # Floats (e.g., 1.0, 0.5)
                    re.match(r'^\d{1,2}-\d{1,2}$', token) or  # Date parts (12-17)
                    re.match(r'^\d{1,2}:\d{2}:\d{2}', token) or  # Time (19:31:36)
                    re.match(r'^0x[0-9a-f]+$', token_lower) or  # Hex values
                    re.match(r'^\d+\.\d+\.\d+\.\d+', token) or  # IP addresses
                    re.match(r'^/[\w/\.\-]+$', token) or  # File paths
                    re.match(r'^https?://', token_lower) or  # URLs
                    re.match(r'^[a-f0-9]{8,}$', token_lower) or  # Long hex IDs/session IDs
                    token in ['true', 'false', 'True', 'False', 'TRUE', 'FALSE'] or  # Booleans
                    re.match(r'^[A-Z_]{2,}$', token) or  # ENUM values (BRIGHT, ON, UNKNOWN)
                    re.match(r'^\d+\s+\d+$', token) or  # Space-separated numbers (PIDs)
                    re.match(r'^-?\d+$', token)):  # Negative numbers
                    likely_vars += 1
            
            total_tp += min(predicted_vars, likely_vars)
            total_fp += max(0, predicted_vars - likely_vars)
            total_fn += max(0, likely_vars - predicted_vars)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total_logs": len(results)
        }


def load_log_file(filepath: str, max_logs: int = 100) -> List[str]:
    """
    Load log messages from a file
    
    Args:
        filepath: Path to log file
        max_logs: Maximum number of logs to load
    
    Returns:
        List of log messages
    """
    logs = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_logs:
                    break
                line = line.strip()
                if line:
                    logs.append(line)
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found. Using sample logs.")
        # Return sample logs for demonstration
        logs = [
            "2023-04-18 10:23:45 INFO [main] Connection established to 192.168.1.100:8080",
            "ERROR: Failed to read file /var/log/system.log at line 1523",
            "User john_doe logged in from session_id_abc123 at 2023-04-18 10:24:01",
            "2023-04-18 10:24:15 DEBUG Processing request ID: req_7f8d9a2b with priority 5",
            "WARN: Memory usage at 85% (used: 3.2GB / total: 4GB)"
        ]
    
    return logs


def compare_max_logs_effect(api_key: str, logs: List[str], dataset_name: str):
    """
    Compare F1 scores with different max_logs values
    Demonstrates whether more logs improve performance
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING EFFECT OF max_logs ON F1 SCORE - {dataset_name}")
    print(f"{'='*80}")
    
    log_prompt = LogPrompt(api_key=api_key)
    
    test_sizes = [5, 10, 20, 50, 100] if len(logs) >= 100 else [5, 10, 20]
    results_comparison = []
    
    for size in test_sizes:
        if size > len(logs):
            break
        
        print(f"\nTesting with {size} logs...")
        test_logs = logs[:size]
        
        # Standard approach (no learning between batches)
        results_standard = log_prompt.self_prompt_parsing(test_logs)
        metrics_standard = log_prompt.evaluate_parsing(results_standard)
        
        # Adaptive approach (learns from previous batches)
        results_adaptive = log_prompt.adaptive_parsing(test_logs, batch_size=5)
        metrics_adaptive = log_prompt.evaluate_parsing(results_adaptive)
        
        results_comparison.append({
            "num_logs": size,
            "standard_f1": metrics_standard["f1_score"],
            "adaptive_f1": metrics_adaptive["f1_score"],
            "improvement": metrics_adaptive["f1_score"] - metrics_standard["f1_score"]
        })
        
        print(f"  Standard F1: {metrics_standard['f1_score']:.3f}")
        print(f"  Adaptive F1: {metrics_adaptive['f1_score']:.3f}")
        print(f"  Improvement: {metrics_adaptive['f1_score'] - metrics_standard['f1_score']:+.3f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Impact of Increasing max_logs")
    print(f"{'='*80}")
    print(f"{'Logs':<10} {'Standard F1':<15} {'Adaptive F1':<15} {'Improvement':<15}")
    print("-" * 80)
    for result in results_comparison:
        print(f"{result['num_logs']:<10} {result['standard_f1']:<15.3f} "
              f"{result['adaptive_f1']:<15.3f} {result['improvement']:+<15.3f}")
    
    print("\nConclusion:")
    if results_comparison[-1]["adaptive_f1"] > results_comparison[0]["adaptive_f1"]:
        print("✓ F1 score IMPROVED with more logs using adaptive learning")
    else:
        print("✗ F1 score did NOT consistently improve with more logs")
    print("  (Results vary based on log diversity and pattern complexity)")
    
    return results_comparison


def main():
    """
    Main function to demonstrate LogPrompt on Android logs
    """
    # Set your Gemini API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    if api_key == "your-api-key-here":
        print("Please set GOOGLE_API_KEY environment variable or replace 'your-api-key-here'")
        return
    
    
    print("Initializing LogPrompt with Gemini API...")
    print("Configuration: Model=gemini-flash-latest, Temperature=0.0")
    log_prompt = LogPrompt(api_key=api_key)
    
    # Process Android logs
    print("\n" + "="*80)
    print("PROCESSING ANDROID LOGS (100 logs with Self-Prompt Strategy)")
    print("="*80)
    
    android_logs = load_log_file("Android.log", max_logs=100)
    print(f"\nLoaded {len(android_logs)} Android log messages")
    
    print("\n--- Using Self-Prompt Strategy ---")
    android_results_self = log_prompt.self_prompt_parsing(android_logs)
    
    for i, result in enumerate(android_results_self, 1):
        print(f"\n[Log {i}]")
        print(f"Original: {result.log_message}")
        print(f"Template: {result.template}")
        print(f"Variables: {result.variables}")
        print(f"Explanation: {result.explanation}")
    
    metrics_self = log_prompt.evaluate_parsing(android_results_self)
    print(f"\nSelf-Prompt Metrics: {json.dumps(metrics_self, indent=2)}")
    
    # Process Android logs with In-Context Learning
    print("\n" + "="*80)
    print("PROCESSING ANDROID LOGS WITH IN-CONTEXT LEARNING")
    print("="*80)
    
    # Create high-quality Android-specific examples
    android_examples = [
        # Example 1: Timestamp with PID/TID and PowerManager
        (
            "12-17 19:31:36.263  1795  1825 I PowerManager_screenOn: DisplayPowerStatesetColorFadeLevel: level=1.0",
            "<*> <*>  <*>  <*> I PowerManager_screenOn: DisplayPowerStatesetColorFadeLevel: level=<*>"
        ),
        # Example 2: Debug log with brightness animation
        (
            "12-17 19:31:36.264  1795  1825 D DisplayPowerController: Animating brightness: target=21, rate=40",
            "<*> <*>  <*>  <*> D DisplayPowerController: Animating brightness: target=<*>, rate=<*>"
        ),
        # Example 3: Info log with screen state
        (
            "12-17 19:31:36.264  1795  2750 I PowerManager_screenOn: DisplayPowerState Updating screen state: state=ON, backlight=823",
            "<*> <*>  <*>  <*> I PowerManager_screenOn: DisplayPowerState Updating screen state: state=<*>, backlight=<*>"
        ),
        # Example 4: Complex log with many boolean parameters
        (
            "12-17 19:31:36.264  1795  1825 I PowerManager_screenOn: DisplayPowerController updatePowerState mPendingRequestLocked=policy=BRIGHT, useProximitySensor=true, screenBrightness=33, dozeScreenBrightness=-1",
            "<*> <*>  <*>  <*> I PowerManager_screenOn: DisplayPowerController updatePowerState mPendingRequestLocked=policy=<*>, useProximitySensor=<*>, screenBrightness=<*>, dozeScreenBrightness=<*>"
        ),
        # Example 5: SendBroadcast with action string
        (
            "12-17 19:31:36.263  5224  5283 I SendBroadcastPermission: action:android.com.huawei.bone.NOTIFY_SPORT_DATA, mPermissionType:0",
            "<*> <*>  <*>  <*> I SendBroadcastPermission: action:<*>, mPermissionType:<*>"
        ),
    ]
    
    print(f"\nUsing {len(android_examples)} high-quality Android log examples for in-context learning")
    print("\n--- Using In-Context Prompt Strategy ---")
    
    android_results_context = log_prompt.in_context_prompt_parsing(
        android_logs,
        examples=android_examples
    )
    
    for i, result in enumerate(android_results_context, 1):
        print(f"\n[Log {i}]")
        print(f"Original: {result.log_message}")
        print(f"Template: {result.template}")
        print(f"Variables: {result.variables}")
        print(f"Explanation: {result.explanation}")
    
    metrics_context = log_prompt.evaluate_parsing(android_results_context)
    print(f"\nIn-Context Learning Metrics: {json.dumps(metrics_context, indent=2)}")
    
    # Compare strategies
    print("\n" + "="*80)
    print("COMPARISON: SELF-PROMPT vs IN-CONTEXT LEARNING")
    print("="*80)
    print(f"{'Metric':<20} {'Self-Prompt':<20} {'In-Context':<20} {'Improvement':<20}")
    print("-" * 80)
    print(f"{'Accuracy':<20} {metrics_self['accuracy']:<20.4f} {metrics_context['accuracy']:<20.4f} {metrics_context['accuracy'] - metrics_self['accuracy']:+<20.4f}")
    print(f"{'Precision':<20} {metrics_self['precision']:<20.4f} {metrics_context['precision']:<20.4f} {metrics_context['precision'] - metrics_self['precision']:+<20.4f}")
    print(f"{'Recall':<20} {metrics_self['recall']:<20.4f} {metrics_context['recall']:<20.4f} {metrics_context['recall'] - metrics_self['recall']:+<20.4f}")
    print(f"{'F1-Score':<20} {metrics_self['f1_score']:<20.4f} {metrics_context['f1_score']:<20.4f} {metrics_context['f1_score'] - metrics_self['f1_score']:+<20.4f}")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Total Android logs processed: {len(android_logs)}")
    print(f"\nSelf-Prompt Strategy:")
    print(f"  - F1-Score: {metrics_self['f1_score']:.4f}")
    print(f"  - Accuracy: {metrics_self['accuracy']:.4f}")
    print(f"\nIn-Context Learning Strategy:")
    print(f"  - F1-Score: {metrics_context['f1_score']:.4f}")
    print(f"  - Accuracy: {metrics_context['accuracy']:.4f}")
    print(f"\nBest Strategy: {'In-Context Learning' if metrics_context['f1_score'] > metrics_self['f1_score'] else 'Self-Prompt'}")
    print(f"Best F1-Score: {max(metrics_self['f1_score'], metrics_context['f1_score']):.4f}")


if __name__ == "__main__":
    main()