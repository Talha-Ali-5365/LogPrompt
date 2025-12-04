"""
Main Execution Script for Multi-Agent Log Parsing System
Demonstrates the LangGraph-based multi-agent architecture
"""

import os
import json
import time
from workflow import create_workflow
from visualization import ResultVisualizer


def load_android_logs(filepath: str = "../Android.log", max_logs: int = 100) -> list[str]:
    """Load Android logs from file"""
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
        print(f"Warning: {filepath} not found. Using sample logs.")
        logs = [
            "12-17 19:31:36.263  1795  1825 I PowerManager_screenOn: DisplayPowerStatesetColorFadeLevel: level=1.0",
            "12-17 19:31:36.263  5224  5283 I SendBroadcastPermission: action:android.com.huawei.bone.NOTIFY_SPORT_DATA, mPermissionType:0",
            "12-17 19:31:36.264  1795  1825 D DisplayPowerController: Animating brightness: target=21, rate=40",
            "12-17 19:31:36.264  1795  1825 I PowerManager_screenOn: DisplayPowerController updatePowerState mPendingRequestLocked=policy=BRIGHT, useProximitySensor=true",
            "12-17 19:31:36.264  1795  2750 I PowerManager_screenOn: DisplayPowerState Updating screen state: state=ON, backlight=823"
        ]
    return logs


def print_banner(title: str):
    """Print a styled banner"""
    print("\n" + "="*100)
    print(f"  {title}")
    print("="*100)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'─'*100}")
    print(f"  {title}")
    print(f"{'─'*100}")


def display_results(final_state: dict):
    """Display comprehensive results from the workflow"""
    
    print_banner("🎯 MULTI-AGENT LOG PARSING SYSTEM - EXECUTION COMPLETE 🎯")
    
    # Agent Execution Timeline
    print_section("📊 AGENT EXECUTION TIMELINE")
    agents = final_state.get("agents_executed", [])
    execution_times = final_state.get("execution_times", {})
    
    print(f"\n{'Agent Name':<40} {'Execution Time (s)':<20} {'Status':<20}")
    print("─" * 80)
    for agent in agents:
        exec_time = execution_times.get(agent, 0.0)
        print(f"{agent:<40} {exec_time:<20.3f} {'✓ Completed':<20}")
    
    total_time = sum(execution_times.values())
    print("─" * 80)
    print(f"{'TOTAL PIPELINE EXECUTION':<40} {total_time:<20.3f} {'✓ Success':<20}")
    
    # Pattern Analysis Results
    print_section("🔍 SEMANTIC PATTERN ANALYSIS RESULTS")
    patterns = final_state.get("identified_patterns", [])
    pattern_confidence = final_state.get("pattern_confidence", 0.0)
    
    print(f"\nPattern Confidence Score: {pattern_confidence:.2%}")
    print(f"\nIdentified Patterns ({len(patterns)}):")
    for i, pattern in enumerate(patterns, 1):
        print(f"  {i}. {pattern.get('pattern_type', 'unknown')}")
        print(f"     Confidence: {pattern.get('confidence', 0.0):.2%}")
        print(f"     Description: {pattern.get('description', 'N/A')}")
    
    # Template Synthesis Results
    print_section("📝 TEMPLATE SYNTHESIS RESULTS")
    templates = final_state.get("generated_templates", [])
    template_scores = final_state.get("template_quality_scores", [])
    
    print(f"\nGenerated Templates: {len(templates)}")
    print(f"Method: {final_state.get('template_generation_method', 'N/A')}")
    if template_scores:
        avg_quality = sum(template_scores) / len(template_scores)
        print(f"Average Template Quality: {avg_quality:.2%}")
    
    print(f"\nSample Templates (first 5):")
    for i, template in enumerate(templates[:5], 1):
        quality = template_scores[i-1] if i-1 < len(template_scores) else 0.0
        print(f"  {i}. {template}")
        print(f"     Quality Score: {quality:.2%}")
    
    # Variable Extraction Results
    print_section("🔢 VARIABLE EXTRACTION RESULTS")
    variables = final_state.get("extracted_variables", [])
    var_types = final_state.get("variable_types", [])
    extraction_confidence = final_state.get("extraction_confidence", [])
    
    total_vars = sum(len(v) for v in variables)
    avg_vars_per_log = total_vars / max(len(variables), 1)
    avg_confidence = sum(extraction_confidence) / max(len(extraction_confidence), 1)
    
    print(f"\nTotal Variables Extracted: {total_vars}")
    print(f"Average Variables per Log: {avg_vars_per_log:.2f}")
    print(f"Average Extraction Confidence: {avg_confidence:.2%}")
    
    # Count variable types
    all_types = [t for types_list in var_types for t in types_list]
    type_counts = {}
    for vtype in all_types:
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
    
    if type_counts:
        print(f"\nVariable Type Distribution:")
        for vtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {vtype:<20} {count:>5} ({count/total_vars*100:.1f}%)")
    
    # Quality Assurance Results
    print_section("✅ QUALITY ASSURANCE RESULTS")
    quality_validated = final_state.get("quality_validated", False)
    quality_score = final_state.get("quality_score", 0.0)
    validation_errors = final_state.get("validation_errors", [])
    recommendations = final_state.get("quality_recommendations", [])
    
    status_symbol = "✓" if quality_validated else "✗"
    status_text = "PASSED" if quality_validated else "FAILED"
    
    print(f"\nValidation Status: {status_symbol} {status_text}")
    print(f"Quality Score: {quality_score:.2%}")
    
    if validation_errors:
        print(f"\nValidation Errors ({len(validation_errors)}):")
        for error in validation_errors:
            print(f"  ⚠️  {error}")
    else:
        print("\n✓ No validation errors detected")
    
    if recommendations:
        print(f"\nRecommendations ({len(recommendations)}):")
        for rec in recommendations:
            print(f"  💡 {rec}")
    
    # Final Metrics - Log Parsing
    print_section("📈 LOG PARSING EVALUATION METRICS")
    accuracy = final_state.get("accuracy", 0.0)
    precision = final_state.get("precision", 0.0)
    recall = final_state.get("recall", 0.0)
    f1_score = final_state.get("f1_score", 0.0)
    total_logs = final_state.get("total_logs", 0)
    
    print(f"\n{'Metric':<30} {'Value':<20} {'Grade':<20}")
    print("─" * 70)
    print(f"{'Total Logs Processed':<30} {total_logs:<20} {'':<20}")
    print(f"{'Accuracy':<30} {accuracy:<20.4f} {get_grade(accuracy):<20}")
    print(f"{'Precision':<30} {precision:<20.4f} {get_grade(precision):<20}")
    print(f"{'Recall':<30} {recall:<20.4f} {get_grade(recall):<20}")
    print(f"{'F1-Score':<30} {f1_score:<20.4f} {get_grade(f1_score):<20}")
    
    # Classification/Anomaly Detection Results
    print_section("🔍 CLASSIFICATION / ANOMALY DETECTION RESULTS")
    classifications = final_state.get("classifications", [])
    classification_confidences = final_state.get("classification_confidences", [])
    classification_explanations = final_state.get("classification_explanations", [])
    normal_count = final_state.get("normal_count", 0)
    abnormal_count = final_state.get("abnormal_count", 0)
    
    print(f"\nClassification Summary:")
    print(f"  Normal Logs: {normal_count} ({normal_count/max(total_logs,1)*100:.1f}%)")
    print(f"  Abnormal Logs: {abnormal_count} ({abnormal_count/max(total_logs,1)*100:.1f}%)")
    print(f"  Method: {final_state.get('classification_method', 'N/A')}")
    
    print(f"\nSample Classifications (first 10):")
    for i in range(min(10, len(classifications))):
        status_icon = "🔴" if classifications[i] == "abnormal" else "🟢"
        confidence = classification_confidences[i] if i < len(classification_confidences) else 0.0
        explanation = classification_explanations[i] if i < len(classification_explanations) else "N/A"
        log_msg = final_state.get("raw_logs", [])[i] if i < len(final_state.get("raw_logs", [])) else "N/A"
        
        print(f"\n  [{i+1}] {status_icon} {classifications[i].upper()}")
        print(f"      Log: {log_msg[:70]}...")
        print(f"      Confidence: {confidence:.2f}")
        print(f"      Explanation: {explanation[:80]}...")
    
    # Classification Metrics
    print_section("📊 CLASSIFICATION EVALUATION METRICS")
    classification_accuracy = final_state.get("classification_accuracy", 0.0)
    classification_precision = final_state.get("classification_precision", 0.0)
    classification_recall = final_state.get("classification_recall", 0.0)
    classification_f1_score = final_state.get("classification_f1_score", 0.0)
    
    print(f"\n{'Metric':<30} {'Value':<20} {'Grade':<20}")
    print("─" * 70)
    print(f"{'Classification Accuracy':<30} {classification_accuracy:<20.4f} {get_grade(classification_accuracy):<20}")
    print(f"{'Classification Precision':<30} {classification_precision:<20.4f} {get_grade(classification_precision):<20}")
    print(f"{'Classification Recall':<30} {classification_recall:<20.4f} {get_grade(classification_recall):<20}")
    print(f"{'Classification F1-Score':<30} {classification_f1_score:<20.4f} {get_grade(classification_f1_score):<20}")
    
    # System Performance Summary
    print_section("⚡ SYSTEM PERFORMANCE SUMMARY")
    print(f"\nProcessing Rate: {total_logs / total_time:.2f} logs/second")
    print(f"Average Time per Log: {(total_time / total_logs) * 1000:.2f} ms")
    print(f"Parallel Efficiency: {'Enabled (Template Synthesis & Variable Extraction)':}")
    print(f"Pipeline Stages: 7 specialized agents (including Classification)")
    print(f"Architecture: LangGraph Multi-Agent System")
    
    # Sample Parsed Results
    print_section("📋 SAMPLE PARSED RESULTS (First 3)")
    parsed_results = final_state.get("parsed_results", [])
    
    for i, result in enumerate(parsed_results[:3], 1):
        print(f"\n[Sample {i}]")
        print(f"Original Log:")
        print(f"  {result.get('log_message', 'N/A')}")
        print(f"Generated Template:")
        print(f"  {result.get('template', 'N/A')}")
        print(f"Extracted Variables: {len(result.get('variables', {}))}")
        print(f"Predicted: {result.get('predicted_var_count', 0)} | Estimated Actual: {result.get('estimated_actual_vars', 0)}")
    
    print_banner("✨ MULTI-AGENT WORKFLOW COMPLETED SUCCESSFULLY ✨")


def get_grade(score: float) -> str:
    """Get letter grade for a metric score"""
    if score >= 0.9:
        return "A+ (Excellent)"
    elif score >= 0.8:
        return "A (Very Good)"
    elif score >= 0.7:
        return "B+ (Good)"
    elif score >= 0.6:
        return "B (Satisfactory)"
    elif score >= 0.5:
        return "C (Fair)"
    else:
        return "D (Needs Improvement)"


def main():
    """Main execution function"""
    
    print_banner("🚀 MULTI-AGENT LOG PARSING SYSTEM 🚀")
    print("\nPowered by:")
    print("  • LangGraph - Multi-Agent Orchestration Framework")
    print("  • Google Gemini - Large Language Model")
    print("  • 6 Specialized AI Agents working in parallel and sequential coordination")
    
    # Get API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your-api-key-here":
        print("\n❌ Error: Please set GOOGLE_API_KEY environment variable")
        return
    
    # Load logs
    print_section("📂 LOADING ANDROID LOGS")
    logs = load_android_logs(max_logs=100)
    print(f"✓ Loaded {len(logs)} Android log entries")
    
    # Create workflow
    print_section("⚙️ INITIALIZING MULTI-AGENT WORKFLOW")
    print("\nCreating agent pipeline:")
    print("  1. Log Ingestion & Validator Agent")
    print("  2. Semantic Pattern Analyzer Agent")
    print("  3. Template Synthesizer Agent (Parallel)")
    print("  4. Variable Extractor Agent (Parallel)")
    print("  5. Quality Assurance Agent")
    print("  6. Metrics Orchestrator Agent")
    print("  7. Classification/Anomaly Detection Agent ⭐ NEW")
    
    workflow = create_workflow(api_key=api_key)
    print("\n✓ Multi-agent workflow initialized successfully")
    
    # Print workflow diagram and save as PNG
    print_section("📊 WORKFLOW ARCHITECTURE")
    workflow.visualize_workflow(save_png=True, output_dir="results")
    
    # Execute workflow
    print_section("🔄 EXECUTING MULTI-AGENT PIPELINE")
    print("\nStarting distributed agent execution...")
    print("(This may take several minutes due to LLM API calls and 8-second delays)\n")
    
    start_time = time.time()
    
    try:
        final_state = workflow.execute(logs)
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        print(f"\n✓ Pipeline execution completed in {total_execution_time:.2f} seconds")
        
        # Display results
        display_results(final_state)
        
        # Save results
        print_section("💾 SAVING RESULTS")
        output_file = "multiagent_results.json"
        
        # Prepare serializable results
        serializable_state = {
            k: v for k, v in final_state.items()
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_state, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")
        
        # Generate visualizations and paper-ready results
        print_section("📊 GENERATING VISUALIZATIONS AND PAPER-READY RESULTS")
        
        # Paper results (from Table 3 - average of BGL and Spirit)
        paper_classification_results = {
            'precision': 0.270,  # Average of 0.249 and 0.290
            'recall': 0.917,     # Average of 0.834 and 0.999
            'f1_score': 0.417,   # Average of 0.384 and 0.450
            'parsing_f1': 0.819  # Android F1 from paper Table 2
        }
        
        visualizer = ResultVisualizer(output_dir="results")
        visualizer.generate_comprehensive_report(
            our_results=final_state,
            paper_results=paper_classification_results,
            execution_times=final_state.get("execution_times", {})
        )
        
        # Print summary table
        print_section("📋 SUMMARY TABLE FOR PAPER")
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON TABLE")
        print("="*80)
        print(f"{'Metric':<35} {'Our Implementation':<25} {'Paper (LogPrompt)':<25}")
        print("-" * 80)
        print(f"{'Classification Precision':<35} {final_state.get('classification_precision', 0):<25.4f} {paper_classification_results['precision']:<25.4f}")
        print(f"{'Classification Recall':<35} {final_state.get('classification_recall', 0):<25.4f} {paper_classification_results['recall']:<25.4f}")
        print(f"{'Classification F1-Score':<35} {final_state.get('classification_f1_score', 0):<25.4f} {paper_classification_results['f1_score']:<25.4f}")
        print(f"{'Parsing Accuracy':<35} {final_state.get('accuracy', 0):<25.4f} {'N/A':<25}")
        print(f"{'Parsing Precision':<35} {final_state.get('precision', 0):<25.4f} {'N/A':<25}")
        print(f"{'Parsing Recall':<35} {final_state.get('recall', 0):<25.4f} {'N/A':<25}")
        print(f"{'Parsing F1-Score':<35} {final_state.get('f1_score', 0):<25.4f} {paper_classification_results['parsing_f1']:<25.4f}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

