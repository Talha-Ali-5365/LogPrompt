"""
Visualization and Reporting Module for LogPrompt Results
Generates graphs, tables, and paper-ready outputs
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional
from datetime import datetime


# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class ResultVisualizer:
    """Generate visualizations and reports for LogPrompt results"""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_classification_comparison(self, our_results: Dict, paper_results: Dict, save_path: Optional[str] = None):
        """
        Create comparison bar chart: Our Results vs Paper Results
        
        Args:
            our_results: Dict with 'precision', 'recall', 'f1_score'
            paper_results: Dict with 'precision', 'recall', 'f1_score' (average from paper)
            save_path: Optional path to save figure
        """
        metrics = ['Precision', 'Recall', 'F1-Score']
        our_values = [
            our_results.get('classification_precision', 0),
            our_results.get('classification_recall', 0),
            our_results.get('classification_f1_score', 0)
        ]
        paper_values = [
            paper_results.get('precision', 0),
            paper_results.get('recall', 0),
            paper_results.get('f1_score', 0)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, our_values, width, label='Our Implementation', 
                       color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, paper_values, width, label='Paper (LogPrompt)', 
                       color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Classification Performance: Our Implementation vs Paper', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1.1])
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'classification_comparison_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison chart to: {save_path}")
        plt.close()
    
    def plot_parsing_comparison(self, our_results: Dict, paper_results: Dict, save_path: Optional[str] = None):
        """
        Create comparison bar chart for log parsing metrics
        
        Args:
            our_results: Dict with 'accuracy', 'precision', 'recall', 'f1_score'
            paper_results: Dict with 'f1_score' (from paper Table 2)
            save_path: Optional path to save figure
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        our_values = [
            our_results.get('accuracy', 0),
            our_results.get('precision', 0),
            our_results.get('recall', 0),
            our_results.get('f1_score', 0)
        ]
        
        # Paper only reports F1-score for parsing, so we'll show that
        paper_f1 = paper_results.get('f1_score', 0)
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x[:3], our_values[:3], width, label='Our Implementation', 
                       color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
        bar_f1_ours = ax.bar(x[3] - width/2, our_values[3], width, color='#2E86AB', 
                             alpha=0.8, edgecolor='black', linewidth=1.2)
        bar_f1_paper = ax.bar(x[3] + width/2, paper_f1, width, label='Paper (LogPrompt)', 
                              color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels - bars1 is a BarContainer, iterate over its patches
        for patch in bars1.patches:
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Handle single bars - extract the rectangle patch from container
        # bar_f1_ours and bar_f1_paper are BarContainers, get the first patch
        if len(bar_f1_ours.patches) > 0:
            patch = bar_f1_ours.patches[0]
            height = patch.get_height()
            x_pos = patch.get_x() + patch.get_width()/2.
            ax.text(x_pos, height, f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        if len(bar_f1_paper.patches) > 0:
            patch = bar_f1_paper.patches[0]
            height = patch.get_height()
            x_pos = patch.get_x() + patch.get_width()/2.
            ax.text(x_pos, height, f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Log Parsing Performance: Our Implementation vs Paper', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1.1])
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'parsing_comparison_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved parsing comparison chart to: {save_path}")
        plt.close()
    
    def plot_agent_execution_times(self, execution_times: Dict, save_path: Optional[str] = None):
        """
        Create bar chart showing execution time for each agent
        
        Args:
            execution_times: Dict mapping agent names to execution times (seconds)
            save_path: Optional path to save figure
        """
        agents = list(execution_times.keys())
        times = list(execution_times.values())
        
        # Sort by execution time (descending)
        sorted_data = sorted(zip(agents, times), key=lambda x: x[1], reverse=True)
        agents, times = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
        bars = ax.barh(agents, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(time + max(times)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{time:.2f}s', ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Execution Time (seconds)', fontweight='bold')
        ax.set_ylabel('Agent', fontweight='bold')
        ax.set_title('Agent Execution Times', fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'execution_times_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved execution times chart to: {save_path}")
        plt.close()
    
    def plot_classification_distribution(self, normal_count: int, abnormal_count: int, save_path: Optional[str] = None):
        """
        Create pie chart showing classification distribution
        
        Args:
            normal_count: Number of normal logs
            abnormal_count: Number of abnormal logs
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        sizes = [normal_count, abnormal_count]
        labels = [f'Normal\n({normal_count})', f'Abnormal\n({abnormal_count})']
        colors = ['#4CAF50', '#F44336']
        explode = (0.05, 0.1)  # Explode abnormal slice
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90,
                                          textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Log Classification Distribution', fontweight='bold', pad=20, fontsize=14)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'classification_distribution_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved classification distribution chart to: {save_path}")
        plt.close()
    
    def plot_metrics_radar(self, our_results: Dict, paper_results: Dict, save_path: Optional[str] = None):
        """
        Create radar chart comparing all metrics
        
        Args:
            our_results: Dict with all metrics
            paper_results: Dict with paper metrics
            save_path: Optional path to save figure
        """
        # Prepare data
        categories = ['Precision\n(Classification)', 'Recall\n(Classification)', 'F1-Score\n(Classification)',
                     'F1-Score\n(Parsing)', 'Accuracy\n(Parsing)']
        
        our_values = [
            our_results.get('classification_precision', 0),
            our_results.get('classification_recall', 0),
            our_results.get('classification_f1_score', 0),
            our_results.get('f1_score', 0),
            our_results.get('accuracy', 0)
        ]
        
        paper_values = [
            paper_results.get('precision', 0),
            paper_results.get('recall', 0),
            paper_results.get('f1_score', 0),
            paper_results.get('parsing_f1', 0.819),  # Android F1 from paper
            paper_results.get('parsing_accuracy', 0)  # Not reported in paper
        ]
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values
        our_values += our_values[:1]
        paper_values += paper_values[:1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, our_values, 'o-', linewidth=2, label='Our Implementation', color='#2E86AB')
        ax.fill(angles, our_values, alpha=0.25, color='#2E86AB')
        
        ax.plot(angles, paper_values, 'o-', linewidth=2, label='Paper (LogPrompt)', color='#A23B72')
        ax.fill(angles, paper_values, alpha=0.25, color='#A23B72')
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)
        
        ax.set_title('Comprehensive Performance Comparison\n(Our Implementation vs Paper)', 
                    fontweight='bold', pad=30, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'radar_chart_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved radar chart to: {save_path}")
        plt.close()
    
    def generate_latex_table(self, our_results: Dict, paper_results: Dict, save_path: Optional[str] = None):
        """
        Generate LaTeX table comparing our results with paper
        
        Args:
            our_results: Dict with all our metrics
            paper_results: Dict with paper metrics
            save_path: Optional path to save LaTeX file
        """
        latex_content = """\\begin{{table}}[h]
\\centering
\\caption{{Performance Comparison: Our Implementation vs Paper (LogPrompt)}}
\\label{{tab:comparison}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Our Implementation}} & \\textbf{{Paper (LogPrompt)}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Classification/Anomaly Detection}}}} \\\\
\\midrule
Precision & {:.3f} & {:.3f} \\\\
Recall & {:.3f} & {:.3f} \\\\
F1-Score & {:.3f} & {:.3f} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Log Parsing}}}} \\\\
\\midrule
Accuracy & {:.3f} & N/A \\\\
Precision & {:.3f} & N/A \\\\
Recall & {:.3f} & N/A \\\\
F1-Score & {:.3f} & {:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""".format(
            our_results.get('classification_precision', 0),
            paper_results.get('precision', 0),
            our_results.get('classification_recall', 0),
            paper_results.get('recall', 0),
            our_results.get('classification_f1_score', 0),
            paper_results.get('f1_score', 0),
            our_results.get('accuracy', 0),
            our_results.get('precision', 0),
            our_results.get('recall', 0),
            our_results.get('f1_score', 0),
            paper_results.get('parsing_f1', 0.819)
        )
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'comparison_table_{self.timestamp}.tex')
        
        with open(save_path, 'w') as f:
            f.write(latex_content)
        
        print(f"✓ Saved LaTeX table to: {save_path}")
        return latex_content
    
    def generate_csv_results(self, our_results: Dict, paper_results: Dict, save_path: Optional[str] = None):
        """
        Generate CSV file with all results
        
        Args:
            our_results: Dict with all our metrics
            paper_results: Dict with paper metrics
            save_path: Optional path to save CSV file
        """
        data = {
            'Metric': [
                'Classification Precision',
                'Classification Recall',
                'Classification F1-Score',
                'Parsing Accuracy',
                'Parsing Precision',
                'Parsing Recall',
                'Parsing F1-Score'
            ],
            'Our Implementation': [
                our_results.get('classification_precision', 0),
                our_results.get('classification_recall', 0),
                our_results.get('classification_f1_score', 0),
                our_results.get('accuracy', 0),
                our_results.get('precision', 0),
                our_results.get('recall', 0),
                our_results.get('f1_score', 0)
            ],
            'Paper (LogPrompt)': [
                paper_results.get('precision', 0),
                paper_results.get('recall', 0),
                paper_results.get('f1_score', 0),
                'N/A',
                'N/A',
                'N/A',
                paper_results.get('parsing_f1', 0.819)
            ]
        }
        
        df = pd.DataFrame(data)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'results_{self.timestamp}.csv')
        
        df.to_csv(save_path, index=False)
        print(f"✓ Saved CSV results to: {save_path}")
        return df
    
    def generate_comprehensive_report(self, our_results: Dict, paper_results: Dict, execution_times: Dict):
        """
        Generate all visualizations and reports
        
        Args:
            our_results: Dict with all our metrics
            paper_results: Dict with paper metrics
            execution_times: Dict with agent execution times
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE RESULTS AND VISUALIZATIONS")
        print("="*80)
        
        # Generate all visualizations
        self.plot_classification_comparison(our_results, paper_results)
        self.plot_parsing_comparison(our_results, paper_results)
        self.plot_agent_execution_times(execution_times)
        self.plot_classification_distribution(
            our_results.get('normal_count', 0),
            our_results.get('abnormal_count', 0)
        )
        self.plot_metrics_radar(our_results, paper_results)
        
        # Generate tables
        self.generate_latex_table(our_results, paper_results)
        self.generate_csv_results(our_results, paper_results)
        
        print("\n" + "="*80)
        print("ALL RESULTS GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {self.timestamp}")

