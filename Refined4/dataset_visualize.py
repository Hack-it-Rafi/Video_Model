# filepath: f:\Attck\Label\POC\Refined4\dataset_visualize.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from config import DATA_ROOT

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_all_annotations():
    """Load all annotations from all video folders"""
    all_data = []
    video_ids = []
    
    for video_dir in sorted(os.listdir(DATA_ROOT)):
        if video_dir.startswith('videos_'):
            annot_path = os.path.join(DATA_ROOT, video_dir, 'annotations.csv')
            if os.path.exists(annot_path):
                df = pd.read_csv(annot_path)
                df.columns = df.columns.str.strip()
                df['target_app'] = df['target_app'].fillna('none')
                df['video_id'] = video_dir
                
                # Extract chunk number
                df['chunk_num'] = df['filename'].str.extract(r'chunk_(\d+)').astype(int)
                all_data.append(df)
                video_ids.append(video_dir)
    
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df, video_ids

def plot_class_distribution(df, output_dir='visualizations'):
    """Plot action and app class distributions"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Action distribution
    action_counts = df['action'].value_counts()
    axes[0, 0].barh(range(len(action_counts)), action_counts.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(action_counts)))
    axes[0, 0].set_yticklabels(action_counts.index, fontsize=8)
    axes[0, 0].set_xlabel('Count', fontsize=10)
    axes[0, 0].set_title(f'Action Distribution (Total: {len(action_counts)} classes)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # App distribution
    app_counts = df['target_app'].value_counts()
    axes[0, 1].barh(range(len(app_counts)), app_counts.values, color='coral')
    axes[0, 1].set_yticks(range(len(app_counts)))
    axes[0, 1].set_yticklabels(app_counts.index, fontsize=10)
    axes[0, 1].set_xlabel('Count', fontsize=10)
    axes[0, 1].set_title(f'App Distribution (Total: {len(app_counts)} classes)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Top 15 actions (pie chart)
    top_actions = action_counts.head(15)
    other_count = action_counts[15:].sum() if len(action_counts) > 15 else 0
    if other_count > 0:
        top_actions['Others'] = other_count
    
    axes[1, 0].pie(top_actions.values, labels=top_actions.index, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Top 15 Actions Distribution', fontsize=12, fontweight='bold')
    
    # App distribution (pie chart)
    axes[1, 1].pie(app_counts.values, labels=app_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('App Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/class_distribution.png")
    plt.close()

def plot_imbalance_analysis(df, output_dir='visualizations'):
    """Analyze and visualize class imbalance"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    
    # Action imbalance (sorted by count)
    action_counts = df['action'].value_counts().sort_values(ascending=True)
    colors = ['red' if count < 10 else 'orange' if count < 50 else 'green' for count in action_counts.values]
    
    axes[0].barh(range(len(action_counts)), action_counts.values, color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(action_counts)))
    axes[0].set_yticklabels(action_counts.index, fontsize=7)
    axes[0].set_xlabel('Sample Count', fontsize=10)
    axes[0].set_title('Action Class Imbalance (Red: <10, Orange: 10-50, Green: >50 samples)', 
                      fontsize=12, fontweight='bold')
    axes[0].axvline(x=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Critical (<10)')
    axes[0].axvline(x=50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Warning (<50)')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # App imbalance
    app_counts = df['target_app'].value_counts().sort_values(ascending=True)
    colors_app = ['red' if count < 10 else 'orange' if count < 50 else 'green' for count in app_counts.values]
    
    axes[1].barh(range(len(app_counts)), app_counts.values, color=colors_app, alpha=0.7)
    axes[1].set_yticks(range(len(app_counts)))
    axes[1].set_yticklabels(app_counts.index, fontsize=10)
    axes[1].set_xlabel('Sample Count', fontsize=10)
    axes[1].set_title('App Class Imbalance', fontsize=12, fontweight='bold')
    axes[1].axvline(x=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axvline(x=50, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_imbalance.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/class_imbalance.png")
    plt.close()

def plot_temporal_distribution(df, output_dir='visualizations'):
    """Analyze temporal patterns in the data"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Chunks per video
    chunks_per_video = df.groupby('video_id').size().sort_values(ascending=False)
    axes[0, 0].bar(range(len(chunks_per_video)), chunks_per_video.values, color='teal')
    axes[0, 0].set_xticks(range(len(chunks_per_video)))
    axes[0, 0].set_xticklabels(chunks_per_video.index, rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('Number of Chunks', fontsize=10)
    axes[0, 0].set_title('Chunks per Video', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Action timeline for each video (heatmap)
    action_timeline = []
    video_labels = []
    for video_id in sorted(df['video_id'].unique()):
        video_df = df[df['video_id'] == video_id].sort_values('chunk_num')
        action_timeline.append(video_df['action'].tolist())
        video_labels.append(video_id)
    
    # Find max length and pad
    max_len = max(len(row) for row in action_timeline)
    
    # Create action to numeric mapping
    unique_actions = sorted(df['action'].unique())
    action_to_num = {action: idx for idx, action in enumerate(unique_actions)}
    
    timeline_matrix = np.zeros((len(action_timeline), max_len))
    for i, actions in enumerate(action_timeline):
        for j, action in enumerate(actions):
            timeline_matrix[i, j] = action_to_num[action]
    
    # Plot heatmap
    im = axes[0, 1].imshow(timeline_matrix, aspect='auto', cmap='tab20', interpolation='nearest')
    axes[0, 1].set_yticks(range(len(video_labels)))
    axes[0, 1].set_yticklabels(video_labels, fontsize=9)
    axes[0, 1].set_xlabel('Chunk Number', fontsize=10)
    axes[0, 1].set_title('Action Timeline Across Videos', fontsize=12, fontweight='bold')
    
    # Chunk distribution histogram
    all_chunk_nums = df['chunk_num'].values
    axes[1, 0].hist(all_chunk_nums, bins=50, color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('Chunk Number', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Chunk Number Distribution (Gaps indicate missing chunks)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Action sequence length distribution
    video_lengths = df.groupby('video_id').size()
    axes[1, 1].hist(video_lengths.values, bins=20, color='mediumpurple', edgecolor='black')
    axes[1, 1].set_xlabel('Number of Chunks per Video', fontsize=10)
    axes[1, 1].set_ylabel('Number of Videos', fontsize=10)
    axes[1, 1].set_title('Video Length Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/temporal_distribution.png")
    plt.close()

def plot_action_app_cooccurrence(df, output_dir='visualizations'):
    """Analyze co-occurrence patterns between actions and apps"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create co-occurrence matrix
    action_app_pairs = df.groupby(['action', 'target_app']).size().reset_index(name='count')
    pivot_table = action_app_pairs.pivot_table(index='action', columns='target_app', values='count', fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 16))
    sns.heatmap(pivot_table, annot=False, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Action-App Co-occurrence Matrix', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Target App', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_app_cooccurrence.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/action_app_cooccurrence.png")
    plt.close()

def generate_statistics_report(df, video_ids, output_dir='visualizations'):
    """Generate detailed text statistics report"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = []
    report.append("=" * 80)
    report.append("DATASET QUALITY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("📊 OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total videos: {len(video_ids)}")
    report.append(f"Total chunks: {len(df)}")
    report.append(f"Unique actions: {df['action'].nunique()}")
    report.append(f"Unique apps: {df['target_app'].nunique()}")
    report.append(f"Average chunks per video: {len(df) / len(video_ids):.2f}")
    report.append("")
    
    # Class imbalance issues
    report.append("⚠️  CLASS IMBALANCE ISSUES")
    report.append("-" * 80)
    action_counts = df['action'].value_counts()
    critical_actions = action_counts[action_counts < 10]
    warning_actions = action_counts[(action_counts >= 10) & (action_counts < 50)]
    
    report.append(f"Critical classes (<10 samples): {len(critical_actions)}")
    if len(critical_actions) > 0:
        report.append("  Actions with <10 samples:")
        for action, count in critical_actions.items():
            report.append(f"    - {action}: {count} samples")
    
    report.append(f"\nWarning classes (10-50 samples): {len(warning_actions)}")
    if len(warning_actions) > 0:
        report.append("  Actions with 10-50 samples:")
        for action, count in warning_actions.head(10).items():
            report.append(f"    - {action}: {count} samples")
    
    report.append("")
    
    # Missing chunks analysis
    report.append("🔍 MISSING CHUNKS ANALYSIS")
    report.append("-" * 80)
    for video_id in sorted(video_ids):
        video_df = df[df['video_id'] == video_id].sort_values('chunk_num')
        chunk_nums = video_df['chunk_num'].values
        
        if len(chunk_nums) > 0:
            expected_range = range(chunk_nums.min(), chunk_nums.max() + 1)
            missing_chunks = set(expected_range) - set(chunk_nums)
            
            if missing_chunks:
                report.append(f"{video_id}:")
                report.append(f"  Total chunks: {len(chunk_nums)}")
                report.append(f"  Missing chunks: {len(missing_chunks)}")
                report.append(f"  Missing: {sorted(list(missing_chunks))[:20]}{'...' if len(missing_chunks) > 20 else ''}")
                report.append(f"  Completeness: {len(chunk_nums) / len(expected_range) * 100:.1f}%")
                report.append("")
    
    # Action distribution per video
    report.append("📹 ACTION DISTRIBUTION PER VIDEO")
    report.append("-" * 80)
    for video_id in sorted(video_ids):
        video_df = df[df['video_id'] == video_id]
        top_actions = video_df['action'].value_counts().head(5)
        report.append(f"{video_id} ({len(video_df)} chunks):")
        for action, count in top_actions.items():
            report.append(f"  - {action}: {count} ({count/len(video_df)*100:.1f}%)")
        report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS FOR DATA QUALITY IMPROVEMENT")
    report.append("-" * 80)
    
    if len(critical_actions) > 0:
        report.append(f"1. CRITICAL: {len(critical_actions)} action classes have <10 samples")
        report.append("   → Consider: Data augmentation, collecting more samples, or merging similar classes")
    
    if len(warning_actions) > 0:
        report.append(f"2. WARNING: {len(warning_actions)} action classes have 10-50 samples")
        report.append("   → May cause poor generalization. Consider collecting more data.")
    
    # Check for imbalance ratio
    max_count = action_counts.max()
    min_count = action_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 100:
        report.append(f"3. SEVERE IMBALANCE: Ratio of {imbalance_ratio:.1f}:1 (max:min)")
        report.append("   → Use class weights, focal loss, or oversampling techniques")
    
    # Missing chunks
    total_expected = sum(df[df['video_id'] == vid]['chunk_num'].max() - 
                        df[df['video_id'] == vid]['chunk_num'].min() + 1 
                        for vid in video_ids if len(df[df['video_id'] == vid]) > 0)
    completeness = len(df) / total_expected * 100 if total_expected > 0 else 0
    
    if completeness < 90:
        report.append(f"4. MISSING DATA: Only {completeness:.1f}% of expected chunks present")
        report.append("   → Review annotation process, check for corrupted videos")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'quality_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Saved: {output_dir}/quality_report.txt")
    print("\n" + report_text)

def main():
    """Main visualization function"""
    print("\n" + "="*80)
    print("DATASET VISUALIZATION AND QUALITY ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    print("Loading annotations...")
    df, video_ids = load_all_annotations()
    print(f"✓ Loaded {len(df)} samples from {len(video_ids)} videos\n")
    
    # Create visualizations
    print("Generating visualizations...\n")
    
    plot_class_distribution(df)
    plot_imbalance_analysis(df)
    plot_temporal_distribution(df)
    plot_action_app_cooccurrence(df)
    
    # Generate report
    print("\nGenerating quality report...\n")
    generate_statistics_report(df, video_ids)
    
    print("\n" + "="*80)
    print("✅ All visualizations and reports generated successfully!")
    print("Check the 'visualizations/' folder for outputs")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
