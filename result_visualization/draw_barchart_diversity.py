import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the all_runs_summary CSV file
df = pd.read_csv('all_runs_summary.csv')

# Filter data by evaluation mode
zero_shot_df = df[df['eval_mode'] == 'zero_shot'][['model_key', 'diversity_score_per_sample_avg']].set_index('model_key')
few_shot_df = df[df['eval_mode'] == 'few_shot'][['model_key', 'diversity_score_per_sample_avg']].set_index('model_key')
fine_tune_df = df[df['eval_mode'] == 'fine_tune'][['model_key', 'diversity_score_per_sample_avg']].set_index('model_key')

# Get all unique models
models = sorted(df['model_key'].unique())

# Prepare data for plotting
zero_shot_scores = [zero_shot_df.loc[model, 'diversity_score_per_sample_avg'] if model in zero_shot_df.index else 0 for model in models]
few_shot_scores = [few_shot_df.loc[model, 'diversity_score_per_sample_avg'] if model in few_shot_df.index else 0 for model in models]
fine_tune_scores = [fine_tune_df.loc[model, 'diversity_score_per_sample_avg'] if model in fine_tune_df.index else 0 for model in models]

# Set up the bar chart
x = np.arange(len(models))  # Label locations
width = 0.25  # Width of bars

fig, ax = plt.subplots(figsize=(14, 6))

# Create bars for each evaluation mode
bars1 = ax.bar(x - width, zero_shot_scores, width, label='Zero-shot', alpha=0.8)
bars2 = ax.bar(x, few_shot_scores, width, label='Few-shot', alpha=0.8)
bars3 = ax.bar(x + width, fine_tune_scores, width, label='Fine-tune', alpha=0.8)

# Customize the chart
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Diversity Score Per Sample', fontsize=12, fontweight='bold')
ax.set_title('Diversity Score Per Sample Comparison Across Models and Evaluation Modes', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add a horizontal line at y=0 for reference
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('diversity_score_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'diversity_score_comparison.png'")

# Display the chart
plt.show()
