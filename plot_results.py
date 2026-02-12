import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("final_results.csv")

# 2. Rename for Publication
df['Method'] = df['Method'].replace({
    'DragonnetWrapper': 'Dragonnet (Deep Learning)',
    'CausalForest': 'Causal Forest (Tree-Based)',
    'LinearDML': 'Linear DML (Baseline)',
    'PSS-WeightedEnsemble': 'PSS-Ensemble (Ours)'
})

# 3. Plotting Grid
sns.set_theme(style="whitegrid", font_scale=1.1)
g = sns.FacetGrid(df, row="Metric", col="Dataset", hue="Method", 
                  sharey=False, height=4, aspect=1.2, margin_titles=True)

# 4. Map Lines
g.map(sns.lineplot, "Noise", "Value", marker="o", linewidth=2.5)

# 5. Make it pretty
g.add_legend(title='Estimator')
g.set_axis_labels("Noise Dimension ($d_{noise}$)", "Score (Lower is Better)")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Figure 1: Null Intervention Stability (PSS) vs. Estimation Error (PEHE)', 
               fontsize=16, fontweight='bold')

plt.savefig("figure1_killshot.png", dpi=300)
print("Figure 1 saved!")
plt.show()