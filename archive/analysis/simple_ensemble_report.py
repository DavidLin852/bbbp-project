"""
简洁的多模型分析报告（避免编码问题）
"""
import pandas as pd
import numpy as np

# 读取预测结果
df = pd.read_csv('ensemble_predictions.csv')

# 设置分子名称
molecule_names = {
    'CC(=O)OC1=CC=CC=C1C(=O)O': 'Aspirin',
    'CC(=O)NC1=CC=C(C=C1)O': 'Paracetamol',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C': 'Caffeine',
    'c1ccccc1': 'Benzene',
    'CCO': 'Ethanol'
}

df['Name'] = df['SMILES'].map(molecule_names)

print("="*100)
print(" "*30 + "Multi-Model BBB Prediction Analysis")
print("="*100)

# 1. Summary Table
print("\n[1] Summary Table")
print("-"*100)
print(f"{'Molecule':<15} {'RF':<8} {'XGB':<8} {'LGBM':<8} {'RF+Sm':<8} {'XGB+Sm':<8} {'LGBM+Sm':<8} {'Mean':<8} {'Std':<8} {'Result':<12}")
print("-"*100)

for idx, row in df.iterrows():
    print(f"{row['Name']:<15} {row['RF']:<8.3f} {row['XGB']:<8.3f} {row['LGBM']:<8.3f} "
          f"{row['RF+SMARTS']:<8.3f} {row['XGB+SMARTS']:<8.3f} {row['LGBM+SMARTS']:<8.3f} "
          f"{row['mean']:<8.3f} {row['std']:<8.3f} {row['prediction']:<12}")

# 2. Ranking by mean probability
print("\n[2] Ranking by Mean Probability (BBB+ likelihood)")
print("-"*100)
df_sorted = df.sort_values('mean', ascending=False)

for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
    conf_symbol = {'极高': '***', '高': '**', '中等': '*', '低': ''}[row['confidence']]
    print(f"{i}. {row['Name']:<15} | Mean: {row['mean']:.4f} | Range: [{row['min']:.3f}, {row['max']:.3f}] | Std: {row['std']:.4f} {conf_symbol}")

print("\nLegend: *** = Very High Confidence, ** = High Confidence, * = Medium Confidence")

# 3. Model consistency analysis
print("\n[3] Model Consistency Analysis")
print("-"*100)

for idx, row in df.iterrows():
    print(f"\n{row['Name']}:")
    print(f"  Mean: {row['mean']:.4f}")
    print(f"  Std: {row['std']:.4f}")

    if row['std'] < 0.05:
        print(f"  Consistency: VERY HIGH - All models agree")
    elif row['std'] < 0.10:
        print(f"  Consistency: HIGH - Most models agree")
    elif row['std'] < 0.20:
        print(f"  Consistency: MEDIUM - Some disagreement")
    else:
        print(f"  Consistency: LOW - High disagreement")

    # Find outlier predictions
    preds = {
        'RF': row['RF'],
        'XGB': row['XGB'],
        'LGBM': row['LGBM'],
        'RF+Sm': row['RF+SMARTS'],
        'XGB+Sm': row['XGB+SMARTS'],
        'LGBM+Sm': row['LGBM+SMARTS']
    }

    max_model = max(preds, key=preds.get)
    min_model = min(preds, key=preds.get)

    print(f"  Max: {max_model} ({preds[max_model]:.4f})")
    print(f"  Min: {min_model} ({preds[min_model]:.4f})")

# 4. Decision recommendations
print("\n[4] Decision Recommendations")
print("-"*100)

high_conf_bbb_plus = df[
    (df['prediction'] == 'BBB+') &
    (df['confidence'].isin(['高', '极高']))
].sort_values('mean', ascending=False)

print("\n[A] High-Confidence BBB+ (Recommended for validation):")
if len(high_conf_bbb_plus) > 0:
    for idx, row in high_conf_bbb_plus.iterrows():
        print(f"  - {row['Name']:<20} | Probability: {row['mean']:.4f} | Confidence: {row['confidence']}")
else:
    print("  None")

medium_conf = df[df['confidence'] == '中等']

print("\n[B] Medium-Confidence (Use with caution):")
if len(medium_conf) > 0:
    for idx, row in medium_conf.iterrows():
        print(f"  - {row['Name']:<20} | Probability: {row['mean']:.4f} | Std: {row['std']:.4f}")
        print(f"    Recommendation: Models disagree, consider additional expert judgment")
else:
    print("  None")

bbb_minus = df[df['prediction'] == 'BBB-']

print("\n[C] Predicted BBB- (Not recommended):")
if len(bbb_minus) > 0:
    for idx, row in bbb_minus.iterrows():
        print(f"  - {row['Name']:<20} | Probability: {row['mean']:.4f}")
else:
    print("  None")

# 5. SMARTS feature impact
print("\n[5] SMARTS Feature Impact Analysis")
print("-"*100)
print("Comparing baseline vs SMARTS-enhanced models:\n")

for idx, row in df.iterrows():
    rf_baseline = row['RF']
    rf_smarts = row['RF+SMARTS']
    rf_diff = rf_smarts - rf_baseline

    xgb_baseline = row['XGB']
    xgb_smarts = row['XGB+SMARTS']
    xgb_diff = xgb_smarts - xgb_baseline

    lgbm_baseline = row['LGBM']
    lgbm_smarts = row['LGBM+SMARTS']
    lgbm_diff = lgbm_smarts - lgbm_baseline

    avg_diff = (rf_diff + xgb_diff + lgbm_diff) / 3

    print(f"{row['Name']}:")
    print(f"  RF:    {rf_baseline:.4f} -> {rf_smarts:.4f} ({rf_diff:+.4f})")
    print(f"  XGB:   {xgb_baseline:.4f} -> {xgb_smarts:.4f} ({xgb_diff:+.4f})")
    print(f"  LGBM:  {lgbm_baseline:.4f} -> {lgbm_smarts:.4f} ({lgbm_diff:+.4f})")
    print(f"  Average SMARTS impact: {avg_diff:+.4f}")

    if abs(avg_diff) > 0.1:
        print(f"  -> SMARTS features have SIGNIFICANT impact on this molecule")
    elif abs(avg_diff) > 0.05:
        print(f"  -> SMARTS features have moderate impact")
    else:
        print(f"  -> SMARTS features have minimal impact")
    print()

# 6. Practical usage guide
print("\n[6] Practical Usage Guide")
print("-"*100)
print("""
For different scenarios, use different columns from the results:

1. Conservative strategy (drug safety assessment):
   - Use 'min' column (minimum prediction across all models)
   - Application: Better to miss a true positive than false positive
   - Example: Toxicity prediction

2. Aggressive strategy (lead compound screening):
   - Use 'max' column (maximum prediction across all models)
   - Application: Better to have false positives than miss compounds
   - Example: Initial high-throughput screening

3. Consensus strategy (standard prediction):
   - Use 'mean' column (average of all models)
   - Application: Balanced approach for standard predictions
   - Example: Regular BBB permeability assessment

4. Confidence-based filtering:
   - Prioritize: High/Very High confidence molecules
   - Use with caution: Medium confidence molecules
   - Investigate: Low confidence molecules (may need expert review)

5. Model interpretation:
   - Compare baseline vs +SMARTS predictions
   - Large differences indicate chemical substructure importance
   - Use for explaining which molecular features affect BBB permeability
""")

print("\n" + "="*100)
print("Analysis Complete!")
print("="*100)

# Save summary to CSV
summary_cols = ['Name', 'SMILES', 'RF', 'XGB', 'LGBM', 'RF+SMARTS', 'XGB+SMARTS',
                'LGBM+SMARTS', 'mean', 'std', 'min', 'max', 'prediction', 'confidence']
df[summary_cols].to_csv('ensemble_predictions_summary.csv', index=False)
print("\nSummary saved to: ensemble_predictions_summary.csv")
