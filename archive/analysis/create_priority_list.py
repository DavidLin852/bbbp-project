"""
Create a priority-ordered list for experimental validation
"""
import pandas as pd

# Read results
df = pd.read_csv('grouped_analysis_results.csv')

# Define priority scoring
def get_priority_score(row):
    """Higher score = higher priority"""
    score = 0

    # Base score from mean probability
    score += row['Mean_Prob'] * 100

    # Confidence bonus
    if row['Confidence'] == 'Very High':
        score += 20
    elif row['Confidence'] == 'High':
        score += 10
    elif row['Confidence'] == 'Medium':
        score += 0
    else:  # Low
        score -= 10

    # Penalty for high disagreement
    if row['Std_Prob'] > 0.2:
        score -= 15
    elif row['Std_Prob'] > 0.15:
        score -= 5

    return score

df['Priority_Score'] = df.apply(get_priority_score, axis=1)

# Sort by priority score
df_sorted = df.sort_values('Priority_Score', ascending=False)

# Assign priority tier
def get_tier(row):
    if row['Pred_Class'] == 'BBB-':
        return 'SKIP'
    elif row['Confidence'] == 'Very High':
        return 'PRIORITY 1'
    elif row['Confidence'] == 'High':
        return 'PRIORITY 2'
    elif row['Std_Prob'] > 0.15:
        return 'PRIORITY 3 - CAUTION'
    else:
        return 'PRIORITY 3'

df_sorted['Tier'] = df_sorted.apply(get_tier, axis=1)

print("="*140)
print(" "*50 + "PRIORITY ORDER FOR EXPERIMENTAL VALIDATION")
print("="*140)

print("\n")
print("Legend:")
print("  PRIORITY 1: Very High Confidence - All models strongly agree")
print("  PRIORITY 2: High Confidence - Most models agree")
print("  PRIORITY 3: Medium Confidence - Some disagreement, use with caution")
print("  SKIP: Predicted BBB- - Not recommended")
print("\n")

for tier in ['PRIORITY 1', 'PRIORITY 2', 'PRIORITY 3', 'PRIORITY 3 - CAUTION', 'SKIP']:
    tier_df = df_sorted[df_sorted['Tier'] == tier]

    if len(tier_df) == 0:
        continue

    if 'SKIP' in tier:
        print("="*140)
        print(f"  {tier}")
        print("="*140)
    else:
        print("\n" + "="*140)
        print(f"  {tier}")
        print("="*140)

    print(f"\n{'Rank':<6} {'Name':<18} {'Group':<35} {'Mean':<8} {'Std':<8} {'GAT':<8} {'XGB':<8} {'LGBM':<8} {'RF':<8} {'Score':<8}")
    print("-"*140)

    for rank, (idx, row) in enumerate(tier_df.iterrows(), 1):
        name = row['Name'][:17]
        group = row['Group'][:34]
        mean = f"{row['Mean_Prob']:.4f}"
        std = f"{row['Std_Prob']:.4f}"
        gat = f"{row['GAT_BBB_Prob.']:.3f}" if not pd.isna(row['GAT_BBB_Prob.']) else 'N/A'
        xgb = f"{row['XGB_BBB_Prob.']:.3f}" if not pd.isna(row['XGB_BBB_Prob.']) else 'N/A'
        lgbm = f"{row['LGBM_BBB_Prob.']:.3f}" if not pd.isna(row['LGBM_BBB_Prob.']) else 'N/A'
        rf = f"{row['RF_BBB_Prob.']:.3f}" if not pd.isna(row['RF_BBB_Prob.']) else 'N/A'
        score = f"{row['Priority_Score']:.1f}"

        print(f"{rank:<6} {name:<18} {group:<35} {mean:<8} {std:<8} {gat:<8} {xgb:<8} {lgbm:<8} {rf:<8} {score:<8}")

# Save priority list
output_cols = ['Tier', 'Priority_Score', 'Name', 'smiles', 'Group', 'GAT_BBB_Prob.',
               'XGB_BBB_Prob.', 'LGBM_BBB_Prob.', 'RF_BBB_Prob.',
               'Mean_Prob', 'Std_Prob', 'Min_Prob', 'Max_Prob',
               'Pred_Class', 'Confidence']
output_df = df_sorted[output_cols]
output_df.to_csv('priority_validation_list.csv', index=False)

print("\n" + "="*140)
print("Priority list saved to: priority_validation_list.csv")
print("="*140)

# Summary statistics
print("\n\nSUMMARY:")
print("-"*140)
for tier in ['PRIORITY 1', 'PRIORITY 2', 'PRIORITY 3', 'PRIORITY 3 - CAUTION', 'SKIP']:
    count = len(df_sorted[df_sorted['Tier'] == tier])
    if count > 0:
        print(f"  {tier:<25}: {count:>3} molecules")

print("\n\nRECOMMENDATION:")
print("-"*140)
print("  1. Start with PRIORITY 1 molecules (4 compounds) - highest confidence of success")
print("  2. Proceed to PRIORITY 2 molecules (4 compounds) - still very promising")
print("  3. Consider PRIORITY 3 molecules (9 compounds) - validate carefully, may need additional analysis")
print("  4. SKIP molecules (7 compounds) - predicted as BBB-, not recommended for BBB applications")
print("="*140)
