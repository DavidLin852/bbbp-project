# BBB Model Visualization Template - User Guide

## 📋 Overview

This template provides an interactive web-based tool for generating model comparison visualizations. You can manually input data or use preset examples to create publication-quality charts.

## 🚀 Quick Start

### Option 1: Web-Based Tool (Recommended)

1. **Open the template** in your web browser:
   ```
   visualization_template.html
   ```
   - Simply double-click the file to open it in any modern browser
   - No internet connection required (uses local JavaScript)
   - Works offline after initial load

2. **Choose your data source:**
   - **Tab 1 - Data Entry**: Paste CSV data manually or upload a CSV file
   - **Tab 2 - Preset 1**: SMARTS pretraining importance comparison (A,B dataset)
   - **Tab 3 - Preset 2**: Dataset impact analysis (across all datasets)

3. **Generate charts** in Tab 4:
   - Click "Generate All 4 Charts" for automatic generation
   - Or generate individual charts selectively
   - Export charts as PNG files with one click

### Option 2: Python Script

```bash
python visualization_template.py --data data.csv --output-dir outputs/
```

## 📊 Data Format

### CSV Format Requirements

```csv
Dataset,Model,AUC,Precision,Recall,F1
A,RF,0.916,0.957,0.968,0.963
A,RF+SMARTS,0.927,0.903,1.000,0.949
A,B,LGBM,0.964,0.944,0.941,0.943
```

**Required columns:**
- `Dataset`: Dataset identifier (A, A,B, A,B,C, A,B,C,D)
- `Model`: Model name (RF, RF+SMARTS, LGBM, LGBM+SMARTS, XGB, XGB+SMARTS, GAT, GAT+SMARTS)
- `AUC`: Area Under ROC Curve (0.80 - 1.0)
- `Precision`: Precision score (0.80 - 1.0)
- `Recall`: Recall score (0.80 - 1.0)
- `F1`: F1 score (0.80 - 1.0)

**Important:**
- Use commas to separate values
- No header row in CSV input (but you can include one, it will be skipped)
- Decimal values must use dots (not commas)

## 🎯 Preset Examples

### Preset 1: SMARTS Pretraining Importance

**Purpose:** Compare baseline models vs SMARTS-enhanced models on A,B dataset

**Models compared:**
- RF vs RF+SMARTS
- LGBM vs LGBM+SMARTS
- XGB vs XGB+SMARTS
- GAT vs GAT+SMARTS

**Analysis questions:**
- Which models benefit most from SMARTS pretraining?
- What is the performance gain (ΔAUC)?
- Does pretraining help more for certain model types?

**Expected insights:**
- SMARTS features should improve AUC by 0.5% - 3%
- GAT models often show largest improvement from pretraining
- Color coding shows same base color, darker shade for SMARTS version

### Preset 2: Dataset Impact Analysis

**Purpose:** Track SMARTS-enhanced models across different dataset sizes

**Datasets compared:**
- A (106 samples) → A,B (496) → A,B,C (1060) → A,B,C,D (6296)

**Models compared:**
- RF+SMARTS, LGBM+SMARTS, XGB+SMARTS, GAT+SMARTS

**Analysis questions:**
- How does performance scale with dataset size?
- Do SMARTS features help more on small or large datasets?
- Which models maintain performance as data grows?

**Expected insights:**
- Performance usually improves with more data
- SMARTS enhancement often most valuable on smaller datasets
- Large datasets may reduce relative benefit of feature engineering

## 📈 Generated Charts

### Chart 1: Performance Metrics Heatmap

**Layout:** 2×2 grid showing AUC, Precision, Recall, F1

**Features:**
- Color coding from red (low) to green (high)
- All models across all datasets
- Numerical values displayed (3 decimal places)

**Use case:** Quick visual comparison of all metrics

### Chart 2: AUC Score Comparison (Barplot)

**Layout:** Grouped bar chart, one group per dataset

**Features:**
- Models grouped by dataset
- Color-coded by model type
- AUC values labeled on bars

**Use case:** Direct AUC comparison across models and datasets

### Chart 3: AUC vs F1 Scatter Plot

**Layout:** Scatter plot with AUC on x-axis, F1 on y-axis

**Features:**
- Each marker represents one model on one dataset
- Different shapes for different model types
- Diagonal reference line (AUC = F1)
- Hover to see dataset details

**Use case:** Understanding performance trade-offs

**Interpretation:**
- **Above diagonal:** F1 > AUC (better balance)
- **Below diagonal:** AUC > F1 (ranking optimized)
- **On diagonal:** Perfect balance

### Chart 4: Dataset Complexity Impact

**Layout:** 4 line plots, one per metric

**Features:**
- X-axis: Dataset size (with sample counts)
- Y-axis: Performance metric (0.80 - 1.0)
- Different line styles for baseline/SMARTS/GAT models
- Trend lines show scaling behavior

**Use case:** Understanding how models scale with data

## 🎨 Color Scheme

The template uses a consistent color scheme:

| Model | Color | Hex Code |
|-------|-------|----------|
| RF | Blue | #3498db |
| RF+SMARTS | Dark Blue | #1a5276 |
| LGBM | Green | #2ecc71 |
| LGBM+SMARTS | Dark Green | #1e8449 |
| XGB | Orange | #f39c12 |
| XGB+SMARTS | Dark Orange | #a04000 |
| GAT | Purple | #9b59b6 |
| GAT+SMARTS | Dark Purple | #6c3483 |

**Pattern:** Baseline models use lighter colors, SMARTS-enhanced versions use darker shades

## 💾 Export Options

### Web Tool Export
- Click "Export All Charts (PNG)" button
- Charts are saved with descriptive names:
  - `BBB_model_comparison_heatmap.png`
  - `BBB_model_comparison_barplot.png`
  - `BBB_model_comparison_scatter.png`
  - `BBB_model_comparison_complexity.png`

### Python Script Export
```bash
# Export all charts
python visualization_template.py --data data.csv --export-all

# Export specific chart
python visualization_template.py --data data.csv --chart heatmap

# Specify output directory
python visualization_template.py --data data.csv --output-dir my_charts/
```

## 🔧 Customization

### Modifying Chart Appearance

**Web tool:**
- Charts are generated using Plotly.js
- Edit the `generateHeatmap()`, `generateBarplot()`, etc. functions
- Adjust colors in the `MODEL_COLORS` object

**Python script:**
- Modify the `generate_final_plots.py` script directly
- Change color schemes in `get_model_colors()`
- Adjust figure sizes in plot functions

### Adding New Models

Simply add new data rows with your model name:
```csv
A,B,MyModel,0.950,0.920,0.940,0.930
```

The template will automatically:
- Detect the new model
- Assign a default color (or you can specify in code)
- Include it in all charts

## 📊 Performance Summary Table

After generating charts, a summary table shows:

1. **Rank**: Performance ranking (1 = best)
2. **Model**: Model name
3. **Avg AUC**: Average AUC across datasets
4. **Avg Precision**: Average precision
5. **Avg Recall**: Average recall
6. **Avg F1**: Average F1 score
7. **Avg Score**: Overall average (primary ranking metric)

**Medals:**
- 🥇 First place
- 🥈 Second place
- 🥉 Third place

## 🐛 Troubleshooting

### Charts not generating
**Problem:** Clicking generate buttons does nothing

**Solutions:**
- Make sure data is loaded first (check Data Statistics section)
- Verify CSV format is correct (Dataset,Model,AUC,Precision,Recall,F1)
- Check browser console for errors (F12 in most browsers)

### Incorrect colors
**Problem:** Models show wrong colors

**Solutions:**
- Make sure model names match exactly (case-sensitive)
- "RF" is different from "rf" or "Rf"
- Check MODEL_COLORS object in the source code

### Missing data points
**Problem:** Some models/datasets don't appear in charts

**Solutions:**
- Verify all data is included in CSV
- Check for typos in dataset names (A,B vs A, B)
- Ensure all 6 columns are present for each row

### Performance issues
**Problem:** Charts load slowly or browser freezes

**Solutions:**
- Reduce dataset size (limit to fewer datasets)
- Use Python script instead (faster for large datasets)
- Close other browser tabs to free memory

## 📝 Example Workflows

### Workflow 1: Compare Your Own Results

```bash
# 1. Export your model results to CSV
python export_my_results.py > my_results.csv

# 2. Open web template
# Double-click: visualization_template.html

# 3. Load your data
# Tab: Data Entry → Method 1: Paste CSV Data
# Paste contents of my_results.csv

# 4. Generate and export charts
# Tab: Visualize → "Generate All 4 Charts" → "Export All Charts"
```

### Workflow 2: Use Preset Examples

```bash
# 1. Open web template
# Double-click: visualization_template.html

# 2. Load preset
# Tab: Preset 1 (for SMARTS importance) or Preset 2 (for dataset impact)

# 3. Click "Load Preset X Data"

# 4. Charts auto-generate in Visualize tab
```

### Workflow 3: Python Script

```bash
# Use existing data
python visualization_template.py \
    --data outputs/model_comparison/complete_model_performance.csv \
    --output-dir my_analysis/ \
    --export-all

# Or use preset data
python visualization_template.py \
    --preset smarts_importance \
    --output-dir smarts_analysis/
```

## 📚 Technical Details

### Web Tool Technologies
- **Plotly.js**: Interactive charting library
- **HTML5/CSS3**: Modern responsive design
- **Vanilla JavaScript**: No framework dependencies

### Python Script Requirements
```bash
pip install pandas numpy matplotlib seaborn
```

### Chart Quality
- **Web tool**: Vector graphics (SVG), scalable to any size
- **Exported PNG**: 1200×800 pixels (can be adjusted)
- **Python script**: 300 DPI (publication quality)

## 🔄 Updating from Existing Data

To update charts with new data:

1. **Export your latest results** to CSV format
2. **Reload the web template** (or refresh the page)
3. **Load new data** (replaces old data)
4. **Regenerate charts**

No need to reinstall or restart - the template is fully dynamic!

## 📧 Support

For issues or questions:
1. Check this README first
2. Review the troubleshooting section
3. Check browser console (F12) for error messages
4. Verify your CSV format matches requirements

## 📄 License

This template is part of the BBB permeability prediction project. Use freely for research and academic purposes.

---

**Last Updated:** 2025-01-27
**Version:** 1.0
**Author:** BBB Project Team
