# SHAP Feature Importance Analysis

## Overview

I've created a comprehensive SHAP (SHapley Additive exPlanations) analysis notebook to help you understand which audio features are most important for your genre classification models.

## What is SHAP?

SHAP is a game-theoretic approach to explain machine learning model predictions. It provides:

- **Global Explanations**: Which features are most important overall
- **Local Explanations**: Why a specific prediction was made
- **Directional Impact**: Whether high/low feature values increase/decrease predictions
- **Feature Interactions**: How features work together

## Files Created

### 1. Notebook: `notebooks/09_shap_analysis.ipynb`

This notebook includes:

1. **Load Models and Data**
   - Loads your trained Random Forest and XGBoost models
   - Prepares test data with proper scaling

2. **Random Forest SHAP Analysis**
   - Global feature importance across all genres
   - Per-genre feature importance
   - SHAP summary plots
   - SHAP dependence plots
   - Heatmap comparing features across genres

3. **XGBoost SHAP Analysis**
   - Same comprehensive analysis for XGBoost
   - Allows comparison between model types

4. **Model Comparison**
   - Side-by-side feature importance comparison
   - Identifies which features both models agree on

5. **Individual Prediction Explanations**
   - Force plots showing why specific songs got certain predictions
   - Waterfall plots breaking down feature contributions

6. **Save Results**
   - Exports feature importance rankings to CSV
   - Creates reports for each genre

## Dependencies Added

SHAP has been installed in your environment:
```bash
uv pip install shap
```

Package details:
- `shap==0.49.1`
- `cloudpickle==3.1.1` (dependency)
- `slicer==0.0.8` (dependency)

## How to Use

### Quick Start

1. Open the notebook:
   ```bash
   jupyter notebook notebooks/09_shap_analysis.ipynb
   ```

2. Run all cells to perform full analysis

3. Results will be saved to: `reports/shap_analysis/`

### Key Parameters to Adjust

#### Sample Size
```python
sample_size = 100  # Adjust based on computational resources
```
- Start with 100 songs for fast iteration
- Increase to 500-1000 for more stable results
- Set to `None` to use all songs (slowest but most accurate)

#### Genre to Analyze
```python
genre_to_analyze = tags[0]  # Change index to analyze different genres
```

#### Song to Explain
```python
song_idx = 0  # Change to explain different predictions
```

## What You'll Learn

### 1. Global Feature Importance

**Question**: Which audio features matter most for genre classification overall?

**Output**:
- Bar charts showing top 20 most important features
- Comparison between Random Forest and XGBoost

**Example Insights**:
- "spectral_centroid_mean is the most important feature"
- "tempo affects predictions more than rhythm features"

### 2. Per-Genre Importance

**Question**: Which features are specific to certain genres?

**Output**:
- Feature importance rankings for each genre
- Heatmap showing which features matter for which genres

**Example Insights**:
- "Rock detection relies heavily on energy features"
- "Jazz classification uses harmonic features more"

### 3. Feature Behavior

**Question**: How do feature values affect predictions?

**Output**:
- SHAP summary plots (colored scatter plots)
- Dependence plots showing feature relationships

**Example Insights**:
- "Higher spectral_centroid → more likely to be predicted as Electronic"
- "Low tempo → more likely to be predicted as Ambient"

### 4. Individual Predictions

**Question**: Why did the model predict these genres for this song?

**Output**:
- Force plots showing contribution of each feature
- Waterfall plots ranking feature impacts

**Example Insights**:
- "This song was predicted as Rock because of high energy (+0.3) and low valence (+0.2)"
- "The model didn't predict Jazz because of low harmonic complexity (-0.15)"

## Interpretation Guide

### SHAP Value Colors (in summary plots)

- **Red/Pink**: High feature value
- **Blue**: Low feature value

### SHAP Value Sign

- **Positive SHAP value**: Feature pushes prediction HIGHER
- **Negative SHAP value**: Feature pushes prediction LOWER

### Example Reading

```
Feature: spectral_centroid_mean
SHAP value: +0.25
Feature value: 3500 Hz (high, shown in red)
```

**Interpretation**: "A high spectral_centroid (bright sound) increases the probability of this genre by 0.25"

## Output Files

After running the notebook, you'll find:

```
reports/shap_analysis/
├── feature_importance_comparison.csv       # RF vs XGBoost comparison
├── rf_Rock_importance.csv                 # Per-genre importance
├── rf_Jazz_importance.csv
├── rf_Electronic_importance.csv
└── ... (one file per genre)
```

## Common Use Cases

### 1. Feature Selection
**Goal**: Remove unimportant features to simplify model

**How**:
1. Look at global importance rankings
2. Remove features with very low SHAP values
3. Retrain model and check if performance is maintained

### 2. Feature Engineering
**Goal**: Create better features based on insights

**How**:
1. Check SHAP dependence plots for non-linear patterns
2. Create new features that capture these patterns
3. Example: If you see interaction between tempo and energy, create a combined feature

### 3. Model Debugging
**Goal**: Understand why model makes mistakes

**How**:
1. Find misclassified songs
2. Use force plots to see what features led to wrong prediction
3. Investigate if those features are noisy or miscalculated

### 4. Domain Validation
**Goal**: Verify model uses features that make musical sense

**How**:
1. Check if important features align with music theory
2. Example: Rock should use energy/loudness, Jazz should use harmonic complexity
3. If model relies on unexpected features, investigate data quality

## Performance Tips

### For Large Datasets

1. **Use sampling**: Start with `sample_size=100`
   ```python
   sample_size = 100
   ```

2. **Analyze fewer genres**: Focus on main genres first
   ```python
   genres_to_analyze = ['Rock', 'Jazz', 'Electronic']
   ```

3. **Use TreeExplainer**: Already implemented (fast for tree-based models)

4. **Save intermediate results**: Don't recompute SHAP values
   ```python
   import pickle
   pickle.dump(rf_shap_values, open('shap_values.pkl', 'wb'))
   ```

### For Faster Iteration

- Comment out cells you don't need
- Focus on one model at a time (RF or XGBoost)
- Generate summary plots only for key genres

## Advanced Analysis (Optional)

### 1. SHAP Interaction Values

See how features interact:
```python
shap_interaction_values = explainer.shap_interaction_values(X_sample_np)
shap.summary_plot(shap_interaction_values[0], X_sample_np, feature_names=feature_cols)
```

### 2. Clustering by SHAP Values

Group songs by similar feature patterns:
```python
from sklearn.cluster import KMeans
shap_matrix = rf_shap_values[genre_to_analyze]
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(shap_matrix)
```

### 3. Feature Correlation with SHAP Values

Find which features have correlated impacts:
```python
import pandas as pd
shap_df = pd.DataFrame(rf_shap_values[genre_to_analyze], columns=feature_cols)
correlation_matrix = shap_df.corr()
```

## Troubleshooting

### Issue: "ValueError: operands could not be broadcast together with shapes"
**Solution**: This was caused by SHAP returning values for both classes in binary classification. The notebook has been updated to handle this correctly by extracting only the positive class (class 1) SHAP values.

**What was fixed**:
- Added proper handling for list format: `shap_values[1]`
- Added handling for 3D array format: `shap_values[:, :, 1]`
- Updated both Random Forest and XGBoost sections
- Fixed `expected_value` extraction in force plots

If you still see this error, make sure you're using the updated notebook and rerun the SHAP computation cells.

### Issue: "Out of memory"
**Solution**: Reduce `sample_size` or analyze fewer genres at once

### Issue: "Plots are too small to read"
**Solution**: Adjust figure size:
```python
plt.figure(figsize=(14, 10))  # Increase these numbers
```

### Issue: "SHAP computation takes too long"
**Solution**:
- Reduce sample size
- Use only one model (RF or XGBoost)
- Process genres in batches

### Issue: "Can't interpret SHAP values"
**Solution**:
- Focus on summary plots first (easiest to understand)
- Compare SHAP rankings with your intuition
- Read SHAP documentation: https://shap.readthedocs.io/

## Next Steps

1. **Run the analysis**: Execute the notebook and review visualizations

2. **Identify key features**: Note which features appear most important

3. **Validate insights**: Check if important features make musical sense

4. **Feature engineering**: Based on insights, consider:
   - Creating new combined features
   - Removing redundant features
   - Transforming non-linear relationships

5. **Model refinement**: Retrain models with improved feature set

6. **Document findings**: Update your project documentation with key insights

## Resources

- **SHAP Documentation**: https://shap.readthedocs.io/
- **SHAP Paper**: https://arxiv.org/abs/1705.07874
- **Interactive Examples**: https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/

## Questions?

Common questions and answers:

**Q: Why do SHAP values differ from sklearn's feature_importances_?**
A: SHAP values account for feature interactions and provide more nuanced importance. sklearn's feature_importances_ is simpler but less detailed.

**Q: Can I use SHAP for other tag groups (Mood, Situation)?**
A: Yes! Just change which model you load. The notebook works for any multi-label classification model.

**Q: Should I trust SHAP values completely?**
A: SHAP is a powerful tool, but always validate insights against domain knowledge and cross-check with other methods.

**Q: How often should I rerun SHAP analysis?**
A: Rerun after:
- Training new models
- Adding new features
- Significant data changes
- When model performance changes unexpectedly