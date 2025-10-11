# Implementation Plan: ML-Based Multi-Label Song Tagger

## Overview

This plan outlines the implementation of a supervised multi-label machine learning model for automatic song tagging in the Rekordbox music library. The model will learn from 714 manually-tagged songs (with an average of 13 tags per song) to predict tags for 1,165 untagged songs.

## Dataset Summary

- **Total Songs**: 1,879 (714 tagged, 1,165 untagged)
- **Available Features**: ContentID, FolderPath, Title, ArtistName, BPM, Length, MyTagNames, MyTagIDs
- **Total Unique Tags**: 118 tags across 5 categories (Genre, Years & Origin, Situation, Mood, and others)
- **Tags per Song**: mean=13.4, std=5.3, median=13, range=[1-38]

## Architecture Overview

### Phase 1: Data Preparation & Feature Engineering

**Goal**: Create a robust dataset with rich feature representations

### Phase 2: Model Development

**Goal**: Build and train multi-label classification models

### Phase 3: Evaluation & Validation

**Goal**: Assess model performance and tune hyperparameters

### Phase 4: Inference Pipeline

**Goal**: Create production-ready inference system for tagging new songs

### Phase 5: Integration & Deployment

**Goal**: Integrate ML predictions with existing Rekordbox database utilities

---

## Detailed Implementation Steps

### Phase 1: Data Preparation & Feature Engineering

#### 1.1 Tag Preprocessing
**File**: `notebooks/04_ml_data_preparation.ipynb`

**Tasks**:
- Load unique tags from `data/unique_tags.csv`
- Analyze tag distribution and correlation
- Filter tags that aren't suitable for ML:
  - Remove or flag "TAG YEAR" and similar placeholder tags
  - Consider filtering country/year tags (as mentioned in requirements)
  - Identify and handle rare tags (present in <5 songs)
- Create tag vocabulary and multi-hot encoding schema
- Save processed tag metadata to `data/processed/tag_vocabulary.json`

**Output**:
- Tag statistics (frequency, co-occurrence matrix)
- Filtered tag list with reasoning
- Tag encoding mappings

#### 1.2 Basic Feature Extraction
**File**: `notebooks/04_ml_data_preparation.ipynb`

**Tasks**:
- Extract numerical features:
  - BPM (normalize)
  - Length (normalize, consider bins: short/medium/long)
- Create text embeddings:
  - Title embeddings using sentence transformers (e.g., `all-MiniLM-L6-v2`)
  - ArtistName embeddings
  - Optional: Combine title + artist for richer context
- One-hot encode or embed GenreName
- Feature normalization and scaling

**Output**:
- Feature matrix for basic features
- Saved embeddings to `data/processed/text_embeddings.npz`

#### 1.3 Audio Feature Extraction (Advanced)
**File**: `notebooks/05_audio_feature_extraction.ipynb`

**Tasks**:
- Load MP3 files from FolderPath
- Extract audio features using librosa:
  - Mel spectrograms (128 bands, standardized length)
  - MFCCs (Mel-frequency cepstral coefficients)
  - Chroma features
  - Spectral features (centroid, rolloff, bandwidth)
  - Tempo and beat features
  - Zero crossing rate
- Alternative: Use pre-trained audio encoders:
  - Hugging Face: `facebook/wav2vec2-base` or `m-a-p/MERT-v1-95M`
  - OpenL3 embeddings
  - MusiCNN features
- Handle audio loading errors gracefully
- Cache extracted features to avoid reprocessing

**Output**:
- Audio feature matrices saved to `data/processed/audio_features/`
- Feature extraction metadata (settings, failures, etc.)

#### 1.4 Dataset Splitting Strategy
**File**: `notebooks/04_ml_data_preparation.ipynb`

**Tasks**:
- Create stratified train/validation/test splits (70/15/15)
- Ensure tag distribution is balanced across splits (use iterative stratification for multi-label)
- Consider artist-based splitting to prevent data leakage (songs from same artist stay in same split)
- Save split indices to `data/processed/splits.json`

**Output**:
- Train/val/test DataFrames with all features
- Split statistics and validation

---

### Phase 2: Model Development

#### 2.1 Baseline Models
**File**: `notebooks/06_ml_baseline_models.ipynb`

**Models to implement**:
1. **Naive Baseline**: Artist propagation (from existing notebook)
2. **Simple ML Baseline**:
   - Logistic Regression with One-vs-Rest
   - Random Forest with multi-label support
   - Use only basic features (BPM, Length, text embeddings)

**Tasks**:
- Implement binary relevance approach (one classifier per tag)
- Implement classifier chains for tag dependencies
- Establish baseline metrics for comparison

**Metrics**:
- Hamming Loss
- Subset Accuracy
- Micro/Macro F1, Precision, Recall
- Per-tag performance
- Coverage@K (percentage of true tags in top K predictions)

#### 2.2 Deep Learning Models
**File**: `notebooks/07_ml_deep_learning_models.ipynb`

**Model architectures**:
1. **Multi-Modal Neural Network**:
   - Input branches:
     - Numerical features (BPM, Length) → Dense layers
     - Text embeddings (Title, Artist) → Dense layers
     - Audio features (spectrograms) → CNN layers
   - Concatenate all branches
   - Shared dense layers
   - Multi-label output with sigmoid activation
   - Loss: Binary Cross Entropy

2. **Audio-Focused Model**:
   - CNN on mel spectrograms
   - Optional: Pre-trained encoder (wav2vec2, MERT) + fine-tuning
   - Additional metadata as auxiliary inputs
   - Multi-label classification head

3. **Transformer-Based Model** (if needed):
   - For text: Fine-tune BERT/DistilBERT for multi-label
   - For audio: Audio Spectrogram Transformer (AST)

**Framework**: PyTorch or TensorFlow/Keras

**Tasks**:
- Implement data loaders with proper batching
- Add data augmentation for audio (time stretch, pitch shift, noise)
- Implement class weighting for imbalanced tags
- Use focal loss or weighted BCE for rare tags
- Add dropout and regularization to prevent overfitting

#### 2.3 Training Pipeline
**File**: `src/models/` (new module)

**Structure**:
```
src/
├── models/
│   ├── __init__.py
│   ├── baseline_models.py      # Sklearn models
│   ├── neural_networks.py      # PyTorch/TF models
│   ├── train.py                # Training loops
│   ├── evaluate.py             # Evaluation functions
│   └── data_loaders.py         # Dataset classes
```

**Tasks**:
- Create reusable training scripts
- Implement early stopping based on validation F1
- Add learning rate scheduling
- Log experiments with wandb or MLflow
- Save model checkpoints

---

### Phase 3: Evaluation & Validation

#### 3.1 Model Comparison
**File**: `notebooks/08_ml_model_evaluation.ipynb`

**Tasks**:
- Compare all models on test set
- Generate evaluation reports:
  - Overall metrics table
  - Per-tag performance breakdown
  - Confusion matrices for each tag
  - Prediction examples (successes and failures)
- Analyze errors:
  - Which tags are hard to predict?
  - Which songs are consistently misclassified?
  - Does model overfit to popular tags?

#### 3.2 Hyperparameter Tuning
**File**: `notebooks/09_ml_hyperparameter_tuning.ipynb`

**Tasks**:
- Grid search or Bayesian optimization for:
  - Learning rates
  - Network architecture (layers, units)
  - Dropout rates
  - Batch sizes
  - Loss function weights
- Use cross-validation on training set
- Select best model based on validation metrics

#### 3.3 Threshold Optimization
**File**: `notebooks/08_ml_model_evaluation.ipynb`

**Tasks**:
- Find optimal decision thresholds for each tag
- Balance precision vs recall based on use case
- Consider using different thresholds per tag
- Implement top-K prediction strategy (predict K most likely tags)

---

### Phase 4: Inference Pipeline

#### 4.1 Model Serving
**File**: `src/inference/predict.py`

**Tasks**:
- Create prediction pipeline:
  - Load trained model
  - Process input features (same as training)
  - Generate predictions with confidence scores
  - Apply threshold or top-K selection
- Handle batch prediction for multiple songs
- Add error handling and logging

**Functions**:
```python
def predict_tags(song_data: dict, model, threshold: float = 0.5) -> list[tuple[str, float]]:
    """Predict tags for a single song.

    Returns:
        List of (tag_name, confidence) tuples
    """
    pass

def predict_tags_batch(songs_df: pl.DataFrame, model, threshold: float = 0.5) -> pl.DataFrame:
    """Predict tags for multiple songs."""
    pass
```

#### 4.2 Integration with Database
**File**: `src/inference/apply_predictions.py`

**Tasks**:
- Extend existing utilities in `utils.py`
- Create function to apply predicted tags to Rekordbox database
- Add safety checks:
  - Don't overwrite existing manual tags
  - Add predicted tags with a marker (e.g., add to a special "ML Predictions" tag group)
  - Require user confirmation before writing
- Handle batch operations efficiently

**Functions**:
```python
def apply_ml_tags(
    db: Rekordbox6Database,
    predictions_df: pl.DataFrame,
    confidence_threshold: float = 0.7,
    dry_run: bool = True
) -> dict:
    """Apply ML-predicted tags to database.

    Args:
        db: Database instance
        predictions_df: DataFrame with ContentID and predicted tags
        confidence_threshold: Minimum confidence to apply tag
        dry_run: If True, show what would be done without applying

    Returns:
        Summary of tags applied
    """
    pass
```

---

### Phase 5: Integration & Deployment

#### 5.1 End-to-End Notebook
**File**: `notebooks/10_ml_end_to_end_inference.ipynb`

**Tasks**:
- Demonstrate complete pipeline:
  - Load untagged songs from database
  - Extract all features
  - Run inference with best model
  - Display predictions with confidence scores
  - Optionally apply to database
- Create visualization of predictions
- Generate report of tagged songs

#### 5.2 CLI Tool (Optional)
**File**: `src/cli/tag_predict.py`

**Tasks**:
- Create command-line interface:
  ```bash
  python -m src.cli.tag_predict \
    --model models/best_model.pt \
    --confidence 0.7 \
    --top-k 15 \
    --output predictions.csv
  ```
- Support different input sources (database, CSV, single file)
- Allow customization of thresholds and filters

#### 5.3 Documentation
**File**: `docs/ml_tagging_guide.md`

**Content**:
- Model architecture explanation
- Feature engineering decisions
- How to retrain models with new data
- How to use inference pipeline
- Performance benchmarks and limitations
- Future improvements

---

## Technical Stack

### Required Libraries
```toml
[project.dependencies]
# Already have:
polars = "*"
pyrekordbox = "*"

# Add for ML:
scikit-learn = ">=1.3"
numpy = ">=1.24"
pandas = ">=2.0"  # some ML libs prefer pandas

# Deep learning:
torch = ">=2.0"  # or tensorflow
torchvision = "*"

# Audio processing:
librosa = ">=0.10"
soundfile = "*"
audioread = "*"

# Text embeddings:
sentence-transformers = ">=2.2"

# Experiment tracking:
wandb = "*"  # or mlflow

# Evaluation:
scikit-multilearn = "*"  # multi-label utilities
matplotlib = ">=3.7"
seaborn = ">=0.12"

# Optional advanced models:
transformers = ">=4.30"  # Hugging Face
```

---

## Success Criteria

1. **Model Performance**:
   - Micro F1-score > 0.7 on test set
   - Macro F1-score > 0.5 on test set (harder due to rare tags)
   - Coverage@15 > 0.8 (80% of true tags in top 15 predictions)
   - Better than artist propagation baseline

2. **Production Readiness**:
   - Inference time < 1 second per song (excluding audio loading)
   - Pipeline handles errors gracefully
   - Clear documentation for maintenance

3. **Business Value**:
   - Successfully tag at least 1,000 untagged songs with >70% confidence
   - Reduce manual tagging time by 50%
   - Predictions are actionable and reasonable to users

---

## Risks & Mitigations

### Risk 1: Small Dataset (714 tagged songs)
**Mitigation**:
- Use transfer learning with pre-trained models
- Apply data augmentation (especially for audio)
- Start with simpler models to avoid overfitting
- Consider semi-supervised learning (use untagged songs)

### Risk 2: Imbalanced Tag Distribution
**Mitigation**:
- Use class weights in loss function
- Apply focal loss for rare tags
- Evaluate per-tag metrics, not just overall
- Consider ensembling multiple models

### Risk 3: Audio Processing Complexity
**Mitigation**:
- Start with text/metadata baseline first
- Use proven audio feature extraction libraries
- Cache extracted features to disk
- Provide graceful degradation if audio unavailable

### Risk 4: Computational Resources
**Mitigation**:
- Start with smaller models and scale up
- Use CPU-friendly models if GPU unavailable
- Batch inference efficiently
- Consider cloud GPUs for training (Colab, Paperspace)

---

## Timeline Estimate

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| Phase 1 | Data Preparation | 2-3 days |
| Phase 2.1 | Baseline Models | 1-2 days |
| Phase 2.2-2.3 | Deep Learning Models | 3-5 days |
| Phase 3 | Evaluation & Tuning | 2-3 days |
| Phase 4 | Inference Pipeline | 2-3 days |
| Phase 5 | Integration & Docs | 1-2 days |
| **Total** | | **11-18 days** |

---

## Future Enhancements

1. **Active Learning**: Identify songs where model is uncertain and request manual tags
2. **Tag Hierarchies**: Model tag relationships (e.g., "House" implies "Electronic")
3. **Temporal Dynamics**: Use listening history and playlist patterns
4. **User Feedback Loop**: Learn from accepted/rejected predictions
5. **Multi-Task Learning**: Jointly predict tags + other attributes (energy, danceability)
6. **Cross-Library Transfer**: Train on larger public datasets (Million Song Dataset, AcousticBrainz)

---

## References & Resources

### Datasets
- [Million Song Dataset](http://millionsongdataset.com/)
- [AcousticBrainz](https://acousticbrainz.org/)
- [FMA (Free Music Archive)](https://github.com/mdeff/fma)

### Papers
- "Deep Learning for Music Tagging" - Choi et al. (2017)
- "Multi-Label Classification: An Overview" - Tsoumakas & Katakis (2007)
- "Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans" - He et al. (2020) [focal loss]

### Pre-trained Models
- [MERT: Acoustic Music Understanding Model](https://huggingface.co/m-a-p/MERT-v1-95M)
- [Music2Vec](https://github.com/minzwon/music2vec)
- [Sentence Transformers](https://www.sbert.net/)

---

## Notes

- Start simple and iterate. Get a working baseline before adding complexity.
- Audio features will likely provide the most signal, but start with metadata to validate pipeline.
- Consider that BPM is a strong signal for genre/mood tags.
- Artist embeddings could capture style better than name alone - consider using listening patterns if available.
- The existing artist propagation (168 songs) provides a good starting point for validation.
