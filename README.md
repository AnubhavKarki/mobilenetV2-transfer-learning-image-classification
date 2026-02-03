# MobileNetV2 Transfer Learning for Dog Breed Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end multi-class image classification using MobileNetV2 transfer learning on the Kaggle Dog Breed Identification dataset. Achieves accurate breed classification across 120 dog breeds using pre-trained ImageNet weights.

## Features

- **Transfer Learning**: MobileNetV2 backbone with custom classification head
- **120 Dog Breeds**: Trained on 10,500+ labeled training images
- **Data Pipeline**: Efficient TensorFlow `tf.data` for preprocessing and batching
- **Callbacks**: TensorBoard logging and early stopping
- **Evaluation**: Prediction visualization, confidence analysis, and validation metrics
- **Production Ready**: Model saving/loading in HDF5 format
- **GPU Optimized**: Automatic GPU detection and utilization

## Dataset

[Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

- **Training**: 10,222 images across 120 breeds
- **Test**: 10,357 unlabeled images for predictions
- **Labels**: CSV mapping image IDs to breed names
- **Images**: JPEG format, variable sizes (resized to 224x224)

## Architecture

```
MobileNetV2 (ImageNet weights)
    ↓
Global Average Pooling 2D
    ↓
Dense (120 neurons, softmax)
```

- **Input Shape**: `(224, 224, 3)`
- **Output Shape**: `(120,)` - probability distribution over breeds
- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam

## Quick Start

### Prerequisites

```bash
pip install tensorflow tensorflow-hub keras tf-keras pandas matplotlib scikit-learn
```

### Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the notebook or clone this repository
3. Mount Google Drive with dataset
4. Run all cells sequentially

### Local Environment

1. Clone repository
2. Download dataset to `data/` directory
3. Update `ROOT_PATH` in notebook to your data location
4. Execute notebook

## Usage

### Training

```python
# Trains on first 200 images (adjust NUM_IMAGES for full dataset)
model = train_model()
```

### Evaluation

```python
# Visualize predictions with confidence scores
plot_pred(predictions, val_labels, val_images)

# Confusion matrix and metrics
# (See notebook for complete evaluation)
```

### Inference

```python
# Load saved model
model = load_model('models/20260203-034454.h5')

# Predict on new images
test_data = create_data_batches(test_filenames, test_data=True)
predictions = model.predict(test_data)
```

## Data Preprocessing Pipeline

1. **Image Loading**: `tf.io.read_file()` + `tf.image.decode_jpeg()`
2. **Normalization**: Pixel values scaled to `[0,1]`
3. **Resizing**: Fixed to `224x224`
4. **Batching**: `tf.data.Dataset` with prefetching and shuffling
5. **Augmentation**: Ready for addition (see enhancements)

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32 | Optimal for most GPUs |
| Image Size | 224x224 | MobileNetV2 standard |
| Epochs | 10+ | Early stopping enabled |
| Optimizer | Adam | Default learning rate |
| Monitor | `val_accuracy` | Early stopping patience=3 |

## Results

- **Validation Accuracy**: Tracks improvement via TensorBoard
- **Prediction Confidence**: Top-10 confidence visualization
- **Test Predictions**: Saved as `preds_log.csv` for Kaggle submission

## Model Saving/Loading

```python
# Save
model_path = save_model(model, suffix="mobilenetv2-Adam")

# Load
loaded_model = load_model(model_path)
```

Models saved in `models/` directory with timestamp suffix.

## Visualization Tools

- **Batch Visualization**: 5x5 grid of images with breed labels
- **Prediction Plots**: Ground truth vs predicted with confidence %
- **Top-10 Confidence**: Bar chart of prediction distribution
- **TensorBoard**: Loss/accuracy curves during training

## File Structure

```
├── data/
│   ├── train/           # 10,222 labeled images
│   ├── test/            # 10,357 test images
│   └── labels.csv       # Image ID -> breed mapping
├── models/              # Saved .h5 models
├── preds_log.csv        # Test predictions
└── notebook.ipynb       # Main training notebook
```

## Performance Optimizations

- **tf.data Pipeline**: Shuffles paths before processing (memory efficient)
- **GPU Detection**: Automatic GPU utilization check
- **Early Stopping**: Prevents overfitting
- **Batch Processing**: Handles large datasets efficiently

## Kaggle Submission

1. Train on full dataset (`NUM_IMAGES = None`)
2. Generate test predictions
3. Format as Kaggle submission CSV
4. Upload to [competition](https://www.kaggle.com/c/dog-breed-identification)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not detected | Restart runtime, check `tf.config.list_physical_devices("GPU")` |
| OOM Error | Reduce `BATCH_SIZE` or `NUM_IMAGES` |
| Path errors | Verify `ROOT_PATH` points to `train/` directory |
| Model loading fails | Ensure `custom_objects={'KerasLayer': hub.KerasLayer}` |

## Enhancements (Next Steps)

- [ ] Data augmentation (rotation, flip, brightness)
- [ ] Learning rate scheduling
- [ ] Model ensembling
- [ ] Test-time augmentation
- [ ] Full dataset training
- [ ] Cross-validation
- [ ] Deployment (TensorFlow Serving / ONNX)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) for core framework
- [Kaggle Dog Breed Dataset](https://www.kaggle.com/c/dog-breed-identification)
- [Google Colab](https://colab.research.google.com/) for free GPU access
