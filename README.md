# Flowers CNN — From a Baseline Model to Something Better

A five-class flower image classifier in TensorFlow/Keras. The point of this one was not to get a high number. It was to see the model overfit, work out why, and then fix it step by step.

## How it went

1. **Baseline CNN**: three Conv-MaxPool blocks, Adam with sparse categorical cross entropy, 25 epochs.
2. **Look at the curves**: training accuracy climbed to **0.9997** while validation accuracy sat around **0.65–0.68**. That is overfitting, very clearly.
3. **Three fixes**:
   - L2 regularisation (`kernel_regularizer`)
   - Dropout
   - Data augmentation (rotation, shift, zoom, flip through `ImageDataGenerator`)
4. **Result of the improved model**: validation accuracy **76.01%**, about 8 points above the best baseline run (68.02%).

| Model | Train acc | Val acc |
|---|---|---|
| Baseline CNN | 0.9997 | 0.6802 (best) |
| Improved (L2 + Dropout + augmentation) | — | **0.7601** |

## Other things I did

- **Tested with a photo from my own phone** so the model is judged on images it has never seen, not on the training split.
- Plotted the `history` curves for each version, so you can see which change moved the validation line.

## Result figures

`results/` has 7 figures: class distribution, baseline curves, improved curves, the two models compared, model summary, and the phone-photo test.

| | |
|---|---|
| ![Baseline curves](results/flowers-cnn_fig03.png) | ![Comparison](results/flowers-cnn_fig06.png) |
| ![Phone test](results/flowers-cnn_fig05.png) | ![Class distribution](results/flowers-cnn_fig01.png) |

## Environment

```bash
pip install tensorflow numpy matplotlib opencv-python
```

The dataset is five folders of flower images (not uploaded here). The notebook explains how the `flowers/` folder should be laid out.
