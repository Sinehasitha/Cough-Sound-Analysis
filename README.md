# Cough Sound Analysis
This project classifies audio cough recordings into respdisease or not_respdisease using a deep learning model.

## Data Collection
- Audio files and a CSV file with labels were downloaded from Kaggle (coughclassifier-trial).

## Preprocessing
- Audio files were converted into spectrogram images.
- Features were extracted including MFCCs, Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, and Spectral Roll-off.
- Labels were encoded, and features were scaled using StandardScaler.

## Model Building
- A fully connected neural network (Dense layers) was created using Keras.
- The network contains layers of sizes 512 → 256 → 128 → 64 → 10 → 2 with ReLU and Softmax activations.
- Model architecture was saved as JSON and weights as HDF5.

## Training
- The model was trained on 67% of the dataset for 100 epochs with batch size 128.

## Evaluation
- Test accuracy achieved: 96.5%
- ROC-AUC score: 0.927
- Predictions on test data were compared with ground truth labels.

## Visualization
- Spectrograms, training progress, and ROC curve were plotted for analysis.

