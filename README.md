# Solar-Cycle-26-Prediction-using-LSTM-GRU-
This notebook performs time series forecasting on sunspot data using a stacked ensemble model of LSTM, GRU, and XGBoost.

## Data Loading and Preprocessing: The code loads sunspot data from a CSV file, sorts it by date, and performs min-max scaling on the sunspot values, month, and year.
## Feature Engineering: It creates sequences of scaled features and corresponding target values (sunspot numbers) for training the models.
## Chronological Train-Validation-Test Split: The data is split into training, validation, and test sets chronologically to maintain the temporal order.
## Model Training:
A stacked LSTM model is trained on the training data.
A GRU model is also trained on the training data.
Stacking Features: Predictions from the trained LSTM and GRU models on the validation set are stacked to be used as input features for the XGBoost model.
XGBoost Ensemble: An XGBoost Regressor is trained on the stacked predictions from the validation set, with the true sunspot values as the target.
Test Prediction: The trained LSTM and GRU models predict on the test set, their predictions are stacked, and the XGBoost model makes the final predictions on the stacked test predictions.
##Model Evaluation: The performance of the ensemble model is evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on the test set.
Forecasting Future Predictions: The trained ensemble model is used to forecast future sunspot values for cycle 25 and cycle 26 by iteratively predicting the next 13 steps and using those predictions as part of the input for the subsequent prediction.
## Plotting: The historical data and the forecasted values are plotted to visualize the model's predictions.
## Peak Sunspot Value: The code identifies and prints the date and value of the peak sunspot number predicted for cycles 25 and 26.
# Model Training Details
This section trains three models: a Stacked LSTM, a GRU, and an XGBoost Regressor used as a meta-model in an ensemble.

## Stacked LSTM
Architecture:
Three stacked LSTM layers with 128, 64, and 32 units respectively. The first LSTM layer has return_sequences=True to pass the sequence output to the next LSTM layer. The second LSTM layer also has return_sequences=True. The third LSTM layer has return_sequences=False as it feeds into a Dense layer.
A Dropout layer with a rate of 0.2 is applied after the last LSTM layer to prevent overfitting.
A final Dense layer with 13 units to output the predicted sequence of 13 values.
Activation Function: The LSTM layers use the 'tanh' activation function.
Optimizer: The 'adam' optimizer is used for training.
Loss Function: Mean Squared Error (MSE) is used as the loss function.
Hyperparameters:
epochs: 40
batch_size: 12
validation_split: 0.1 (10% of the training data is used for validation during training)
Callbacks: Early Stopping is used with a patience of 5 epochs. This means training will stop if the validation loss does not improve for 5 consecutive epochs, and the best weights from the training run will be restored.
## GRU
Architecture:
A GRU layer with 64 units. return_sequences is set to False as it directly feeds into a Dense layer.
A Dropout layer with a rate of 0.2 is applied after the GRU layer.
A final Dense layer with 13 units to output the predicted sequence of 13 values.
Activation Function: The GRU layer uses the 'tanh' activation function.
Optimizer: The 'adam' optimizer is used for training.
Loss Function: Mean Squared Error (MSE) is used as the loss function.
Hyperparameters:
epochs: 100
batch_size: 16
validation_split: 0.1 (10% of the training data is used for validation during training)
Callbacks: Early Stopping is used with a patience of 10 epochs.
## XGBoost Ensemble (Meta-model)
Purpose: This model is trained on the predictions of the LSTM and GRU models (stacked features) to make the final prediction.
Model Type: XGBoost Regressor.
Hyperparameters:
n_estimators: 100 (number of boosting rounds)
learning_rate: 0.1 (step size shrinkage used in update to prevent overfitting)
Training Data: The model is trained on the stacked predictions of the LSTM and GRU models on the validation set, using the true values from the validation set as the target. This allows the XGBoost model to learn how to best combine the predictions of the base models.
