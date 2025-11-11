import numpy as np
import pandas as pd
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv("four_tank_data.csv", sep=',')
df = df.drop('Time', axis=1)

# Define input and output columns
input_cols = ['v1', 'v2', 'h1', 'h2', 'h3', 'h4']
output_cols = ['h1', 'h2', 'h3', 'h4']

# Extract values
X_raw = df[input_cols].values
y_raw = df[output_cols].values

# Scale inputs and outputs separately
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# Convert to LSTM sequences
def create_sequences(X, y, window):
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i-window:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

window = 20
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(4))
model.compile(optimizer='adam', loss='mse')

print("Model Summary:")
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train model
print("\nTraining model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    callbacks=[early_stop], validation_split=0.2, verbose=1)

print("\nSaving model and scalers...")

# Save the trained model
model.save('lstm_four_tank_model.h5')

# Save the scalers and window size
model_artifacts = {
    'x_scaler': x_scaler,
    'y_scaler': y_scaler,
    'window': window,
    'input_cols': input_cols,
    'output_cols': output_cols
}

with open('model_artifacts.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("Model saved as 'lstm_four_tank_model.h5'")
print("Scalers and artifacts saved as 'model_artifacts.pkl'")
print("\nTraining complete!")
