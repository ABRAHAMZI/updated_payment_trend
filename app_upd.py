import streamlit as st
import bcrypt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from scipy.stats import linregress
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import tensorflow as tf
from numpy.polynomial.polynomial import Polynomial

# Ensure .env file is loaded before using any environment variables
load_dotenv("cred.env")

# Read the password from the environment
secure_password = os.getenv("PASSWORD")

# Check if password is loaded correctly
if secure_password is None:
    st.error("Password environment variable is not set. Please check your .env file.")
    st.stop()  # Stop the app if the password is not found.

# Hash the password for authentication
stored_hash = bcrypt.hashpw(secure_password.encode('utf-8'), bcrypt.gensalt())

# Function to handle authentication with password hashing
def authenticate():
    username = st.text_input("Username", "")
    password = st.text_input("Password", "", type="password")

    if st.button("Login"):
        if username == "adam" and bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid username or password.")
            return False

# LSTM Model and Visualization with Smoothing and Trend Line
def run_lstm_model():
    df = pd.read_csv('updated_data_with_salary.csv', index_col=0)

    # Feature Engineering
    df['date'] = pd.to_datetime(df['date'],format="%d-%m-%Y")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df[['sender', 'transaction_type']])
    encoded_feature_names = encoder.get_feature_names_out(['sender', 'transaction_type'])
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    df = pd.concat([df, df_encoded], axis=1).drop(columns=['sender', 'transaction_type'])

    # Scale numerical features
    scaler = MinMaxScaler()
    try:
        df[['transaction_amount', 'balance']] = scaler.fit_transform(df[['transaction_amount', 'balance']])
    except ValueError as e:
        st.error(f"Scaling error: {e}")
        return

    # Prepare data for LSTM
    data = df.drop(columns=['date']).values

    def create_sequences(data, window_size):
        sequences, targets = [], []
        for i in range(len(data) - window_size):
            sequence = data[i:i + window_size]
            target = data[i + window_size, 0]
            sequences.append(sequence)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    window_size = 60
    X, y = create_sequences(data, window_size)

    # Reshape for LSTM input
    num_features = X.shape[2]
    X = X.reshape((X.shape[0], X.shape[1], num_features))

    # Set the random seed for reproducibility
    tf.random.set_seed(42)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(window_size, num_features)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile the model
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse')

    # Training the model
    st.write("Training the model...")
    epochs = 30
    batch_size = 90
    progress_bar = st.progress(0)
    loss_placeholder = st.empty()

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Training loop
    for epoch in range(epochs):
        history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, callbacks=[early_stopping])
        loss = history.history['loss'][0]

        if np.isnan(loss):
            st.error("Loss became NaN during training. Check your data or model settings.")
            return

        progress_bar.progress((epoch + 1) / epochs)
        loss_placeholder.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        if early_stopping.stopped_epoch > 0:
            st.write(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    # Generate future predictions
    n_future = 90
    future_predictions = []
    current_sequence = X[-1]

    for _ in range(5):
        temp_future_predictions = []
        current_sequence = X[-1]
        for _ in range(n_future):
            next_value = model.predict(current_sequence.reshape(1, window_size, num_features), verbose=0)[0, 0]
            temp_future_predictions.append(next_value)
            next_sequence = np.append(current_sequence[1:], [[next_value] + [0] * (num_features - 1)], axis=0)
            current_sequence = next_sequence
        future_predictions.append(np.array(temp_future_predictions))

    future_predictions = np.mean(future_predictions, axis=0)

    # Rescale predictions
    filler_array = np.zeros((future_predictions.shape[0], 2))
    filler_array[:, 0] = future_predictions
    future_predictions_rescaled = scaler.inverse_transform(filler_array)
    future_transaction_amounts = future_predictions_rescaled[:, 0]

    # Smooth Future Predictions using a simple moving average (3-point moving average)
    window_size_smooth = 3
    smoothed_predictions = np.convolve(future_transaction_amounts, np.ones(window_size_smooth) / window_size_smooth, mode='valid')

    # Linear regression to get trend line
    x_values = np.arange(len(smoothed_predictions))
    slope, intercept, _, _, _ = linregress(x_values, smoothed_predictions)
    trend = "Positive" if slope > 0 else "Negative"
    st.write(f"The trend of future transactions is: {trend}")

    # Polynomial regression for best fit line
    p = Polynomial.fit(x_values, smoothed_predictions, 2)
    y_poly_fit = p(x_values)

    # Visualization of future predictions and trend lines (smoothed and polynomial fit)
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_predictions, label="Smoothed Future Predictions", color="blue")
    plt.plot(x_values, y_poly_fit, label="Polynomial Trend Line (Degree 2)", color="red", linestyle="--")
    plt.title("Future Transaction Amount Predictions with Trend Lines")
    plt.xlabel("Future Transactions")
    plt.ylabel("Transaction Amount")
    plt.legend()
    st.pyplot(plt)

    # Combine past and future transactions for second plot
    past_transactions_rescaled = scaler.inverse_transform(df[['transaction_amount', 'balance']])[:, 0]
    combined_transactions = np.concatenate((past_transactions_rescaled[-window_size:], smoothed_predictions))

    # Plot combined past and future transactions
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(past_transactions_rescaled[-window_size:])), past_transactions_rescaled[-window_size:], label="Past Transactions", color="green")
    plt.plot(np.arange(len(past_transactions_rescaled[-window_size:]), len(combined_transactions)), smoothed_predictions, label="Smoothed Future Predictions", color="blue")
    plt.axvline(x=window_size - 1, color="red", linestyle="--", label="Prediction Start")
    plt.xlim(0, len(combined_transactions) - 1)
    plt.title("Combined Past and Smoothed Future Transaction Amounts")
    plt.xlabel("Transactions")
    plt.ylabel("Transaction Amount")
    plt.legend()
    st.pyplot(plt)

# Main function
def main():
    st.title("Welcome to the Streamlit App!")

    if authenticate():
        st.write("You are now logged in.")
        run_lstm_model()

if __name__ == "__main__":
    main()
