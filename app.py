import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#1. Create a Mock Dataset
# Define the holidays and special_days
holidays = ['2022-01-01', '2022-01-26']  # New Year and a random date for demonstration
special_days = ['2022-01-10', '2022-01-20']  # Two random dates for demonstration
date_rng = pd.date_range(start='2022-01-01', end='2022-01-31', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df['customer_traffic'] = np.random.randint(100, 500, size=(len(date_rng)))
df['day'] = df['date'].dt.day
df['time'] = np.random.choice(['morning', 'afternoon', 'evening'], len(date_rng))
df['month'] = df['date'].dt.month
df['season'] = np.where(df['month'].isin([12,1,2]), 'winter', 'other')
df['weather'] = np.random.choice(['sunny', 'rainy', 'cloudy'], len(date_rng))
df['chicken_cooked'] = np.random.randint(10, 50, size=(len(date_rng)))
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
df['is_holiday'] = df['date'].isin(holidays).astype(int)
df['special_event'] = df['date'].isin(special_days).astype(int)
df['prev_day_sales'] = df['chicken_cooked'].shift(1)
df['prev_day_sales'].fillna(df['chicken_cooked'].mean(), inplace=True)

# Preprocess the data
df = pd.get_dummies(df, columns=['time', 'season', 'weather'])

# Splitting the dataset
train_size = int(0.7 * len(df))
train, test = df.iloc[:train_size], df.iloc[train_size:]
X_train = train.drop(['date', 'chicken_cooked'], axis=1).values.astype('float32')
y_train = train['chicken_cooked'].values.astype('float32')
X_test = test.drop(['date', 'chicken_cooked'], axis=1).values.astype('float32')
y_test = test['chicken_cooked'].values.astype('float32')

# 3. Build a Neural Network for Demand Forecasting
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)


#Making a simple prediction for the next 10 days
for i in range(10):
    sample = X_test[i]
    prediction = model.predict(np.array([sample]))[0][0]
    print(f"Forecasted chicken cooked for {i} is: {prediction:.2f} kg")


# Make predictions for the entire test set
predictions = model.predict(X_test).flatten()

#Actual vs. Predicted Plot
#note: since we are training the model on an uncomprehensive mock dataset, the predictions will not be accurate. 
#However we hope this serves as a proof of concept. 
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual", color='blue')
plt.plot(predictions, label="Predicted", color='red', linestyle='dashed')
plt.title("Actual vs. Predicted Chicken Cooked")
plt.xlabel("Days")
plt.ylabel("Quantity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
