import boto3
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# S3 Configuration
S3_BUCKET_NAME = 'malware-detection-project-sppu'  # Your S3 bucket name
DATA_FILE_KEY = 'Malware dataset.csv'  # The dataset file key in the S3 bucket
MODEL_FILE_KEY = 'models/malware_model.h5'  # S3 path to save the model
SCALER_FILE_KEY = 'models/scaler.pkl'  # S3 path to save the scaler

# Initialize S3 client
s3_client = boto3.client('s3')

# Fetch the dataset from S3
s3_client.download_file(S3_BUCKET_NAME, DATA_FILE_KEY, 'Malware dataset.csv')
print(f"Downloaded {DATA_FILE_KEY} from S3")

# Data Preprocessing
df = pd.read_csv('Malware dataset.csv')

# Drop unnecessary columns from the dataset
columns_to_drop = [
    'hash', 'millisecond', 'prio', 'static_prio', 'normal_prio',
    'vm_pgoff', 'vm_truncate_count', 'nr_ptes', 'end_data', 'last_interval'
]
df_cleaned = df.drop(columns=columns_to_drop)

# Map 'malware' to 1 and 'benign' to 0 in the classification column
df_cleaned['classification'] = df_cleaned['classification'].map({'malware': 1, 'benign': 0})

# Split data into features (X) and target (y)
X = df_cleaned.drop(columns='classification')  # Features
y = df_cleaned['classification']  # Target

# Scale features using StandardScaler (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a Sequential neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Dropout layer to avoid overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(1, activation='sigmoid'))  # Binary classification: 'malware' or 'benign'

# Compile the model with Adam optimizer and binary cross-entropy loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
y_pred = (model.predict(X_test) > 0.5).astype('int32')  # Sigmoid output is thresholded at 0.5
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a .h5 file (Keras format)
model.save('malware_model.h5')
print("Model saved locally as malware_model.h5")

# Upload the trained model to S3
s3_client.upload_file('malware_model.h5', S3_BUCKET_NAME, MODEL_FILE_KEY)
print(f"Model uploaded to S3 at {MODEL_FILE_KEY}")

# Save and upload the scaler for future prediction preprocessing
joblib.dump(scaler, 'scaler.pkl')
s3_client.upload_file('scaler.pkl', S3_BUCKET_NAME, SCALER_FILE_KEY)
print(f"Scaler uploaded to S3 at {SCALER_FILE_KEY}")
