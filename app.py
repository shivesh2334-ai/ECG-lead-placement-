# ECG Lead Misplacement Detection - Machine Learning Application
# Based on "Incorrect electrode cable connection during electrocardiographic recording"
# by Batchvarov et al., Europace 2007

# ========================================
# INSTALLATION AND IMPORTS
# ========================================

# Install required packages
!pip install tensorflow scikit-learn matplotlib seaborn pandas numpy wfdb neurokit2
!pip install plotly kaleido

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("ECG Lead Misplacement Detection System")
print("=====================================")
print("Libraries imported successfully!")

# ========================================
# ECG DATA SIMULATION
# ========================================

class ECGSignalGenerator:
    """
    Generates synthetic ECG signals for different lead configurations
    Based on the paper's descriptions of lead misplacement patterns
    """

    def __init__(self, sampling_rate=500, duration=10):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.n_samples = int(sampling_rate * duration)
        self.time = np.linspace(0, duration, self.n_samples)

    def generate_normal_ecg_lead(self, lead_type='I', amplitude_factor=1.0):
        """Generate a normal ECG waveform for a specific lead"""
        # Basic ECG components (simplified model)
        heart_rate = 75  # bpm
        rr_interval = 60 / heart_rate

        signal = np.zeros(self.n_samples)

        # Generate multiple heartbeats
        for beat_time in np.arange(0, self.duration, rr_interval):
            beat_start = int(beat_time * self.sampling_rate)

            if beat_start + int(0.8 * self.sampling_rate) < self.n_samples:
                # P wave (0.08-0.12s)
                p_wave = self._generate_p_wave(lead_type) * amplitude_factor
                p_start = beat_start + int(0.05 * self.sampling_rate)
                p_end = p_start + int(0.1 * self.sampling_rate)
                signal[p_start:p_end] += p_wave[:p_end-p_start]

                # QRS complex (0.06-0.10s)
                qrs_complex = self._generate_qrs_complex(lead_type) * amplitude_factor
                qrs_start = beat_start + int(0.18 * self.sampling_rate)
                qrs_end = qrs_start + int(0.08 * self.sampling_rate)
                signal[qrs_start:qrs_end] += qrs_complex[:qrs_end-qrs_start]

                # T wave (0.16s)
                t_wave = self._generate_t_wave(lead_type) * amplitude_factor
                t_start = beat_start + int(0.32 * self.sampling_rate)
                t_end = t_start + int(0.16 * self.sampling_rate)
                signal[t_start:t_end] += t_wave[:t_end-t_start]

        # Add noise
        noise = np.random.normal(0, 0.05, self.n_samples)
        signal += noise

        return signal

    def _generate_p_wave(self, lead_type):
        """Generate P wave based on lead type"""
        t = np.linspace(0, 0.1, int(0.1 * self.sampling_rate))

        # Lead-specific P wave characteristics
        if lead_type in ['I', 'II', 'V4', 'V5', 'V6']:
            # Positive P wave
            return 0.2 * np.exp(-((t - 0.05) / 0.02) ** 2)
        elif lead_type == 'AVR':
            # Negative P wave in AVR
            return -0.2 * np.exp(-((t - 0.05) / 0.02) ** 2)
        else:
            return 0.1 * np.exp(-((t - 0.05) / 0.02) ** 2)

    def _generate_qrs_complex(self, lead_type):
        """Generate QRS complex based on lead type"""
        t = np.linspace(0, 0.08, int(0.08 * self.sampling_rate))

        # Lead-specific QRS characteristics
        if lead_type == 'I':
            # Normal lead I: positive QRS
            q_wave = -0.1 * np.exp(-((t - 0.01) / 0.005) ** 2)
            r_wave = 1.0 * np.exp(-((t - 0.04) / 0.01) ** 2)
            s_wave = -0.2 * np.exp(-((t - 0.06) / 0.005) ** 2)
            return q_wave + r_wave + s_wave
        elif lead_type == 'II':
            # Normal lead II: tall R wave
            r_wave = 1.5 * np.exp(-((t - 0.04) / 0.01) ** 2)
            s_wave = -0.1 * np.exp(-((t - 0.06) / 0.005) ** 2)
            return r_wave + s_wave
        elif lead_type == 'AVR':
            # Normal AVR: negative QRS
            return -1.0 * np.exp(-((t - 0.04) / 0.01) ** 2)
        else:
            # Default QRS pattern
            r_wave = 0.8 * np.exp(-((t - 0.04) / 0.01) ** 2)
            return r_wave

    def _generate_t_wave(self, lead_type):
        """Generate T wave based on lead type"""
        t = np.linspace(0, 0.16, int(0.16 * self.sampling_rate))

        # Most leads have positive T waves
        if lead_type != 'AVR':
            return 0.3 * np.exp(-((t - 0.08) / 0.04) ** 2)
        else:
            return -0.2 * np.exp(-((t - 0.08) / 0.04) ** 2)

class ECGLeadMisplacementSimulator:
    """
    Simulates various types of ECG lead misplacements as described in the paper
    """

    def __init__(self):
        self.generator = ECGSignalGenerator()
        self.lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def generate_normal_12_lead_ecg(self):
        """Generate a normal 12-lead ECG"""
        ecg_data = {}

        for lead in self.lead_names:
            ecg_data[lead] = self.generator.generate_normal_ecg_lead(lead)

        return ecg_data

    def simulate_ra_la_reversal(self, normal_ecg):
        """
        Simulate RA/LA cable reversal
        Effects: Lead I inverted, Leads II and III swapped
        """
        misplaced_ecg = normal_ecg.copy()

        # Invert lead I (most obvious sign)
        misplaced_ecg['I'] = -normal_ecg['I']

        # Swap leads II and III
        misplaced_ecg['II'] = normal_ecg['III']
        misplaced_ecg['III'] = normal_ecg['II']

        # Swap AVL and AVR
        misplaced_ecg['AVL'] = normal_ecg['AVR']
        misplaced_ecg['AVR'] = normal_ecg['AVL']

        # AVF remains the same

        return misplaced_ecg

    def simulate_ra_ll_reversal(self, normal_ecg):
        """
        Simulate RA/LL cable reversal
        Effects: Lead II inverted, AVR and AVF swapped
        """
        misplaced_ecg = normal_ecg.copy()

        # Invert lead II
        misplaced_ecg['II'] = -normal_ecg['II']

        # Lead I becomes -III
        misplaced_ecg['I'] = -normal_ecg['III']

        # Lead III becomes -I
        misplaced_ecg['III'] = -normal_ecg['I']

        # Swap AVR and AVF
        misplaced_ecg['AVR'] = normal_ecg['AVF']
        misplaced_ecg['AVF'] = normal_ecg['AVR']

        return misplaced_ecg

    def simulate_la_ll_reversal(self, normal_ecg):
        """
        Simulate LA/LL cable reversal
        Often difficult to detect - may look "more normal"
        """
        misplaced_ecg = normal_ecg.copy()

        # Lead I becomes lead II
        misplaced_ecg['I'] = normal_ecg['II']

        # Lead III inverted
        misplaced_ecg['III'] = -normal_ecg['III']

        # Swap AVL and AVF
        misplaced_ecg['AVL'] = normal_ecg['AVF']
        misplaced_ecg['AVF'] = normal_ecg['AVL']

        return misplaced_ecg

    def simulate_precordial_reversal(self, normal_ecg, lead1='V1', lead2='V2'):
        """
        Simulate precordial lead reversal (e.g., V1/V2)
        """
        misplaced_ecg = normal_ecg.copy()

        # Swap the specified precordial leads
        misplaced_ecg[lead1] = normal_ecg[lead2]
        misplaced_ecg[lead2] = normal_ecg[lead1]

        return misplaced_ecg

    def simulate_limb_ground_reversal(self, normal_ecg, limb_lead='RA'):
        """
        Simulate limb electrode with ground reversal
        Creates flat line in one of the standard leads
        """
        misplaced_ecg = normal_ecg.copy()

        if limb_lead == 'RA':
            # Lead II becomes flat
            misplaced_ecg['II'] = np.zeros_like(normal_ecg['II']) + np.random.normal(0, 0.01, len(normal_ecg['II']))
            # Lead I becomes -III
            misplaced_ecg['I'] = -normal_ecg['III']

        elif limb_lead == 'LA':
            # Lead III becomes flat
            misplaced_ecg['III'] = np.zeros_like(normal_ecg['III']) + np.random.normal(0, 0.01, len(normal_ecg['III']))
            # Lead I becomes II
            misplaced_ecg['I'] = normal_ecg['II']

        return misplaced_ecg

# ========================================
# FEATURE EXTRACTION
# ========================================

class ECGFeatureExtractor:
    """
    Extract features from ECG signals for misplacement detection
    Based on the diagnostic criteria mentioned in the paper
    """

    def __init__(self):
        self.features = []

    def extract_features(self, ecg_data):
        """
        Extract comprehensive features from 12-lead ECG
        """
        features = {}

        # 1. Polarity features (key diagnostic criteria)
        features.update(self._extract_polarity_features(ecg_data))

        # 2. Amplitude features
        features.update(self._extract_amplitude_features(ecg_data))

        # 3. Morphology features
        features.update(self._extract_morphology_features(ecg_data))

        # 4. Cross-lead correlation features
        features.update(self._extract_correlation_features(ecg_data))

        # 5. Axis-related features
        features.update(self._extract_axis_features(ecg_data))

        return features

    def _extract_polarity_features(self, ecg_data):
        """
        Extract polarity-based features (most important for detection)
        """
        features = {}

        # Check for negative P-QRS in leads I and II (major red flags)
        for lead in ['I', 'II']:
            mean_amplitude = np.mean(ecg_data[lead])
            features[f'{lead}_negative'] = 1 if mean_amplitude < -0.1 else 0
            features[f'{lead}_mean_amplitude'] = mean_amplitude

        # Check for positive QRS in AVR (abnormal)
        avr_mean = np.mean(ecg_data['AVR'])
        features['AVR_positive'] = 1 if avr_mean > 0.1 else 0
        features['AVR_mean_amplitude'] = avr_mean

        # Check for flat line indicators
        for lead in ['I', 'II', 'III']:
            std_dev = np.std(ecg_data[lead])
            features[f'{lead}_flat_line'] = 1 if std_dev < 0.05 else 0
            features[f'{lead}_std_dev'] = std_dev

        return features

    def _extract_amplitude_features(self, ecg_data):
        """
        Extract amplitude-based features
        """
        features = {}

        for lead in ecg_data.keys():
            signal = ecg_data[lead]
            features[f'{lead}_max'] = np.max(signal)
            features[f'{lead}_min'] = np.min(signal)
            features[f'{lead}_range'] = np.max(signal) - np.min(signal)
            features[f'{lead}_rms'] = np.sqrt(np.mean(signal**2))

        return features

    def _extract_morphology_features(self, ecg_data):
        """
        Extract morphological features
        """
        features = {}

        for lead in ecg_data.keys():
            signal = ecg_data[lead]

            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
            features[f'{lead}_zero_crossings'] = zero_crossings

            # Peak detection
            peaks = self._find_peaks(signal)
            features[f'{lead}_n_peaks'] = len(peaks)

            # Skewness and kurtosis
            features[f'{lead}_skewness'] = self._calculate_skewness(signal)
            features[f'{lead}_kurtosis'] = self._calculate_kurtosis(signal)

        return features

    def _extract_correlation_features(self, ecg_data):
        """
        Extract cross-lead correlation features
        Important for detecting precordial reversals
        """
        features = {}

        leads = list(ecg_data.keys())

        # Calculate correlations between specific lead pairs
        important_pairs = [
            ('I', 'V6'),  # Should be similar polarity
            ('II', 'AVF'),  # Related leads
            ('V1', 'V2'),  # Adjacent precordial leads
            ('V5', 'V6'),  # Adjacent precordial leads
        ]

        for lead1, lead2 in important_pairs:
            if lead1 in leads and lead2 in leads:
                corr = np.corrcoef(ecg_data[lead1], ecg_data[lead2])[0, 1]
                features[f'corr_{lead1}_{lead2}'] = corr

        return features

    def _extract_axis_features(self, ecg_data):
        """
        Extract axis-related features
        """
        features = {}

        # Simplified axis calculation based on leads I and AVF
        if 'I' in ecg_data and 'AVF' in ecg_data:
            lead_I_amplitude = np.mean(ecg_data['I'])
            lead_AVF_amplitude = np.mean(ecg_data['AVF'])

            # Quadrant determination
            if lead_I_amplitude > 0 and lead_AVF_amplitude > 0:
                quadrant = 1  # Normal axis
            elif lead_I_amplitude < 0 and lead_AVF_amplitude > 0:
                quadrant = 2  # Right axis deviation
            elif lead_I_amplitude < 0 and lead_AVF_amplitude < 0:
                quadrant = 3  # Extreme axis deviation
            else:
                quadrant = 4  # Left axis deviation

            features['axis_quadrant'] = quadrant
            features['lead_I_amplitude'] = lead_I_amplitude
            features['lead_AVF_amplitude'] = lead_AVF_amplitude

        return features

    def _find_peaks(self, signal, threshold=None):
        """Simple peak detection"""
        if threshold is None:
            threshold = 0.3 * np.max(np.abs(signal))

        peaks = []
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and
                signal[i] > signal[i+1] and
                signal[i] > threshold):
                peaks.append(i)
        return peaks

    def _calculate_skewness(self, signal):
        """Calculate skewness of signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        return np.mean(((signal - mean) / std) ** 3)

    def _calculate_kurtosis(self, signal):
        """Calculate kurtosis of signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        return np.mean(((signal - mean) / std) ** 4) - 3

# ========================================
# DATASET GENERATION
# ========================================

def generate_comprehensive_dataset(n_samples=1000):
    """
    Generate a comprehensive dataset of normal and misplaced ECGs
    """
    print("Generating comprehensive ECG dataset...")

    simulator = ECGLeadMisplacementSimulator()
    feature_extractor = ECGFeatureExtractor()

    all_features = []
    all_labels = []

    # Generate samples for each class
    classes = {
        'normal': n_samples // 6,
        'ra_la_reversal': n_samples // 6,
        'ra_ll_reversal': n_samples // 6,
        'la_ll_reversal': n_samples // 6,
        'precordial_reversal': n_samples // 6,
        'limb_ground_reversal': n_samples // 6
    }

    for class_name, n_class_samples in classes.items():
        print(f"Generating {n_class_samples} samples for class: {class_name}")

        for i in range(n_class_samples):
            # Generate normal ECG
            normal_ecg = simulator.generate_normal_12_lead_ecg()

            # Apply misplacement based on class
            if class_name == 'normal':
                ecg_data = normal_ecg
            elif class_name == 'ra_la_reversal':
                ecg_data = simulator.simulate_ra_la_reversal(normal_ecg)
            elif class_name == 'ra_ll_reversal':
                ecg_data = simulator.simulate_ra_ll_reversal(normal_ecg)
            elif class_name == 'la_ll_reversal':
                ecg_data = simulator.simulate_la_ll_reversal(normal_ecg)
            elif class_name == 'precordial_reversal':
                # Random precordial reversal
                precordial_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                lead1, lead2 = np.random.choice(precordial_leads, 2, replace=False)
                ecg_data = simulator.simulate_precordial_reversal(normal_ecg, lead1, lead2)
            elif class_name == 'limb_ground_reversal':
                limb = np.random.choice(['RA', 'LA'])
                ecg_data = simulator.simulate_limb_ground_reversal(normal_ecg, limb)

            # Extract features
            features = feature_extractor.extract_features(ecg_data)

            all_features.append(features)
            all_labels.append(class_name)

    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    df['label'] = all_labels

    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Classes: {df['label'].value_counts()}")

    return df

# Generate the dataset
dataset = generate_comprehensive_dataset(n_samples=1800)

# ========================================
# DATA PREPROCESSING
# ========================================

def preprocess_data(df):
    """
    Preprocess the dataset for machine learning
    """
    print("Preprocessing dataset...")

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Handle missing values
    X = X.fillna(X.mean())

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

# Preprocess the data
X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(dataset)

# ========================================
# MACHINE LEARNING MODELS
# ========================================

def train_random_forest_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest classifier
    """
    print("\nTraining Random Forest Classifier...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")

    # Classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return rf_model

def train_neural_network_model(X_train, y_train, X_test, y_test):
    """
    Train a Neural Network classifier
    """
    print("\nTraining Neural Network Classifier...")

    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

    nn_model.fit(X_train, y_train)

    # Predictions
    y_pred = nn_model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Neural Network Accuracy: {accuracy:.4f}")

    # Classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return nn_model

def train_deep_learning_model(X_train, y_train, X_test, y_test, n_classes):
    """
    Train a deep learning model using TensorFlow/Keras
    """
    print("\nTraining Deep Learning Model...")

    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, n_classes)
    y_test_cat = keras.utils.to_categorical(y_test, n_classes)

    # Build model
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),

        layers.Dense(n_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        X_train, y_train_cat,
        batch_size=32,
        epochs=50,
        validation_data=(X_test, y_test_cat),
        verbose=0
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Deep Learning Model Accuracy: {test_accuracy:.4f}")

    # Evaluate on test set and print classification report
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    class_names = label_encoder.classes_
    print("\nDeep Learning Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))


    return model, history

# Train all models
print("Training multiple machine learning models...")

# Random Forest
rf_model = train_random_forest_model(X_train, y_train, X_test, y_test)

# Neural Network
nn_model = train_neural_network_model(X_train, y_train, X_test, y_test)

# Deep Learning
n_classes = len(label_encoder.classes_)
dl_model, dl_history = train_deep_learning_model(X_train, y_train, X_test, y_test, n_classes)

# ========================================
# FEATURE IMPORTANCE ANALYSIS
# ========================================

def analyze_feature_importance(model, feature_names, top_k=20):
    """
    Analyze and visualize feature importance
    """
    print(f"\nTop {top_k} Most Important Features:")

    # Get feature importance
    # Check if the model has feature_importances_ attribute (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Check if the model is a Keras model (for deep learning models)
    elif isinstance(model, tf.keras.Model):
        # For deep learning models, feature importance is not directly available
        # You might use techniques like SHAP or permutation importance
        print("Feature importance is not directly available for Deep Learning models.")
        return None # Or implement a different feature importance method for DL
    else:
        print(f"Feature importance not supported for this model type: {type(model)}")
        return None


    # Create DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_k)

    print(feature_importance_df)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title(f'Top {top_k} Feature Importances for ECG Lead Misplacement Detection')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

    return feature_importance_df

# Analyze feature importance for Random Forest
feature_names = dataset.drop('label', axis=1).columns.tolist()
feature_importance_df = analyze_feature_importance(rf_model, feature_names)

# ========================================
# VISUALIZATION AND ANALYSIS
# ========================================

def plot_training_history(history):
    """
    Plot training history for deep learning model
    """
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(dl_history)

def visualize_confusion_matrix(y_true, y_pred, class_names):
    """
    Create and visualize confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ECG Lead Misplacement Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Generate confusion matrix for Random Forest
rf_pred = rf_model.predict(X_test)
class_names = label_encoder.classes_
visualize_confusion_matrix(y_test, rf_pred, class_names)

# ========================================
# CLINICAL DECISION SUPPORT SYSTEM
# ========================================

class ECGMisplacementDetector:
    """
    Clinical decision support system for ECG lead misplacement detection
    """

    def __init__(self, model, scaler, label_encoder, feature_extractor):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_extractor = feature_extractor

    def predict_misplacement(self, ecg_data):
        """
        Predict misplacement for a new 12-lead ECG data sample
        Returns the predicted label and prediction probabilities.
        """
        # Extract features from the new ECG data
        features = self.feature_extractor.extract_features(ecg_data)

        # Convert features to DataFrame and handle potential missing columns
        features_df = pd.DataFrame([features])

        # Align columns with the training data columns (important for consistent scaling)
        # Assuming the training data columns are available from the `dataset` variable
        training_columns = dataset.drop('label', axis=1).columns
        features_df = features_df.reindex(columns=training_columns, fill_value=0)

        # Scale the features
        scaled_features = self.scaler.transform(features_df)

        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            # For scikit-learn models with probability prediction
            probabilities = self.model.predict_proba(scaled_features)[0]
            predicted_class_index = np.argmax(probabilities)
            predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
            return predicted_label, dict(zip(self.label_encoder.classes_, probabilities))
        elif hasattr(self.model, 'predict') and isinstance(self.model, tf.keras.Model):
            # For deep learning model (TensorFlow/Keras)
            probabilities = self.model.predict(scaled_features, verbose=0)[0]
            predicted_class_index = np.argmax(probabilities)
            predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
            return predicted_label, dict(zip(self.label_encoder.classes_, probabilities))
        else:
            raise TypeError("Unsupported model type for prediction.")


# ========================================
# DEMONSTRATION
# ========================================

# Choose a model for the detector (e.g., the best tuned Deep Learning model)
detector_model = best_model  # Use the best model from hyperparameter tuning

# Initialize the detector
misplacement_detector = ECGMisplacementDetector(
    model=detector_model,
    scaler=scaler,
    label_encoder=label_encoder,
    feature_extractor=ECGFeatureExtractor()
)

# --- Demonstrate with a normal ECG ---
print("\nTesting detector with a normal ECG...")
simulator = ECGLeadMisplacementSimulator()
normal_ecg_sample = simulator.generate_normal_12_lead_ecg()
predicted_normal, probabilities_normal = misplacement_detector.predict_misplacement(normal_ecg_sample)
print(f"Predicted class for normal ECG: {predicted_normal}")
print(f"Probabilities: {probabilities_normal}")

# --- Demonstrate with a RA/LA reversal ECG ---
print("\nTesting detector with a RA/LA reversal ECG...")
ra_la_misplaced_ecg_sample = simulator.simulate_ra_la_reversal(simulator.generate_normal_12_lead_ecg())
predicted_ra_la, probabilities_ra_la = misplacement_detector.predict_misplacement(ra_la_misplaced_ecg_sample)
print(f"Predicted class for RA/LA reversal ECG: {predicted_ra_la}")
print(f"Probabilities: {probabilities_ra_la}")

# --- Demonstrate with a V1/V2 precordial reversal ECG ---
print("\nTesting detector with a V1/V2 precordial reversal ECG...")
precordial_misplaced_ecg_sample = simulator.simulate_precordial_reversal(simulator.generate_normal_12_lead_ecg(), lead1='V1', lead2='V2')
predicted_precordial, probabilities_precordial = misplacement_detector.predict_misplacement(precordial_misplaced_ecg_sample)
print(f"Predicted class for precordial reversal ECG: {predicted_precordial}")
print(f"Probabilities: {probabilities_precordial}")


print("\nECG Lead Misplacement Detection Application Complete.")
