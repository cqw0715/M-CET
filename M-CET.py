import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Reshape, Multiply, Conv2D, Concatenate, Lambda,GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv1D, Permute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ==================== Attention Module Definitions ====================

# CBAM Module
def channel_attention(input_tensor, ratio=16):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    channels = input_tensor.shape[channel_axis]
    
    shared_layer_one = Dense(channels//ratio, activation='relu', use_bias=False)
    shared_layer_two = Dense(channels, use_bias=False)
    
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channels))(avg_pool) if channel_axis == -1 else Reshape((channels, 1, 1))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channels))(max_pool) if channel_axis == -1 else Reshape((channels, 1, 1))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    channel_attention = tf.keras.layers.Add()([avg_pool, max_pool])
    channel_attention = tf.keras.layers.Activation('sigmoid')(channel_attention)
    
    return Multiply()([input_tensor, channel_attention])

def spatial_attention(input_tensor, kernel_size=7):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    
    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=channel_axis, keepdims=True))(input_tensor)
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=channel_axis, keepdims=True))(input_tensor)
    concat = Concatenate(axis=channel_axis)([avg_pool, max_pool])
    
    spatial_attention = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same', 
                               activation='sigmoid', use_bias=False)(concat)
    
    return Multiply()([input_tensor, spatial_attention])

def cbam_block(input_tensor, ratio=16, kernel_size=7):
    x = channel_attention(input_tensor, ratio)
    x = spatial_attention(x, kernel_size)
    return x

# ECA Module
def eca_block(input_tensor, kernel_size=3):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    channels = input_tensor.shape[channel_axis]
    
    gap = GlobalAveragePooling1D()(input_tensor)
    gap = Reshape((1, channels))(gap)
    conv = Conv1D(1, kernel_size=kernel_size, padding='same', use_bias=False)(gap)
    attention = tf.keras.activations.sigmoid(conv)
    attention = Reshape((channels,))(attention)
    
    return Multiply()([input_tensor, attention])

# Triplet Attention Module
class BasicConv(tf.keras.layers.Layer):
    def __init__(self, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = Conv2D(out_planes, kernel_size=kernel_size, strides=stride, 
                          padding='same' if padding else 'valid', dilation_rate=dilation,
                          groups=groups, use_bias=bias)
        self.bn = BatchNormalization() if bn else None
        self.relu = tf.keras.layers.ReLU() if relu else None

    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(tf.keras.layers.Layer):
    def call(self, x):
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        return tf.concat([max_pool, avg_pool], axis=-1)

class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(AttentionGate, self).__init__()
        self.compress = ZPool()
        self.conv = BasicConv(1, kernel_size, stride=1, padding=(kernel_size-1)//2, relu=False)

    def call(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = tf.sigmoid(x_out)
        return Multiply()([x, scale])

class TripletAttention(tf.keras.layers.Layer):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def call(self, x):
        x_perm1 = Permute((2, 1, 3))(x)
        x_out1 = self.cw(x_perm1)
        x_out11 = Permute((2, 1, 3))(x_out1)
        
        x_perm2 = Permute((3, 2, 1))(x)
        x_out2 = self.hc(x_perm2)
        x_out21 = Permute((3, 2, 1))(x_out2)
        
        if not self.no_spatial:
            x_out = self.hw(x)
            return (x_out + x_out11 + x_out21) / 3.0
        else:
            return (x_out11 + x_out21) / 2.0

# ==================== Data Preparation and Feature Engineering ====================

# Read complete dataset
all_data = pd.read_csv('All.csv', usecols=['Sequence', 'Label'])

# Define amino acid physicochemical properties
HYDROPHOBICITY = {'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
                  'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
                  'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
                  'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26}

HYDROPHILICITY = {'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5,
                  'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8,
                  'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0,
                  'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3}

MASS = {'A': 15, 'C': 47, 'D': 59, 'E': 73, 'F': 91,
        'G': 1, 'H': 82, 'I': 57, 'K': 72, 'L': 57,
        'M': 75, 'N': 58, 'P': 42, 'Q': 72, 'R': 101,
        'S': 31, 'T': 45, 'V': 43, 'W': 130, 'Y': 107}

AMINO_ACIDS = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
DIPEPTIDES = [aa1 + aa2 for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS]
DIPEPTIDE_INDEX = {d: i for i, d in enumerate(DIPEPTIDES)}

def compute_aac(sequence):
    total = len(sequence)
    return [sequence.count(aa)/total if total > 0 else 0 for aa in AMINO_ACIDS]

def compute_dpc(sequence):
    total = max(len(sequence) - 1, 1)
    dpc = [0] * 400
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if dipeptide in DIPEPTIDE_INDEX:
            dpc[DIPEPTIDE_INDEX[dipeptide]] += 1
    return [count/total for count in dpc]

def compute_pseaac(sequence, lambda_value=5, w=0.05):
    aac = [sequence.count(aa)/len(sequence) for aa in AMINO_ACIDS]
    theta_features = []
    for prop_dict in [HYDROPHOBICITY, HYDROPHILICITY, MASS]:
        prop_values = [prop_dict.get(aa, 0) for aa in sequence]
        theta = []
        for lag in range(1, lambda_value+1):
            if len(sequence) > lag:
                sum_val = sum(
                    (prop_values[i] - prop_values[i+lag])**2 
                    for i in range(len(sequence)-lag)
                ) / (len(sequence)-lag)
                theta.append(sum_val)
            else:
                theta.append(0)
        theta_features.extend(theta)
    return aac + [w*val for val in theta_features]

def extract_combined_features(sequence):
    aac = compute_aac(sequence)
    dpc = compute_dpc(sequence)
    pseaac = compute_pseaac(sequence)
    return np.concatenate([aac, dpc, pseaac])

# Feature extraction
X_all = np.array([extract_combined_features(seq) for seq in all_data['Sequence']])
y_all = all_data['Label'].values

# Data standardization
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

# ==================== Model Construction ====================

def build_cbam_model(input_dim=455):
    inputs = Input(shape=(input_dim,), name='input_layer')  
    
    # First layer
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)  
    x = BatchNormalization()(x)
    x_reshaped = Reshape((1, 512, 1))(x)
    x_cbam = cbam_block(x_reshaped, ratio=16, kernel_size=7)
    x_cbam = Reshape((512,))(x_cbam)
    x = Dropout(0.5)(x_cbam)  

    # Second layer
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  
    x = BatchNormalization()(x)
    x_reshaped = Reshape((1, 256, 1))(x)
    x_cbam = cbam_block(x_reshaped, ratio=16, kernel_size=7)
    x_cbam = Reshape((256,))(x_cbam)
    x = Dropout(0.4)(x_cbam)  

    # Third layer
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  
    x = BatchNormalization()(x)
    x_reshaped = Reshape((1, 128, 1))(x)
    x_cbam = cbam_block(x_reshaped, ratio=16, kernel_size=7)
    x_cbam = Reshape((128,))(x_cbam)
    x = Dropout(0.3)(x_cbam)  

    output = Dense(1, activation='sigmoid')(x)  

    model = Model(inputs=inputs, outputs=output)  
    
    optimizer = Adam(learning_rate=0.0005)  
    model.compile(  
        optimizer=optimizer,  
        loss='binary_crossentropy',  
        metrics=[  
            'accuracy',  
            tf.keras.metrics.AUC(name='auc'),  
            tf.keras.metrics.Precision(name='precision'),  
            tf.keras.metrics.Recall(name='recall'),  
            tf.keras.metrics.AUC(curve='PR', name='pr_auc')  
        ]  
    )  
    
    return model 

def build_eca_model(input_dim=455):
    inputs = Input(shape=(input_dim,), name='input_layer')  
    
    # First layer
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)  
    x = BatchNormalization()(x)
    x = eca_block(Reshape((512, 1))(x))
    x = Reshape((512,))(x)
    x = Dropout(0.5)(x)  

    # Second layer
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  
    x = BatchNormalization()(x)
    x = eca_block(Reshape((256, 1))(x))
    x = Reshape((256,))(x)
    x = Dropout(0.4)(x)  

    # Third layer
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  
    x = BatchNormalization()(x)
    x = eca_block(Reshape((128, 1))(x))
    x = Reshape((128,))(x)
    x = Dropout(0.3)(x)  

    output = Dense(1, activation='sigmoid')(x)  

    model = Model(inputs=inputs, outputs=output)  
    
    optimizer = Adam(learning_rate=0.0005)  
    model.compile(  
        optimizer=optimizer,  
        loss='binary_crossentropy',  
        metrics=[  
            'accuracy',  
            tf.keras.metrics.AUC(name='auc'),  
            tf.keras.metrics.Precision(name='precision'),  
            tf.keras.metrics.Recall(name='recall'),  
            tf.keras.metrics.AUC(curve='PR', name='pr_auc')  
        ]  
    )  
    
    return model 

def build_triplet_model(input_dim=455):
    inputs = Input(shape=(input_dim,), name='input_layer')  
    
    # First layer
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)  
    x = BatchNormalization()(x)
    x_reshaped = Reshape((1, 512, 1))(x)
    x_att = TripletAttention(no_spatial=False)(x_reshaped)
    x_att = Reshape((512,))(x_att)
    x = Dropout(0.5)(x_att)  

    # Second layer
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  
    x = BatchNormalization()(x)
    x_reshaped = Reshape((1, 256, 1))(x)
    x_att = TripletAttention(no_spatial=False)(x_reshaped)
    x_att = Reshape((256,))(x_att)
    x = Dropout(0.4)(x_att)  

    # Third layer
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  
    x = BatchNormalization()(x)
    x_reshaped = Reshape((1, 128, 1))(x)
    x_att = TripletAttention(no_spatial=False)(x_reshaped)
    x_att = Reshape((128,))(x_att)
    x = Dropout(0.3)(x_att)  

    output = Dense(1, activation='sigmoid')(x)  

    model = Model(inputs=inputs, outputs=output)  
    
    optimizer = Adam(learning_rate=0.0005)  
    model.compile(  
        optimizer=optimizer,  
        loss='binary_crossentropy',  
        metrics=[  
            'accuracy',  
            tf.keras.metrics.AUC(name='auc'),  
            tf.keras.metrics.Precision(name='precision'),  
            tf.keras.metrics.Recall(name='recall'),  
            tf.keras.metrics.AUC(curve='PR', name='pr_auc')  
        ]  
    )  
    
    return model 

# ==================== Ensemble Model Class ====================

class EnsembleModel:
    def __init__(self, model_paths, weights=None):
        self.models = [load_model(path, custom_objects={
            'TripletAttention': TripletAttention,
            'AttentionGate': AttentionGate,
            'ZPool': ZPool,
            'BasicConv': BasicConv
        }) for path in model_paths]
        self.weights = weights if weights else [1.0, 1.0, 1.0]
        
    def predict_proba(self, X):
        predictions = [model.predict(X).flatten() for model in self.models]
        return np.array(predictions)
    
    def predict_weighted(self, X, weights=None):
        if weights is None:
            weights = self.weights
        predictions = self.predict_proba(X)
        weighted_avg = np.average(predictions, axis=0, weights=weights)
        return (weighted_avg > 0.5).astype(int), weighted_avg
    
    def predict_voting(self, X):
        predictions = self.predict_proba(X)
        binary_preds = (predictions > 0.5).astype(int)
        vote_pred = np.sum(binary_preds, axis=0)
        final_pred = (vote_pred >= 2).astype(int)
        final_proba = np.mean(predictions, axis=0)
        return final_pred, final_proba
    
    def evaluate(self, X, y, method='weighted', weights=None):
        if method == 'weighted':
            y_pred, y_proba = self.predict_weighted(X, weights)
        else:
            y_pred, y_proba = self.predict_voting(X)
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'specificity': tn/(tn+fp),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }

# ==================== Stacking Ensemble Class ====================

class StackingModel:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        # First train all base models
        for model in self.base_models:
            model.fit(X, y)
        
        # Generate base model predictions as new features
        base_preds = [model.predict(X).reshape(-1, 1) for model in self.base_models]
        X_meta = np.concatenate(base_preds, axis=1)
        
        # Train meta model
        self.meta_model.fit(X_meta, y)
    
    def predict_proba(self, X):
        # Get base model predictions
        base_preds = [model.predict(X).reshape(-1, 1) for model in self.base_models]
        X_meta = np.concatenate(base_preds, axis=1)
        
        # Return meta model prediction probabilities
        return self.meta_model.predict_proba(X_meta)[:, 1]
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'specificity': tn/(tn+fp),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }

# ==================== Training and Evaluation Process ====================

def train_and_evaluate():
    n_splits = 10
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for all models and ensembles
    cbam_results = []
    eca_results = []
    triplet_results = []
    weighted_results = []
    voting_results = []
    stacking_results = []  
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_all, y_all)):
        print(f"\n=== Fold {fold+1}/{n_splits} Cross Validation ===")
        
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # Training configuration
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]
        
        # Train CBAM model
        print("\nTraining CBAM model...")
        cbam_model = build_cbam_model()
        cbam_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        cbam_model.save(f'MLP_fold{fold+1}_cbam.h5')
        
        # Evaluate CBAM model
        y_pred_proba = cbam_model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cbam_results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'specificity': tn/(tn+fp),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        })
        print(f"CBAM Model - Accuracy: {cbam_results[-1]['accuracy']:.4f} | AUC: {cbam_results[-1]['roc_auc']:.4f}")
        
        # Train ECA model
        print("\nTraining ECA model...")
        eca_model = build_eca_model()
        eca_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        eca_model.save(f'MLP_fold{fold+1}_eca.h5')
        
        # Evaluate ECA model
        y_pred_proba = eca_model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        eca_results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'specificity': tn/(tn+fp),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        })
        print(f"ECA Model - Accuracy: {eca_results[-1]['accuracy']:.4f} | AUC: {eca_results[-1]['roc_auc']:.4f}")
        
        # Train Triplet model
        print("\nTraining Triplet model...")
        triplet_model = build_triplet_model()
        triplet_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        triplet_model.save(f'MLP_fold{fold+1}_triplet.h5')
        
        # Evaluate Triplet model
        y_pred_proba = triplet_model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        triplet_results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'specificity': tn/(tn+fp),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        })
        print(f"Triplet Model - Accuracy: {triplet_results[-1]['accuracy']:.4f} | AUC: {triplet_results[-1]['roc_auc']:.4f}")
        
        # Create ensemble model
        model_paths = [
            f'MLP_fold{fold+1}_cbam.h5',
            f'MLP_fold{fold+1}_eca.h5',
            f'MLP_fold{fold+1}_triplet.h5'
        ]
        
        # Calculate weights (based on each model's AUC)
        model_aucs = [
            cbam_results[-1]['roc_auc'],
            eca_results[-1]['roc_auc'],
            triplet_results[-1]['roc_auc']
        ]
        weights = [auc/sum(model_aucs) for auc in model_aucs]
        
        ensemble = EnsembleModel(model_paths, weights=weights)
        
        # Evaluate weighted average method
        weighted_result = ensemble.evaluate(X_test, y_test, method='weighted')
        weighted_results.append(weighted_result)
        print(f"Weighted Average Ensemble - Accuracy: {weighted_result['accuracy']:.4f} | AUC: {weighted_result['roc_auc']:.4f}")
        
        # Evaluate voting method
        voting_result = ensemble.evaluate(X_test, y_test, method='voting')
        voting_results.append(voting_result)
        print(f"Voting Ensemble - Accuracy: {voting_result['accuracy']:.4f} | AUC: {voting_result['roc_auc']:.4f}")
        
        # Stacking ensemble
        print("\nTraining Stacking ensemble...")
        # Load base models
        base_models = [
            load_model(f'MLP_fold{fold+1}_cbam.h5', custom_objects={
                'TripletAttention': TripletAttention,
                'AttentionGate': AttentionGate,
                'ZPool': ZPool,
                'BasicConv': BasicConv
            }),
            load_model(f'MLP_fold{fold+1}_eca.h5', custom_objects={
                'TripletAttention': TripletAttention,
                'AttentionGate': AttentionGate,
                'ZPool': ZPool,
                'BasicConv': BasicConv
            }),
            load_model(f'MLP_fold{fold+1}_triplet.h5', custom_objects={
                'TripletAttention': TripletAttention,
                'AttentionGate': AttentionGate,
                'ZPool': ZPool,
                'BasicConv': BasicConv
            })
        ]
        
        # Define meta model (simple logistic regression)
        meta_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        
        # Create Stacking model
        stacking_model = StackingModel(base_models, meta_model)
        
        # Train Stacking model
        stacking_model.fit(X_train, y_train)
        
        # Evaluate Stacking model
        stacking_result = stacking_model.evaluate(X_test, y_test)
        stacking_results.append(stacking_result)
        print(f"Stacking Ensemble - Accuracy: {stacking_result['accuracy']:.4f} | AUC: {stacking_result['roc_auc']:.4f}")
        
    # Result statistics function
    def calculate_stats(results):
        avg = {
            'accuracy': np.mean([r['accuracy'] for r in results]),
            'precision': np.mean([r['precision'] for r in results]),
            'recall': np.mean([r['recall'] for r in results]),
            'specificity': np.mean([r['specificity'] for r in results]),
            'f1_score': np.mean([r['f1_score'] for r in results]),
            'roc_auc': np.mean([r['roc_auc'] for r in results])
        }
        std = {
            'accuracy': np.std([r['accuracy'] for r in results]),
            'precision': np.std([r['precision'] for r in results]),
            'recall': np.std([r['recall'] for r in results]),
            'specificity': np.std([r['specificity'] for r in results]),
            'f1_score': np.std([r['f1_score'] for r in results]),
            'roc_auc': np.std([r['roc_auc'] for r in results])
        }
        return avg, std

    # Calculate statistics for all models and ensemble methods
    cbam_avg, cbam_std = calculate_stats(cbam_results)
    eca_avg, eca_std = calculate_stats(eca_results)
    triplet_avg, triplet_std = calculate_stats(triplet_results)
    weighted_avg, weighted_std = calculate_stats(weighted_results)
    voting_avg, voting_std = calculate_stats(voting_results)
    stacking_avg, stacking_std = calculate_stats(stacking_results) 

    # ==================== Result Output ====================

    print("\n\n=== Final Evaluation Results ===")
    
    print("\nCBAM Model Results (±Std Dev):")
    print(f"Accuracy: {cbam_avg['accuracy']:.4f} ± {cbam_std['accuracy']:.4f}")
    print(f"Precision: {cbam_avg['precision']:.4f} ± {cbam_std['precision']:.4f}")
    print(f"Recall: {cbam_avg['recall']:.4f} ± {cbam_std['recall']:.4f}")
    print(f"Specificity: {cbam_avg['specificity']:.4f} ± {cbam_std['specificity']:.4f}")
    print(f"F1 Score: {cbam_avg['f1_score']:.4f} ± {cbam_std['f1_score']:.4f}")
    print(f"ROC AUC: {cbam_avg['roc_auc']:.4f} ± {cbam_std['roc_auc']:.4f}")

    print("\nECA Model Results (±Std Dev):")
    print(f"Accuracy: {eca_avg['accuracy']:.4f} ± {eca_std['accuracy']:.4f}")
    print(f"Precision: {eca_avg['precision']:.4f} ± {eca_std['precision']:.4f}")
    print(f"Recall: {eca_avg['recall']:.4f} ± {eca_std['recall']:.4f}")
    print(f"Specificity: {eca_avg['specificity']:.4f} ± {eca_std['specificity']:.4f}")
    print(f"F1 Score: {eca_avg['f1_score']:.4f} ± {eca_std['f1_score']:.4f}")
    print(f"ROC AUC: {eca_avg['roc_auc']:.4f} ± {eca_std['roc_auc']:.4f}")

    print("\nTriplet Model Results (±Std Dev):")
    print(f"Accuracy: {triplet_avg['accuracy']:.4f} ± {triplet_std['accuracy']:.4f}")
    print(f"Precision: {triplet_avg['precision']:.4f} ± {triplet_std['precision']:.4f}")
    print(f"Recall: {triplet_avg['recall']:.4f} ± {triplet_std['recall']:.4f}")
    print(f"Specificity: {triplet_avg['specificity']:.4f} ± {triplet_std['specificity']:.4f}")
    print(f"F1 Score: {triplet_avg['f1_score']:.4f} ± {triplet_std['f1_score']:.4f}")
    print(f"ROC AUC: {triplet_avg['roc_auc']:.4f} ± {triplet_std['roc_auc']:.4f}")

    print("\nWeighted Average Ensemble Results (±Std Dev):")
    print(f"Accuracy: {weighted_avg['accuracy']:.4f} ± {weighted_std['accuracy']:.4f}")
    print(f"Precision: {weighted_avg['precision']:.4f} ± {weighted_std['precision']:.4f}")
    print(f"Recall: {weighted_avg['recall']:.4f} ± {weighted_std['recall']:.4f}")
    print(f"Specificity: {weighted_avg['specificity']:.4f} ± {weighted_std['specificity']:.4f}")
    print(f"F1 Score: {weighted_avg['f1_score']:.4f} ± {weighted_std['f1_score']:.4f}")
    print(f"ROC AUC: {weighted_avg['roc_auc']:.4f} ± {weighted_std['roc_auc']:.4f}")

    print("\nVoting Ensemble Results (±Std Dev):")
    print(f"Accuracy: {voting_avg['accuracy']:.4f} ± {voting_std['accuracy']:.4f}")
    print(f"Precision: {voting_avg['precision']:.4f} ± {voting_std['precision']:.4f}")
    print(f"Recall: {voting_avg['recall']:.4f} ± {voting_std['recall']:.4f}")
    print(f"Specificity: {voting_avg['specificity']:.4f} ± {voting_std['specificity']:.4f}")
    print(f"F1 Score: {voting_avg['f1_score']:.4f} ± {voting_std['f1_score']:.4f}")
    print(f"ROC AUC: {voting_avg['roc_auc']:.4f} ± {voting_std['roc_auc']:.4f}")

    print("\nStacking Ensemble Results (±Std Dev):")
    print(f"Accuracy: {stacking_avg['accuracy']:.4f} ± {stacking_std['accuracy']:.4f}")
    print(f"Precision: {stacking_avg['precision']:.4f} ± {stacking_std['precision']:.4f}")
    print(f"Recall: {stacking_avg['recall']:.4f} ± {stacking_std['recall']:.4f}")
    print(f"Specificity: {stacking_avg['specificity']:.4f} ± {stacking_std['specificity']:.4f}")
    print(f"F1 Score: {stacking_avg['f1_score']:.4f} ± {stacking_std['f1_score']:.4f}")
    print(f"ROC AUC: {stacking_avg['roc_auc']:.4f} ± {stacking_std['roc_auc']:.4f}")

    # Return all results for further analysis
    return {
        'cbam': {'avg': cbam_avg, 'std': cbam_std},
        'eca': {'avg': eca_avg, 'std': eca_std},
        'triplet': {'avg': triplet_avg, 'std': triplet_std},
        'weighted': {'avg': weighted_avg, 'std': weighted_std},
        'voting': {'avg': voting_avg, 'std': voting_std},
        'stacking': {'avg': stacking_avg, 'std': stacking_std} 
    }

# ==================== Execute Training and Evaluation ====================

if __name__ == "__main__":
    # Execute complete training and evaluation process
    final_results = train_and_evaluate()
    
    #Save results to file
    import json
    with open('final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
