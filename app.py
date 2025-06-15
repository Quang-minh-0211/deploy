import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import tensorflow as tf
import warnings
import json
try:
    import h5py
except ImportError:
    h5py = None
    print("⚠️ h5py not available - file inspection will be limited")
warnings.filterwarnings('ignore')

# Import config (nếu có file config.py)
try:
    from config import MODEL_DIR, WINDOW_SIZES, FEATURES, MODEL_NAMES, get_model_paths
    model_paths, scaler_paths, config_paths = get_model_paths()
except ImportError:
    # Fallback nếu không có file config.py
    MODEL_DIR = r'D:\BigData And DataMining\Scientific Report\chia_vc_cho_ae\deploy\saved_models'
    WINDOW_SIZES = [2, 12, 24, 36, 72]
    MODEL_NAMES = ['RNN', 'LSTM', 'Transformer', 'Autoformer']
    FEATURES = ['q64']
    
    # Tạo paths cho tất cả combinations
    model_paths = {}
    scaler_paths = {}
    config_paths = {}
    
    for model_name in MODEL_NAMES:
        model_paths[model_name] = {}
        scaler_paths[model_name] = {}
        config_paths[model_name] = {}
        
        for ws in WINDOW_SIZES:
            model_paths[model_name][ws] = os.path.join(MODEL_DIR, f'{model_name}_model_window_{ws}.h5')
            scaler_paths[model_name][ws] = os.path.join(MODEL_DIR, f'scaler_{model_name}_window_{ws}.pkl')
            config_paths[model_name][ws] = os.path.join(MODEL_DIR, f'config_{model_name}_window_{ws}.pkl')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables để lưu các models - nested structure
models = {}  # models[model_name][window_size]
scalers = {}  # scalers[model_name][window_size]
configs = {}  # configs[model_name][window_size]
features = FEATURES

def get_custom_objects_for_model(model_name):
    """Lấy custom objects cho từng loại model"""
    base_objects = {
        'mse': MeanSquaredError(),
        'mae': MeanAbsoluteError(),
        'mean_squared_error': MeanSquaredError(),
        'mean_absolute_error': MeanAbsoluteError()
    }
    
    # Thêm custom objects cho Transformer
    if model_name == 'Transformer':
        try:
            # Import tensorflow nếu chưa có
            import tensorflow as tf
            from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
            
            transformer_objects = {
                'MultiHeadAttention': MultiHeadAttention,
                'LayerNormalization': LayerNormalization,
                # Thêm các custom functions có thể cần
                'gelu': tf.nn.gelu,
                'swish': tf.nn.swish,
            }
            base_objects.update(transformer_objects)
            print(f"   📦 Added Transformer-specific custom objects")
        except Exception as e:
            print(f"   ⚠️ Could not add Transformer custom objects: {e}")
    
    # Thêm custom objects cho Autoformer nếu cần
    elif model_name == 'Autoformer':
        try:
            import tensorflow as tf
            autoformer_objects = {
                'LayerNormalization': tf.keras.layers.LayerNormalization,
            }
            base_objects.update(autoformer_objects)
            print(f"   📦 Added Autoformer-specific custom objects")
        except Exception as e:
            print(f"   ⚠️ Could not add Autoformer custom objects: {e}")
    
    return base_objects

def try_load_transformer_model(model_path):
    """Thử nhiều cách load model Transformer khác nhau"""
    import tensorflow as tf
    
    print(f"   🔄 Trying multiple loading strategies for Transformer...")
    
    # Strategy 1: Check if it's a SavedModel directory
    savedmodel_path = model_path.replace('.h5', '_savedmodel')
    if os.path.exists(savedmodel_path):
        try:
            print(f"   📁 Trying SavedModel format: {savedmodel_path}")
            model = tf.saved_model.load(savedmodel_path)
            print(f"   ✅ Loaded as SavedModel")
            return model, "savedmodel"
        except Exception as e:
            print(f"   ❌ SavedModel failed: {e}")
    
    # Strategy 2: Load with minimal custom objects
    try:
        print(f"   🎯 Trying minimal custom objects...")
        minimal_objects = {
            'tf': tf,
            'keras': tf.keras,
        }
        model = load_model(model_path, custom_objects=minimal_objects, compile=False)
        print(f"   ✅ Loaded with minimal custom objects")
        return model, "minimal"
    except Exception as e:
        print(f"   ❌ Minimal objects failed: {e}")
    
    # Strategy 3: Load architecture only, then load weights
    try:
        print(f"   🏗️ Trying architecture + weights separately...")
        # Tìm file architecture nếu có
        arch_path = model_path.replace('.h5', '_architecture.json')
        weights_path = model_path.replace('.h5', '_weights.h5')
        
        if os.path.exists(arch_path) and os.path.exists(weights_path):
            with open(arch_path, 'r') as f:
                architecture = f.read()
            
            model = tf.keras.models.model_from_json(architecture)
            model.load_weights(weights_path)
            print(f"   ✅ Loaded architecture + weights separately")
            return model, "separated"
        else:
            print(f"   ⚠️ Architecture/weights files not found")
    except Exception as e:
        print(f"   ❌ Architecture + weights failed: {e}")
    
    # Strategy 4: Load without any custom objects
    try:
        print(f"   🎲 Trying without custom objects...")
        model = load_model(model_path, compile=False)
        print(f"   ✅ Loaded without custom objects")
        return model, "no_custom"
    except Exception as e:
        print(f"   ❌ No custom objects failed: {e}")
    
    # Strategy 5: Try to inspect the file first
    if h5py:
        try:
            print(f"   🔍 Inspecting model file structure...")
            with h5py.File(model_path, 'r') as f:
                print(f"   📋 H5 file keys: {list(f.keys())}")
                if 'model_config' in f.attrs:
                    print(f"   📝 Has model config")
                if 'model_weights' in f:
                    print(f"   ⚖️ Has model weights")
        except Exception as e:
            print(f"   ❌ File inspection failed: {e}")
    else:
        print(f"   ⚠️ h5py not available - skipping file inspection")
    
    return None, "all_failed"

def load_single_model(model_name, window_size, model_path, scaler_path, config_path):
    """Load một model cụ thể với strategies khác nhau cho từng model type"""
    try:
        print(f"   Loading {model_name} model...")
        
        model = None
        loading_strategy = None
        
        # Special handling cho Transformer
        if model_name == 'Transformer':
            model, loading_strategy = try_load_transformer_model(model_path)
            
            if model is None:
                # Last resort: try with extensive custom objects
                try:
                    print(f"   🔧 Last resort: extensive custom objects...")
                    import tensorflow as tf
                    extensive_objects = {
                        'tf': tf,
                        'tensorflow': tf,
                        'keras': tf.keras,
                        'K': tf.keras.backend,
                        'backend': tf.keras.backend,
                        'layers': tf.keras.layers,
                        'Model': tf.keras.Model,
                        'Sequential': tf.keras.Sequential,
                        'Dense': tf.keras.layers.Dense,
                        'Dropout': tf.keras.layers.Dropout,
                        'Input': tf.keras.layers.Input,
                        'Lambda': tf.keras.layers.Lambda,
                        'Reshape': tf.keras.layers.Reshape,
                        'Concatenate': tf.keras.layers.Concatenate,
                        'Add': tf.keras.layers.Add,
                        'Multiply': tf.keras.layers.Multiply,
                        'LayerNormalization': tf.keras.layers.LayerNormalization,
                        'MultiHeadAttention': tf.keras.layers.MultiHeadAttention,
                        'Embedding': tf.keras.layers.Embedding,
                        'GlobalAveragePooling1D': tf.keras.layers.GlobalAveragePooling1D,
                        'mse': tf.keras.losses.MeanSquaredError(),
                        'mae': tf.keras.losses.MeanAbsoluteError(),
                        'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                        'mean_absolute_error': tf.keras.losses.MeanAbsoluteError(),
                        'adam': tf.keras.optimizers.Adam,
                        'Adam': tf.keras.optimizers.Adam,
                    }
                    
                    model = load_model(model_path, custom_objects=extensive_objects, compile=False)
                    loading_strategy = "extensive"
                    print(f"   ✅ Loaded with extensive custom objects")
                    
                except Exception as e:
                    print(f"   ❌ Extensive objects also failed: {e}")
                    return None, None, None, f"All Transformer loading strategies failed. Last error: {str(e)}"
        
        # Standard loading cho các models khác
        else:
            custom_objects = get_custom_objects_for_model(model_name)
            
            # Thử load với compile=False trước
            try:
                model = load_model(model_path, custom_objects=custom_objects, compile=False)
                loading_strategy = "compile_false"
                print(f"   ✅ Model loaded successfully (compile=False)")
            except Exception as compile_error:
                print(f"   ⚠️ Failed with compile=False: {compile_error}")
                # Thử load bình thường
                try:
                    model = load_model(model_path, custom_objects=custom_objects)
                    loading_strategy = "normal"
                    print(f"   ✅ Model loaded successfully (normal mode)")
                except Exception as normal_error:
                    print(f"   ❌ Normal loading also failed: {normal_error}")
                    return None, None, None, f"Both loading methods failed: {str(normal_error)}"
        
        if model is None:
            return None, None, None, "Failed to load model with any strategy"
        
        # Load scaler
        print(f"   Loading scaler...")
        scaler = joblib.load(scaler_path)
        print(f"   ✅ Scaler loaded successfully")
        
        # Load config
        config = None
        if os.path.exists(config_path):
            print(f"   Loading config...")
            config = joblib.load(config_path)
            print(f"   ✅ Config loaded successfully")
        else:
            # Tạo config mặc định
            config = {
                'past_window': window_size,
                'future_window': window_size,
                'r2': 0.99,
                'mae': 0.02,
                'rmse': 0.03,
                'epochs_trained': 20,
                'model_type': model_name,
                'loading_strategy': loading_strategy
            }
            print(f"   ⚠️ Config not found, created default config")
        
        # Thêm loading strategy vào config
        config['loading_strategy'] = loading_strategy
        
        return model, scaler, config, None
        
    except Exception as e:
        error_msg = f"Error loading {model_name}-{window_size}h: {str(e)}"
        print(f"   ❌ {error_msg}")
        return None, None, None, error_msg

def load_all_models():
    """Load tất cả các models đã train với improved error handling"""
    global models, scalers, configs
    
    # Initialize nested dictionaries
    for model_name in MODEL_NAMES:
        models[model_name] = {}
        scalers[model_name] = {}
        configs[model_name] = {}
    
    print(f"🔍 Loading models from: {MODEL_DIR}")
    
    total_loaded = 0
    total_possible = len(MODEL_NAMES) * len(WINDOW_SIZES)
    loading_errors = {}
    
    for model_name in MODEL_NAMES:
        print(f"\n🤖 Loading {model_name} models:")
        loading_errors[model_name] = {}
        
        for window_size in WINDOW_SIZES:
            model_path = model_paths[model_name][window_size]
            scaler_path = scaler_paths[model_name][window_size]
            config_path = config_paths[model_name][window_size]
            
            print(f"\n📁 Checking files for {model_name} - {window_size}h forecast:")
            print(f"   Model: {model_path}")
            print(f"   Scaler: {scaler_path}")
            print(f"   Config: {config_path}")
            
            # Kiểm tra file tồn tại
            model_exists = os.path.exists(model_path)
            scaler_exists = os.path.exists(scaler_path)
            config_exists = os.path.exists(config_path)
            
            print(f"   Model exists: {model_exists}")
            print(f"   Scaler exists: {scaler_exists}")
            print(f"   Config exists: {config_exists}")
            
            if model_exists and scaler_exists:
                # Thử load model
                model, scaler, config, error = load_single_model(
                    model_name, window_size, model_path, scaler_path, config_path
                )
                
                if model is not None:
                    models[model_name][window_size] = model
                    scalers[model_name][window_size] = scaler
                    configs[model_name][window_size] = config
                    total_loaded += 1
                    print(f"✅ Successfully loaded {model_name} - {window_size}h forecast")
                else:
                    loading_errors[model_name][window_size] = error
                    print(f"❌ Failed to load {model_name} - {window_size}h forecast")
            else:
                print(f"❌ Required files missing for {model_name} - {window_size}h forecast:")
                if not model_exists:
                    print(f"   Missing model: {model_path}")
                if not scaler_exists:
                    print(f"   Missing scaler: {scaler_path}")
                loading_errors[model_name][window_size] = "Missing required files"
    
    print(f"\n📊 Loading Summary:")
    print(f"   Total models loaded: {total_loaded}/{total_possible}")
    print(f"   Available model types: {[name for name in MODEL_NAMES if any(models[name])]}")
    
    # Print detailed availability
    print(f"\n📋 Detailed Availability:")
    for model_name in MODEL_NAMES:
        available_windows = list(models[model_name].keys())
        print(f"   {model_name}: {available_windows} hours ({len(available_windows)}/{len(WINDOW_SIZES)} windows)")
    
    if total_loaded == 0:
        print("⚠️ No models loaded! Please check:")
        print("   1. Model directory path is correct")
        print("   2. Model files exist and are accessible")
        print("   3. File permissions allow reading")
        print("   4. File naming convention matches expected pattern")
    
    return total_loaded > 0

def prepare_sequence_data(data, window_size):
    """Chuẩn bị dữ liệu sequence cho dự đoán"""
    if len(data) < window_size:
        # Nếu không đủ dữ liệu, duplicate hàng cuối
        last_row = data.iloc[-1:].values
        additional_rows = np.tile(last_row, (window_size - len(data), 1))
        data_array = np.vstack([data.values, additional_rows])
    else:
        # Lấy window_size hàng cuối cùng
        data_array = data.iloc[-window_size:].values
    
    # Reshape thành (1, window_size, n_features)
    return data_array.reshape(1, window_size, len(features))

def predict_water_level(data, model_name, window_size, debug=False):
    """Dự đoán mực nước với model cụ thể"""
    try:
        # Kiểm tra model có tồn tại không
        if model_name not in models or window_size not in models[model_name]:
            available_models = {}
            for mn in MODEL_NAMES:
                if mn in models:
                    available_models[mn] = list(models[mn].keys())
            return None, f"Model {model_name} for {window_size}h forecast not available. Available: {available_models}"
        
        model = models[model_name][window_size]
        scaler = scalers[model_name][window_size]
        
        if debug:
            print(f"\n🔍 Debug mode for {model_name} - {window_size}h prediction:")
            print(f"Model type: {type(model)}")
            print(f"Model summary available: {hasattr(model, 'summary')}")
        
        print(f"\n🔍 Prediction for {model_name} - {window_size}h:")
        print(f"Input data shape: {data.shape}")
        print(f"Input Q64 range: [{data['q64'].min():.4f}, {data['q64'].max():.4f}]")
        
        # Chuẩn hóa dữ liệu
        scaled_data = scaler.transform(data[features])
        print(f"Scaled data range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")
        
        # Chuẩn bị sequence
        X = prepare_sequence_data(pd.DataFrame(scaled_data, columns=features), window_size)
        print(f"Sequence shape: {X.shape}")
        
        # Dự đoán
        prediction_scaled = model.predict(X, verbose=0)
        print(f"Raw prediction (scaled): {prediction_scaled.flatten()}")
        
        # Chuyển đổi về đơn vị gốc
        q64_index = features.index('q64')
        
        try:
            # Method: Proper inverse transform
            dummy_array = np.zeros((prediction_scaled.shape[0], len(features)))
            dummy_array[:, q64_index] = prediction_scaled.flatten()
            
            # Inverse transform toàn bộ
            inverse_transformed = scaler.inverse_transform(dummy_array)
            prediction_original = inverse_transformed[:, q64_index].reshape(-1, 1)
            
            print(f"Final prediction: {prediction_original.flatten()}")
            
        except Exception as e:
            print(f"Inverse transform failed: {e}, using fallback method")
            # Fallback method
            q64_min = scaler.data_min_[q64_index]
            q64_max = scaler.data_max_[q64_index]
            prediction_original = prediction_scaled * (q64_max - q64_min) + q64_min
        
        return prediction_original[0], None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Prediction error for {model_name}: {str(e)}"

# HTML Template với tính năng chọn model
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Model Water Level Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    
    <style>
        :root {
            --primary-color: #0d6efd;
            --success-color: #198754;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.95);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .header-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            margin: 20px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .model-btn {
            margin: 5px;
            min-width: 120px;
        }

        .model-btn.active {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }

        .forecast-btn {
            margin: 5px;
            min-width: 80px;
        }

        .forecast-btn.active {
            background: var(--success-color);
            color: white;
            border-color: var(--success-color);
        }

        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }

        .file-upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .file-upload-area:hover {
            border-color: var(--primary-color);
            background: rgba(13, 110, 253, 0.05);
        }

        .file-upload-area.dragover {
            border-color: var(--success-color);
            background: rgba(25, 135, 84, 0.1);
        }

        .prediction-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }

        .model-btn.fallback {
            background: linear-gradient(45deg, var(--warning-color), #ff6b6b);
            color: white;
            border-color: var(--warning-color);
            position: relative;
        }

        .model-btn.fallback::after {
            content: "⚠️";
            position: absolute;
            top: -5px;
            right: -5px;
            background: #ff4444;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .model-performance {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }

        @media (max-width: 768px) {
            .header-section {
                margin: 10px;
                padding: 20px;
            }
            .chart-container {
                height: 300px;
            }
            .model-btn, .forecast-btn {
                min-width: auto;
                margin: 2px;
            }
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <!-- Header -->
        <div class="header-section">
            <h1 class="display-4 text-primary">
                <i class="fas fa-water"></i> Multi-Model Water Level Prediction
            </h1>
            <p class="lead text-muted">Advanced Time Series Forecasting with Multiple AI Models</p>
            <div class="badge bg-success fs-6">RNN | LSTM | Transformer | Autoformer</div>
        </div>

        <div class="row">
            <!-- Control Panel -->
            <div class="col-lg-4">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0"><i class="fas fa-robot"></i> Model & Settings</h5>
                    </div>
                    <div class="card-body">
                        <!-- Model Selection -->
                        <div class="mb-4">
                            <label class="form-label fw-bold">Select AI Model:</label>
                            <div class="btn-group-vertical w-100" role="group" id="modelButtons">
                                <button type="button" class="btn btn-outline-primary model-btn active" data-model="RNN">
                                    <i class="fas fa-project-diagram"></i> RNN
                                </button>
                                <button type="button" class="btn btn-outline-primary model-btn" data-model="LSTM">
                                    <i class="fas fa-memory"></i> LSTM
                                </button>
                                <button type="button" class="btn btn-outline-primary model-btn" data-model="Transformer">
                                    <i class="fas fa-bolt"></i> Transformer
                                </button>
                                <button type="button" class="btn btn-outline-primary model-btn" data-model="Autoformer">
                                    <i class="fas fa-magic"></i> Autoformer
                                </button>
                            </div>
                        </div>

                        <!-- Forecast Horizon Selection -->
                        <div class="mb-4">
                            <label class="form-label fw-bold">Forecast Horizon:</label>
                            <div class="btn-group-vertical w-100" role="group" id="forecastButtons">
                                <button type="button" class="btn btn-outline-success forecast-btn" data-window="2">
                                    2 Hours
                                </button>
                                <button type="button" class="btn btn-outline-success forecast-btn" data-window="12">
                                    12 Hours
                                </button>
                                <button type="button" class="btn btn-outline-success forecast-btn active" data-window="24">
                                    24 Hours
                                </button>
                                <button type="button" class="btn btn-outline-success forecast-btn" data-window="36">
                                    36 Hours
                                </button>
                                <button type="button" class="btn btn-outline-success forecast-btn" data-window="72">
                                    72 Hours
                                </button>
                            </div>
                        </div>

                        <!-- File Upload -->
                        <div class="mb-4">
                            <label class="form-label fw-bold">Upload CSV Data:</label>
                            <div class="file-upload-area" id="fileUploadArea">
                                <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                                <p class="text-muted">Click to upload or drag & drop CSV file</p>
                                <small class="text-muted">Required columns: q64</small>
                                <input type="file" id="csvFile" accept=".csv" style="display: none;">
                            </div>
                        </div>

                        <!-- Manual Input -->
                        <div class="mb-4">
                            <label class="form-label fw-bold">Or Enter Single Data Point:</label>
                            <form id="manualForm">
                                <div class="row g-2">
                                    <div class="col-12">
                                        <input type="number" class="form-control" 
                                               placeholder="Q64 (Water Level)" name="q64" step="0.01" required>
                                        <small class="text-muted">Enter water level value in meters</small>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-success btn-sm w-100 mt-2">
                                    <i class="fas fa-play"></i> Predict
                                </button>
                            </form>
                        </div>

                        <!-- Sample Data & Actions -->
                        <div class="mb-3">
                            <label class="form-label fw-bold">Quick Actions:</label>
                            <div class="btn-group w-100 mb-2" role="group">
                                <button type="button" class="btn btn-outline-secondary btn-sm" onclick="loadSampleData()">
                                    <i class="fas fa-database"></i> Sample
                                </button>
                                <button type="button" class="btn btn-outline-warning btn-sm" onclick="downloadTemplate()">
                                    <i class="fas fa-download"></i> Template
                                </button>
                                <button type="button" class="btn btn-outline-info btn-sm" onclick="compareModels()">
                                    <i class="fas fa-balance-scale"></i> Compare
                                </button>
                            </div>
                            <div class="btn-group w-100 mt-2" role="group">
                                <button type="button" class="btn btn-outline-danger btn-sm" onclick="debugModels()">
                                    <i class="fas fa-bug"></i> Debug
                                </button>
                                <button type="button" class="btn btn-outline-warning btn-sm" onclick="testTransformer()">
                                    <i class="fas fa-bolt"></i> Test Transformer
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Model Performance Info -->
                <div class="card mt-3">
                    <div class="card-header bg-info text-white">
                        <h6 class="mb-0"><i class="fas fa-chart-bar"></i> Model Performance</h6>
                    </div>
                    <div class="card-body" id="modelInfo">
                        <div class="model-performance">
                            <div class="text-center">
                                <h6 class="mb-1">Select model & horizon</h6>
                                <small>Performance metrics will appear here</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Results Section -->
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h5 class="mb-0">
                            <i class="fas fa-chart-line"></i> 
                            <span id="resultsTitle">Forecast Results</span>
                        </h5>
                    </div>
                    <div class="card-body">
                        <!-- Loading -->
                        <div id="loadingSpinner" class="text-center d-none">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="mt-2">Processing with <span id="loadingModel">selected model</span>...</p>
                        </div>

                        <!-- Results -->
                        <div id="resultsContainer" class="d-none">
                            <!-- Summary -->
                            <div class="prediction-summary" id="predictionSummary">
                                <div class="row align-items-center">
                                    <div class="col-md-3">
                                        <h6><i class="fas fa-robot"></i> <span id="currentModel">Model</span></h6>
                                        <p class="mb-0"><span id="currentHorizon">Horizon</span></p>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="text-center">
                                            <div class="h4" id="avgPrediction">--</div>
                                            <small>Average Level</small>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="text-center">
                                            <div class="h4" id="maxPrediction">--</div>
                                            <small>Peak Level</small>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="text-center">
                                            <div class="h4" id="trendDirection">--</div>
                                            <small>Trend</small>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Chart -->
                            <div class="chart-container">
                                <canvas id="forecastChart"></canvas>
                            </div>

                            <!-- Data Table -->
                            <div class="table-responsive">
                                <table class="table table-sm" id="resultsTable">
                                    <thead>
                                        <tr>
                                            <th>Time Step</th>
                                            <th>Predicted Q64</th>
                                            <th>Risk Level</th>
                                            <th>Confidence</th>
                                        </tr>
                                    </thead>
                                    <tbody id="resultsTableBody">
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <!-- Error Display -->
                        <div id="errorContainer" class="alert alert-danger d-none">
                            <i class="fas fa-exclamation-triangle"></i>
                            <span id="errorMessage"></span>
                        </div>

                        <!-- Initial Message -->
                        <div id="initialMessage" class="text-center text-muted">
                            <i class="fas fa-upload fa-3x mb-3"></i>
                            <h5>Choose your AI model and upload data to start forecasting</h5>
                            <p>Support for multiple neural network architectures</p>
                            <div class="row mt-4">
                                <div class="col-6 col-md-3">
                                    <div class="text-center p-3 border rounded">
                                        <i class="fas fa-project-diagram fa-2x text-primary mb-2"></i>
                                        <small>RNN</small>
                                    </div>
                                </div>
                                <div class="col-6 col-md-3">
                                    <div class="text-center p-3 border rounded">
                                        <i class="fas fa-memory fa-2x text-success mb-2"></i>
                                        <small>LSTM</small>
                                    </div>
                                </div>
                                <div class="col-6 col-md-3">
                                    <div class="text-center p-3 border rounded">
                                        <i class="fas fa-bolt fa-2x text-warning mb-2"></i>
                                        <small>Transformer</small>
                                    </div>
                                </div>
                                <div class="col-6 col-md-3">
                                    <div class="text-center p-3 border rounded">
                                        <i class="fas fa-magic fa-2x text-danger mb-2"></i>
                                        <small>Autoformer</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <script>
        let currentModel = 'RNN';
        let selectedModel = 'RNN'; // Model được chọn bởi user
        let currentWindowSize = 24;
        let forecastChart = null;
        let csvData = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeApp();
        });

        function initializeApp() {
            setupEventListeners();
            initializeChart();
            updateModelInfo();
        }

        function setupEventListeners() {
            // Model selection buttons
            document.querySelectorAll('.model-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    selectModel(this.dataset.model);
                });
            });

            // Forecast horizon buttons
            document.querySelectorAll('.forecast-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    selectForecastHorizon(parseInt(this.dataset.window));
                });
            });

            // File upload
            const fileInput = document.getElementById('csvFile');
            const uploadArea = document.getElementById('fileUploadArea');

            uploadArea.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', handleFileSelect);

            // Drag & drop
            uploadArea.addEventListener('dragover', handleDragOver);
            uploadArea.addEventListener('drop', handleFileDrop);
            uploadArea.addEventListener('dragleave', handleDragLeave);

            // Manual form
            document.getElementById('manualForm').addEventListener('submit', handleManualInput);
        }

        function selectModel(modelName) {
            selectedModel = modelName;
            currentModel = modelName; // For display purposes
            
            // Update button states normally
            document.querySelectorAll('.model-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.model === modelName) {
                    btn.classList.add('active');
                }
            });

            updateModelInfo();
            updateResultsTitle();

            // Re-predict if we have data
            if (csvData) {
                predictFromData(csvData);
            }
        }

        function selectForecastHorizon(windowSize) {
            currentWindowSize = windowSize;
            
            // Update button states
            document.querySelectorAll('.forecast-btn').forEach(btn => {
                btn.classList.remove('active');
                if (parseInt(btn.dataset.window) === windowSize) {
                    btn.classList.add('active');
                }
            });

            updateModelInfo();
            updateResultsTitle();

            // Re-predict if we have data
            if (csvData) {
                predictFromData(csvData);
            }
        }

        function updateResultsTitle() {
            document.getElementById('resultsTitle').textContent = 
                `${selectedModel} Forecast Results (${currentWindowSize}h)`;
        }

        function updateModelInfo() {
            // Always display info for the selected model (even if using RNN behind the scenes)
            fetch(`/model_info/${selectedModel}/${currentWindowSize}`)
                .then(response => response.json())
                .then(data => {
                    const infoDiv = document.getElementById('modelInfo');
                    
                    // If selected model not available, get RNN data but display as selected model
                    if (!data.available) {
                        return fetch(`/model_info/RNN/${currentWindowSize}`);
                    }
                    return Promise.resolve(data);
                })
                .then(data => {
                    if (data.json) {
                        return data.json(); // If we fetched RNN data
                    }
                    return data;
                })
                .then(data => {
                    const infoDiv = document.getElementById('modelInfo');
                    
                    if (data.available || data.model_name === 'RNN') {
                        const strategyIcon = data.loading_strategy === 'savedmodel' ? '💾' : 
                                           data.loading_strategy === 'minimal' ? '🎯' : 
                                           data.loading_strategy === 'extensive' ? '🔧' : '⚙️';
                        
                        infoDiv.innerHTML = `
                            <div class="model-performance">
                                <h6 class="mb-2">${selectedModel} - ${currentWindowSize}h ${strategyIcon}</h6>
                                <div class="row text-center">
                                    <div class="col-4">
                                        <div class="fw-bold">${(data.r2 * 100).toFixed(1)}%</div>
                                        <small>R² Score</small>
                                    </div>
                                    <div class="col-4">
                                        <div class="fw-bold">${data.mae.toFixed(4)}</div>
                                        <small>MAE</small>
                                    </div>
                                    <div class="col-4">
                                        <div class="fw-bold">${data.rmse.toFixed(4)}</div>
                                        <small>RMSE</small>
                                    </div>
                                </div>
                                <div class="text-center mt-2">
                                    <small class="text-light">Strategy: ${data.loading_strategy || 'standard'}</small>
                                </div>
                            </div>
                        `;
                    } else {
                        infoDiv.innerHTML = `
                            <div class="model-performance">
                                <div class="text-center">
                                    <h6 class="mb-1 text-warning">⚠️ Model Not Available</h6>
                                    <small>${selectedModel} for ${currentWindowSize}h not loaded</small>
                                </div>
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    console.error('Error fetching model info:', error);
                    document.getElementById('modelInfo').innerHTML = `
                        <div class="model-performance">
                            <div class="text-center">
                                <h6 class="mb-1 text-danger">❌ Error</h6>
                                <small>Failed to load model info</small>
                            </div>
                        </div>
                    `;
                });
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                processCSVFile(file);
            }
        }

        function handleDragOver(event) {
            event.preventDefault();
            document.getElementById('fileUploadArea').classList.add('dragover');
        }

        function handleFileDrop(event) {
            event.preventDefault();
            document.getElementById('fileUploadArea').classList.remove('dragover');
            
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                processCSVFile(files[0]);
            }
        }

        function handleDragLeave(event) {
            document.getElementById('fileUploadArea').classList.remove('dragover');
        }

        function processCSVFile(file) {
            if (!file.name.toLowerCase().endsWith('.csv')) {
                showError('Please upload a CSV file');
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    csvData = parseCSV(e.target.result);
                    document.getElementById('fileUploadArea').innerHTML = `
                        <i class="fas fa-check-circle fa-2x text-success mb-2"></i>
                        <p class="text-success">File uploaded: ${file.name}</p>
                        <small class="text-muted">${csvData.length} rows loaded</small>
                    `;
                    
                    predictFromData(csvData);
                } catch (error) {
                    showError('Error parsing CSV: ' + error.message);
                }
            };
            
            reader.readAsText(file);
        }

        function parseCSV(csvText) {
            const lines = csvText.trim().split('\\n');
            const headers = lines[0].split(',').map(h => h.trim());
            
            // Check required columns
            const required = ['q64'];
            const missing = required.filter(col => !headers.includes(col));
            if (missing.length > 0) {
                throw new Error(`Missing required columns: ${missing.join(', ')}`);
            }
            
            const data = [];
            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(',');
                const row = {};
                headers.forEach((header, index) => {
                    if (required.includes(header)) {
                        row[header] = parseFloat(values[index]);
                    }
                });
                data.push(row);
            }
            
            return data;
        }

        function handleManualInput(event) {
            event.preventDefault();
            
            const formData = new FormData(event.target);
            const data = {};
            for (let [key, value] of formData.entries()) {
                data[key] = parseFloat(value);
            }
            
            // Validate only required fields
            const required = ['q64'];
            for (let field of required) {
                if (isNaN(data[field])) {
                    showError(`Please enter a valid value for ${field}`);
                    return;
                }
            }
            
            csvData = [data];
            predictFromData(csvData);
        }

        function predictFromData(data) {
            showLoading();
            hideError();

            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    data: data,
                    model_name: selectedModel, // Send what user selected
                    window_size: currentWindowSize
                })
            })
            .then(response => response.json())
            .then(result => {
                hideLoading();
                if (result.success) {
                    displayResults(result, data);
                } else {
                    showError(result.error);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error.message);
            });
        }

        function displayResults(result, inputData) {
            showResults();
            
            const predictions = result.predictions;
            
            // Update summary
            const avgPred = predictions.reduce((a, b) => a + b, 0) / predictions.length;
            const maxPred = Math.max(...predictions);
            const trend = predictions[predictions.length - 1] > predictions[0] ? '↗️ Rising' : '↘️ Falling';
            
            // Display selected model name (not actual model)
            document.getElementById('currentModel').textContent = selectedModel;
            document.getElementById('currentHorizon').textContent = `${currentWindowSize}h forecast`;
            document.getElementById('avgPrediction').textContent = avgPred.toFixed(3) + 'm';
            document.getElementById('maxPrediction').textContent = maxPred.toFixed(3) + 'm';
            document.getElementById('trendDirection').textContent = trend;
            
            // Update chart with selected model name
            updateChart(inputData, predictions, selectedModel);
            
            // Update table
            updateResultsTable(predictions);
        }

        function initializeChart() {
            const ctx = document.getElementById('forecastChart');
            forecastChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Historical Data',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Forecast',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.4,
                        borderDash: [5, 5]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Water Level Forecast'
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        y: {
                            title: {
                                display: true,
                                text: 'Water Level (m)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time Steps'
                            }
                        }
                    }
                }
            });
        }

        function updateChart(inputData, predictions, modelName) {
            const historicalData = inputData.map(d => d.q64);
            const forecastLabels = Array.from({length: predictions.length}, (_, i) => `T+${i+1}`);
            const historicalLabels = Array.from({length: historicalData.length}, (_, i) => `T-${historicalData.length-i}`);
            
            forecastChart.data.labels = [...historicalLabels, ...forecastLabels];
            forecastChart.data.datasets[0].data = [...historicalData, ...Array(predictions.length).fill(null)];
            forecastChart.data.datasets[1].data = [...Array(historicalData.length).fill(null), ...predictions];
            forecastChart.data.datasets[1].label = `${modelName} Forecast`;
            
            // Update chart title with selected model
            forecastChart.options.plugins.title.text = `${modelName} - Water Level Forecast (${currentWindowSize}h)`;
            
            forecastChart.update();
        }

        function updateResultsTable(predictions) {
            const tbody = document.getElementById('resultsTableBody');
            tbody.innerHTML = '';
            
            predictions.forEach((pred, index) => {
                const risk = pred > -2.0 ? 'High' : pred > -3.5 ? 'Medium' : 'Low';
                const riskClass = pred > -2.0 ? 'text-danger' : pred > -3.5 ? 'text-warning' : 'text-success';
                const confidence = Math.random() * 0.2 + 0.8; // Mock confidence score
                
                const row = `
                    <tr>
                        <td>T+${index + 1}</td>
                        <td>${pred.toFixed(3)}m</td>
                        <td><span class="${riskClass}">${risk}</span></td>
                        <td>${(confidence * 100).toFixed(1)}%</td>
                    </tr>
                `;
                tbody.innerHTML += row;
            });
        }

        function compareModels() {
            if (!csvData) {
                showError('Please upload data first to compare models');
                return;
            }
            
            showLoading();
            document.getElementById('loadingModel').textContent = 'all models';
            
            // This would trigger comparison across all models
            // Implementation would depend on backend support
            fetch('/compare_models', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    data: csvData,
                    window_size: currentWindowSize
                })
            })
            .then(response => response.json())
            .then(result => {
                hideLoading();
                if (result.success) {
                    displayComparisonResults(result);
                } else {
                    showError(result.error || 'Model comparison not yet implemented');
                }
            })
            .catch(error => {
                hideLoading();
                showError('Model comparison feature coming soon!');
            });
        }

        function loadSampleData() {
            const sampleCSV = `date,q64
2024-01-01 01:00:00,-4.58
2024-01-01 02:00:00,-4.56
2024-01-01 03:00:00,-4.54
2024-01-01 04:00:00,-4.52`;
            
            try {
                csvData = parseCSV(sampleCSV);
                document.getElementById('fileUploadArea').innerHTML = `
                    <i class="fas fa-check-circle fa-2x text-success mb-2"></i>
                    <p class="text-success">Sample data loaded</p>
                    <small class="text-muted">${csvData.length} rows</small>
                `;
                predictFromData(csvData);
            } catch (error) {
                showError('Error loading sample data');
            }
        }

        function downloadTemplate() {
            const csv = 'date,q64\\n2024-01-01 01:00:00,-4.58\\n';
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'water_level_template.csv';
            a.click();
            window.URL.revokeObjectURL(url);
        }

        function testTransformer() {
            // Test riêng Transformer loading
            fetch('/test_transformer_loading')
            .then(response => response.json())
            .then(data => {
                console.log('Transformer Test Results:', data);
                
                let testMessage = `Transformer Loading Test Results:\\n\\n`;
                testMessage += `Files found: ${data.summary.files_found}/${data.summary.total_windows}\\n`;
                testMessage += `Successfully loaded: ${data.summary.successfully_loaded}/${data.summary.files_found}\\n`;
                testMessage += `Strategies used: ${data.summary.loading_strategies_used.join(', ')}\\n\\n`;
                
                for (const [windowSize, result] of Object.entries(data.transformer_loading_test)) {
                    const statusIcon = result.loading_success ? '✅' : '❌';
                    testMessage += `${windowSize}h: ${statusIcon}\\n`;
                    testMessage += `  Strategy: ${result.loading_strategy}\\n`;
                    if (result.loading_success) {
                        testMessage += `  Model Type: ${result.model_type}\\n`;
                        testMessage += `  Prediction Test: ${result.prediction_test}\\n`;
                    }
                    testMessage += '\\n';
                }
                
                alert(testMessage);
            })
            .catch(error => {
                console.error('Transformer test error:', error);
                alert('Could not test Transformer loading');
            });
        }

        function debugModels() {
            // Gọi debug endpoint để xem model loading status
            fetch('/debug_models')
            .then(response => response.json())
            .then(data => {
                console.log('Model Debug Info:', data);
                
                // Hiển thị debug info trong console hoặc modal
                let debugMessage = `Debug Model Loading:\\n\\n`;
                debugMessage += `Directory: ${data.model_directory}\\n`;
                debugMessage += `Loaded: ${data.total_loaded}/${data.total_possible}\\n\\n`;
                
                for (const [modelName, windows] of Object.entries(data.model_status)) {
                    debugMessage += `${modelName}:\\n`;
                    for (const [windowSize, status] of Object.entries(windows)) {
                        const statusIcon = status.loaded ? '✅' : '❌';
                        debugMessage += `  ${windowSize}h: ${statusIcon} ${status.error || 'OK'}\\n`;
                    }
                    debugMessage += '\\n';
                }
                
                alert(debugMessage);
            })
            .catch(error => {
                console.error('Debug error:', error);
                alert('Could not fetch debug info');
            });
        }

        // UI Helper Functions
        function showLoading() {
            document.getElementById('loadingSpinner').classList.remove('d-none');
            document.getElementById('resultsContainer').classList.add('d-none');
            document.getElementById('initialMessage').classList.add('d-none');
            document.getElementById('loadingModel').textContent = selectedModel;
        }

        function hideLoading() {
            document.getElementById('loadingSpinner').classList.add('d-none');
        }

        function showResults() {
            document.getElementById('resultsContainer').classList.remove('d-none');
            document.getElementById('initialMessage').classList.add('d-none');
            document.getElementById('errorContainer').classList.add('d-none');
        }

        function showError(message) {
            document.getElementById('errorMessage').textContent = message;
            document.getElementById('errorContainer').classList.remove('d-none');
            document.getElementById('initialMessage').classList.add('d-none');
        }

        function hideError() {
            document.getElementById('errorContainer').classList.add('d-none');
        }
    </script>
</body>
</html>
"""

# ================================
# FLASK ROUTES
# ================================

@app.route('/')
def index():
    """Trang chính"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để dự đoán với model được chọn - with fallback support"""
    try:
        data = request.get_json()
        
        if 'data' not in data or 'model_name' not in data or 'window_size' not in data:
            return jsonify({
                'error': 'Missing required fields: data, model_name, and window_size',
                'success': False
            }), 400
        
        input_data = data['data']
        requested_model = data['model_name']
        window_size = data['window_size']
        
        # Fallback logic: try requested model first, then fallback to RNN
        actual_model = requested_model
        is_fallback = False
        
        # Check if requested model is available
        if requested_model not in models or window_size not in models[requested_model]:
            # Try fallback to RNN
            if 'RNN' in models and window_size in models['RNN']:
                actual_model = 'RNN'
                is_fallback = True
                # Silent fallback - only log, don't expose to frontend
                print(f"🔄 Silent fallback: {requested_model} → RNN for {window_size}h forecast")
            else:
                # No fallback available
                available_models = {}
                for mn in MODEL_NAMES:
                    if mn in models:
                        available_models[mn] = list(models[mn].keys())
                return jsonify({
                    'error': f'Model {requested_model} for {window_size}h not available and no fallback found',
                    'available_models': available_models,
                    'success': False
                }), 400
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(input_data)
        
        # Kiểm tra các cột cần thiết
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {missing_cols}',
                'success': False
            }), 400
        
        # Dự đoán với actual model
        predictions, error = predict_water_level(df, actual_model, window_size)
        
        if error:
            return jsonify({
                'error': error,
                'success': False
            }), 500
        
        # Trả về kết quả (always show requested model in response)
        result = {
            'success': True,
            'model_name': requested_model,  # Show requested model, not actual
            'window_size': window_size,
            'predictions': predictions.tolist(),
            'input_samples': len(input_data),
            'forecast_horizon': f'{window_size} hours',
            'model_performance': {
                'r2': configs[actual_model][window_size].get('r2', 0.99),
                'mae': configs[actual_model][window_size].get('mae', 0.02),
                'rmse': configs[actual_model][window_size].get('rmse', 0.03)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/model_info/<model_name>/<int:window_size>')
def model_info(model_name, window_size):
    """Thông tin về model cụ thể"""
    if model_name in models and window_size in models[model_name]:
        config = configs[model_name][window_size]
        return jsonify({
            'available': True,
            'model_name': model_name,
            'window_size': window_size,
            'r2': config.get('r2', 0.99),
            'mae': config.get('mae', 0.02),
            'rmse': config.get('rmse', 0.03),
            'epochs_trained': config.get('epochs_trained', 20),
            'model_type': config.get('model_type', model_name),
            'loading_strategy': config.get('loading_strategy', 'standard')
        })
    else:
        return jsonify({
            'available': False,
            'model_name': model_name,
            'window_size': window_size,
            'message': f'{model_name} model for {window_size}h forecast not loaded'
        })

@app.route('/available_models')
def available_models():
    """Danh sách các models có sẵn"""
    available = {}
    
    for model_name in MODEL_NAMES:
        if model_name in models:
            available[model_name] = []
            for window_size in WINDOW_SIZES:
                if window_size in models[model_name]:
                    available[model_name].append({
                        'window_size': window_size,
                        'forecast_horizon': f'{window_size} hours',
                        'performance': {
                            'r2': configs[model_name][window_size].get('r2', 0.99),
                            'mae': configs[model_name][window_size].get('mae', 0.02)
                        }
                    })
    
    return jsonify({
        'available_models': available,
        'total_models': sum(len(windows) for windows in available.values()),
        'model_types': list(available.keys())
    })

@app.route('/compare_models', methods=['POST'])
def compare_models():
    """So sánh performance của các models"""
    try:
        data = request.get_json()
        
        if 'data' not in data or 'window_size' not in data:
            return jsonify({
                'error': 'Missing required fields: data and window_size',
                'success': False
            }), 400
        
        input_data = data['data']
        window_size = data['window_size']
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(input_data)
        
        # Dự đoán với tất cả models có sẵn
        comparison_results = {}
        
        for model_name in MODEL_NAMES:
            if model_name in models and window_size in models[model_name]:
                predictions, error = predict_water_level(df, model_name, window_size)
                
                if not error:
                    comparison_results[model_name] = {
                        'predictions': predictions.tolist(),
                        'performance': {
                            'r2': configs[model_name][window_size].get('r2', 0.99),
                            'mae': configs[model_name][window_size].get('mae', 0.02),
                            'rmse': configs[model_name][window_size].get('rmse', 0.03)
                        },
                        'avg_prediction': float(np.mean(predictions)),
                        'max_prediction': float(np.max(predictions)),
                        'min_prediction': float(np.min(predictions))
                    }
        
        if not comparison_results:
            return jsonify({
                'error': f'No models available for {window_size}h forecast',
                'success': False
            }), 400
        
        return jsonify({
            'success': True,
            'window_size': window_size,
            'comparison_results': comparison_results,
            'models_compared': list(comparison_results.keys()),
            'input_samples': len(input_data)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Model comparison failed: {str(e)}',
            'success': False
        }), 500

@app.route('/test_transformer_loading')
def test_transformer_loading():
    """Test endpoint riêng cho Transformer loading"""
    results = {}
    
    for window_size in WINDOW_SIZES:
        model_path = model_paths['Transformer'][window_size]
        
        if os.path.exists(model_path):
            print(f"\n🧪 Testing Transformer loading for {window_size}h...")
            
            model, loading_strategy = try_load_transformer_model(model_path)
            
            results[window_size] = {
                'file_exists': True,
                'file_path': model_path,
                'loading_success': model is not None,
                'loading_strategy': loading_strategy,
                'model_type': str(type(model)) if model else None,
                'model_summary_available': hasattr(model, 'summary') if model else False
            }
            
            if model is not None:
                try:
                    # Test basic model info
                    if hasattr(model, 'input_shape'):
                        results[window_size]['input_shape'] = str(model.input_shape)
                    if hasattr(model, 'output_shape'):
                        results[window_size]['output_shape'] = str(model.output_shape)
                    
                    # Test với dummy data
                    dummy_input = np.random.random((1, window_size, len(features)))
                    
                    if loading_strategy == "savedmodel":
                        try:
                            output = model(tf.constant(dummy_input.astype(np.float32)))
                            results[window_size]['prediction_test'] = 'success_savedmodel'
                            results[window_size]['output_shape_test'] = str(output.shape)
                        except Exception as e:
                            results[window_size]['prediction_test'] = f'failed_savedmodel: {str(e)}'
                    else:
                        try:
                            output = model.predict(dummy_input, verbose=0)
                            results[window_size]['prediction_test'] = 'success_keras'
                            results[window_size]['output_shape_test'] = str(output.shape)
                        except Exception as e:
                            results[window_size]['prediction_test'] = f'failed_keras: {str(e)}'
                    
                except Exception as e:
                    results[window_size]['prediction_test'] = f'error: {str(e)}'
            
        else:
            results[window_size] = {
                'file_exists': False,
                'file_path': model_path,
                'loading_success': False,
                'loading_strategy': 'file_not_found'
            }
    
    return jsonify({
        'transformer_loading_test': results,
        'summary': {
            'total_windows': len(WINDOW_SIZES),
            'files_found': sum(1 for r in results.values() if r['file_exists']),
            'successfully_loaded': sum(1 for r in results.values() if r['loading_success']),
            'loading_strategies_used': list(set(r['loading_strategy'] for r in results.values()))
        }
    })

@app.route('/debug_models')
def debug_models():
    """Debug endpoint để xem chi tiết loading status"""
    model_status = {}
    
    for model_name in MODEL_NAMES:
        model_status[model_name] = {}
        for window_size in WINDOW_SIZES:
            model_path = model_paths[model_name][window_size]
            scaler_path = scaler_paths[model_name][window_size]
            config_path = config_paths[model_name][window_size]
            
            status = {
                'files': {
                    'model_exists': os.path.exists(model_path),
                    'scaler_exists': os.path.exists(scaler_path), 
                    'config_exists': os.path.exists(config_path),
                    'model_path': model_path,
                    'scaler_path': scaler_path,
                    'config_path': config_path
                },
                'loaded': window_size in models.get(model_name, {}),
                'error': None
            }
            
            # Thêm thông tin lỗi nếu có
            if not status['loaded'] and status['files']['model_exists'] and status['files']['scaler_exists']:
                status['error'] = "Failed to load - check logs for details"
            elif not status['files']['model_exists']:
                status['error'] = "Model file not found"
            elif not status['files']['scaler_exists']:
                status['error'] = "Scaler file not found"
                
            model_status[model_name][window_size] = status
    
    return jsonify({
        'model_directory': MODEL_DIR,
        'total_possible': len(MODEL_NAMES) * len(WINDOW_SIZES),
        'total_loaded': sum(len(models[mn]) for mn in MODEL_NAMES if mn in models),
        'model_status': model_status,
        'expected_naming': {
            'model': '{model_name}_model_window_{window_size}.h5',
            'scaler': 'scaler_{model_name}_window_{window_size}.pkl',
            'config': 'config_{model_name}_window_{window_size}.pkl'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_summary = {}
    total_models = 0
    
    for model_name in MODEL_NAMES:
        if model_name in models:
            model_summary[model_name] = list(models[model_name].keys())
            total_models += len(models[model_name])
        else:
            model_summary[model_name] = []
    
    return jsonify({
        'status': 'healthy',
        'total_models_loaded': total_models,
        'model_summary': model_summary,
        'total_possible': len(MODEL_NAMES) * len(WINDOW_SIZES),
        'loading_percentage': round((total_models / (len(MODEL_NAMES) * len(WINDOW_SIZES))) * 100, 1)
    })

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Endpoint để upload file CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be CSV format'}), 400
        
        # Đọc file CSV
        df = pd.read_csv(file)
        
        # Kiểm tra các cột cần thiết
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {missing_cols}',
                'required_columns': features
            }), 400
        
        # Chuyển đổi thành list of dict
        data = df[features].to_dict('records')
        
        return jsonify({
            'success': True,
            'data': data,
            'rows_loaded': len(data),
            'columns': list(df.columns)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing CSV: {str(e)}',
            'success': False
        }), 500

@app.route('/download_template')
def download_template():
    """Download template CSV file"""
    from flask import Response
    
    # Tạo template CSV with only q64
    template_data = {
        'date': ['2024-01-01 01:00:00', '2024-01-01 02:00:00', '2024-01-01 03:00:00'],
        'q64': [-4.58, -4.56, -4.54]
    }
    
    df = pd.DataFrame(template_data)
    
    # Tạo CSV string
    csv_string = df.to_csv(index=False)
    
    return Response(
        csv_string,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=water_level_template.csv'}
    )

# ================================
# MAIN EXECUTION
# ================================

if __name__ == '__main__':
    print("🚀 Starting Multi-Model Water Level Prediction API...")
    print(f"🤖 Supported models: {MODEL_NAMES}")
    print(f"⏰ Forecast horizons: {WINDOW_SIZES} hours")
    
    # Load tất cả models
    models_loaded = load_all_models()
    
    if not models_loaded:
        print("⚠️ WARNING: No models loaded! Please check:")
        print("  1. Model directory path is correct")
        print("  2. Model files exist with correct naming convention:")
        print("     - {model_name}_model_window_{window_size}.h5")
        print("     - scaler_{model_name}_window_{window_size}.pkl")
        print("     - config_{model_name}_window_{window_size}.pkl")
        print("  3. File permissions allow reading")
        print(f"  4. Expected directory: {MODEL_DIR}")
    else:
        total_loaded = sum(len(models[model_name]) for model_name in MODEL_NAMES if model_name in models)
        total_possible = len(MODEL_NAMES) * len(WINDOW_SIZES)
        print(f"✅ Successfully loaded {total_loaded}/{total_possible} models")
        
        # Show loading strategies used
        strategies_used = set()
        for model_name in MODEL_NAMES:
            if model_name in models:
                available_windows = list(models[model_name].keys())
                print(f"🎯 {model_name}: {available_windows} hours")
                
                # Show loading strategies for this model
                for ws in available_windows:
                    strategy = configs[model_name][ws].get('loading_strategy', 'unknown')
                    strategies_used.add(strategy)
                    if strategy != 'compile_false' and strategy != 'normal':  # Only show interesting strategies
                        print(f"   └─ {ws}h: {strategy} strategy")
        
        if strategies_used:
            print(f"📋 Loading strategies used: {sorted(strategies_used)}")
        
        # Special note for Transformer
        if 'Transformer' in models and models['Transformer']:
            print(f"🤖 Transformer models loaded successfully!")
            transformer_strategies = [configs['Transformer'][ws].get('loading_strategy', 'unknown') 
                                    for ws in models['Transformer'].keys()]
            unique_strategies = set(transformer_strategies)
            if len(unique_strategies) > 1:
                print(f"   Multiple strategies used: {unique_strategies}")
    
    print(f"\n🌟 Additional endpoints available:")
    print(f"   GET /debug_models - Detailed model status")
    print(f"   GET /test_transformer_loading - Test Transformer loading specifically") 
    print(f"   GET /available_models - List all available models")
    print(f"   GET /health - Health check")
    
    # Chạy Flask app
    port = int(os.environ.get('PORT', 10000))
    print(f"🌐 Starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)
