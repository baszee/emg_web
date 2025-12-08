# ==================================
# 📚 1. IMPORTS & KONFIGURASI
# ==================================
import os # <-- TAMBAHKAN INI
import joblib
import numpy as np
import random
import time
import io
import csv
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

# --- KONSTANTA & VARIABEL GLOBAL ---
# Dapatkan directory tempat file app.py ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Gabungkan path: BASE_DIR + nama file
MODEL_PATH = os.path.join(BASE_DIR, 'model_svm_emg.joblib') # <-- UBAH BARIS INI!
GESTURE_MAP = {0: 'Rock ✊', 1: 'Scissors ✌️', 2: 'Paper ✋', 3: 'OK 👌'}

# --- PLACEHOLDERS ---
model = None
mean_scaler = np.zeros(64)
std_scaler = np.ones(64)
akurasi_test = 0.0
parameter_terbaik = {'kernel': 'Simulasi', 'C': 'N/A', 'gamma': 'N/A'}
sample_data = None
MODE_SIMULASI = True

# ==================================
# ⚙️ 2. INISIALISASI APLIKASI
# ==================================
app = Flask(__name__)
app.secret_key = 'emg-classifier-secret-key-change-in-production'

# ==================================
# 💾 3. LOAD MODEL
# ==================================
try:
    data_model = joblib.load(MODEL_PATH)
    model = data_model['model']
    mean_scaler = data_model['mean_scaler']
    std_scaler = data_model['std_scaler']
    
    if isinstance(data_model['nama_gesture'], dict):
        GESTURE_MAP = data_model['nama_gesture']
    else:
        GESTURE_MAP = dict(enumerate(data_model['nama_gesture']))
    
    akurasi_test = data_model.get('akurasi_test', 0.0)
    parameter_terbaik = data_model.get('parameter_terbaik', parameter_terbaik)
    sample_data = data_model.get('sample_data', None)
    
    MODE_SIMULASI = False
    print("✓ Model loaded successfully (Production Mode)")
    
except Exception as e:
    print(f"⚠ Model load failed: {e}. Running in SIMULATION mode.")

# ==================================
# 🎲 4. HELPER: GENERATE SAMPLE
# ==================================
def generate_balanced_random_sample(gesture_class=None):
    """Generate realistic EMG sample"""
    
    if sample_data is not None and not MODE_SIMULASI:
        if gesture_class is None:
            gesture_class = random.choice(list(sample_data.keys()))
        
        if gesture_class in sample_data and len(sample_data[gesture_class]) > 0:
            samples = sample_data[gesture_class]
            random_idx = random.randint(0, len(samples) - 1)
            return samples[random_idx], gesture_class
    
    # Fallback: Synthetic data
    if gesture_class is None:
        gesture_class = random.randint(0, 3)
    
    if gesture_class == 0:  # Rock
        base_values = np.random.uniform(20, 35, 64)
        noise = np.random.normal(0, 5, 64)
    elif gesture_class == 1:  # Scissors
        base_values = np.random.uniform(5, 15, 64)
        for i in range(8):
            base_values[i*8 + 2] += random.uniform(15, 25)
            base_values[i*8 + 3] += random.uniform(15, 25)
        noise = np.random.normal(0, 3, 64)
    elif gesture_class == 2:  # Paper
        base_values = np.random.uniform(8, 18, 64)
        noise = np.random.normal(0, 4, 64)
    else:  # OK
        base_values = np.random.uniform(10, 20, 64)
        for i in range(8):
            base_values[i*8 + 1] += random.uniform(10, 20)
            base_values[i*8 + 2] += random.uniform(10, 20)
        noise = np.random.normal(0, 4, 64)
    
    emg_data = base_values + noise
    return emg_data, gesture_class

# ==================================
# 🌐 5. ROUTES - BASIC PAGES
# ==================================

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def index():
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    info = {
        'akurasi': akurasi_test,
        'parameter': parameter_terbaik,
        'kernel': parameter_terbaik.get('kernel', 'N/A'),
        'C': parameter_terbaik.get('C', 'N/A'),
        'gamma': parameter_terbaik.get('gamma', 'N/A'),
        'total_samples': 11678,
        'train_samples': 9341,
        'test_samples': 2337,
        'split_ratio': '80:20',
        'training_time': '219.07',
        'cv_accuracy': 0.9148,
        'per_class_accuracy': {
            'Rock': 0.945,
            'Scissors': 0.976,
            'Paper': 0.902,
            'OK': 0.884
        },
        'simulasi_mode': MODE_SIMULASI
    }
    return render_template('model_info.html', info=info)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    is_forced_simulasi = request.args.get('force_sim', 'false') == 'true'
    expected_gesture = request.args.get('expected_gesture', None)
    
    if request.method == 'POST':
        emg_data_str = request.form.get('emg_data')
        
        try:
            data_list = [float(x.strip()) for x in emg_data_str.split(',')]
        except ValueError:
            return render_template('predict.html', 
                                 error="Input harus 64 nilai numerik dipisahkan koma")
        
        if len(data_list) != 64:
            return render_template('predict.html',
                                 error=f"Input harus tepat 64 nilai (Anda input {len(data_list)})")
        
        if MODE_SIMULASI or is_forced_simulasi:
            prediction_class = random.choice(list(GESTURE_MAP.keys()))
            probabilities = np.ones(len(GESTURE_MAP)) / len(GESTURE_MAP)
            result_gesture = f"{GESTURE_MAP.get(prediction_class, 'Unknown')} (SIMULASI)"
        else:
            final_input_raw = np.array(data_list).reshape(1, -1)
            final_input_scaled = (final_input_raw - mean_scaler) / std_scaler
            
            prediction_class = model.predict(final_input_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(final_input_scaled)[0]
            else:
                probabilities = np.zeros(len(GESTURE_MAP))
                probabilities[prediction_class] = 1.0
            
            result_gesture = GESTURE_MAP.get(prediction_class, "Unknown")
        
        return render_template('result.html',
                             gesture=result_gesture,
                             probabilities=probabilities.tolist(),
                             gesture_map=GESTURE_MAP,
                             input_data_64=data_list,
                             expected_gesture=expected_gesture)
    
    emg_data_random = request.args.get('emg_data_random', '')
    return render_template('predict.html', emg_data_default=emg_data_random)

@app.route('/generate-random-sample', methods=['POST'])
def generate_random_sample():
    gesture_class_str = request.form.get('gesture_class', None)
    
    if gesture_class_str is not None:
        try:
            gesture_class = int(gesture_class_str)
        except ValueError:
            gesture_class = None
    else:
        gesture_class = None
    
    emg_data_array, selected_gesture = generate_balanced_random_sample(gesture_class)
    emg_data_str = ', '.join([f"{x:.2f}" for x in emg_data_array.tolist()])
    
    return redirect(url_for('predict',
                           emg_data_random=emg_data_str,
                           force_sim='false',
                           expected_gesture=GESTURE_MAP.get(selected_gesture, 'Unknown')))

@app.route('/generate-specific-sample/<int:gesture_class>', methods=['POST'])
def generate_specific_sample(gesture_class):
    if gesture_class not in GESTURE_MAP:
        return redirect(url_for('predict'))
    
    emg_data_array, _ = generate_balanced_random_sample(gesture_class)
    emg_data_str = ', '.join([f"{x:.2f}" for x in emg_data_array.tolist()])
    
    return redirect(url_for('predict',
                           emg_data_random=emg_data_str,
                           force_sim='false',
                           expected_gesture=GESTURE_MAP.get(gesture_class, 'Unknown')))

# ==================================
# 🆕 6. NEW FEATURE #1: CONFIDENCE TRACKER
# ==================================

@app.route('/confidence-tracker')
def confidence_tracker():
    return render_template('confidence_tracker.html')

@app.route('/api/predict-with-tracking', methods=['POST'])
def predict_with_tracking():
    try:
        data = request.get_json()
        emg_data_str = data.get('emg_data', '')
        
        data_list = [float(x.strip()) for x in emg_data_str.split(',')]
        
        if len(data_list) != 64:
            return jsonify({'error': 'Must have exactly 64 values'}), 400
        
        if MODE_SIMULASI:
            prediction_class = random.choice(list(GESTURE_MAP.keys()))
            probabilities = np.random.dirichlet(np.ones(4))
        else:
            final_input_raw = np.array(data_list).reshape(1, -1)
            final_input_scaled = (final_input_raw - mean_scaler) / std_scaler
            
            prediction_class = model.predict(final_input_scaled)[0]
            probabilities = model.predict_proba(final_input_scaled)[0]
        
        result_gesture = GESTURE_MAP.get(prediction_class, 'Unknown')
        max_confidence = float(max(probabilities))
        
        # Store in session
        if 'prediction_history' not in session:
            session['prediction_history'] = []
        
        session['prediction_history'].append({
            'gesture': result_gesture,
            'confidence': max_confidence,
            'all_probs': probabilities.tolist(),
            'timestamp': time.time()
        })
        
        session['prediction_history'] = session['prediction_history'][-10:]
        session.modified = True
        
        return jsonify({
            'success': True,
            'prediction': result_gesture,
            'confidence': max_confidence,
            'probabilities': probabilities.tolist(),
            'history': session['prediction_history']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    session['prediction_history'] = []
    session.modified = True
    return jsonify({'success': True})

# ==================================
# 🆕 7. NEW FEATURE #2: GESTURE COMPARISON
# ==================================

@app.route('/compare-gestures')
def compare_gestures():
    return render_template('compare_gestures.html', gestures=GESTURE_MAP)

@app.route('/api/compare', methods=['POST'])
def compare_gestures_api():
    try:
        data = request.get_json()
        gesture1_id = int(data.get('gesture1'))
        gesture2_id = int(data.get('gesture2'))
        
        if gesture1_id not in GESTURE_MAP or gesture2_id not in GESTURE_MAP:
            return jsonify({'error': 'Invalid gesture IDs'}), 400
        
        sample1, _ = generate_balanced_random_sample(gesture1_id)
        sample2, _ = generate_balanced_random_sample(gesture2_id)
        
        diff = np.abs(sample1 - sample2)
        similarity_score = float(1 - (np.mean(diff) / np.max([np.max(sample1), np.max(sample2)])))
        
        return jsonify({
            'gesture1': GESTURE_MAP[gesture1_id],
            'gesture2': GESTURE_MAP[gesture2_id],
            'heatmap1': sample1.reshape(8, 8).tolist(),
            'heatmap2': sample2.reshape(8, 8).tolist(),
            'difference': diff.reshape(8, 8).tolist(),
            'similarity_score': similarity_score * 100
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================================
# 🆕 8. NEW FEATURE #3: BATCH PREDICTION
# ==================================

@app.route('/batch-predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return render_template('batch_predict.html', 
                                 error='No file uploaded')
        
        file = request.files['csv_file']
        
        if file.filename == '':
            return render_template('batch_predict.html',
                                 error='No file selected')
        
        if not file.filename.endswith('.csv'):
            return render_template('batch_predict.html',
                                 error='File must be CSV format')
        
        try:
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_input = csv.reader(stream)
            
            results = []
            for idx, row in enumerate(csv_input):
                if len(row) != 64:
                    continue
                
                try:
                    data_array = np.array([float(x) for x in row]).reshape(1, -1)
                    
                    if MODE_SIMULASI:
                        prediction_class = random.choice(list(GESTURE_MAP.keys()))
                        confidence = random.uniform(0.7, 0.99)
                    else:
                        data_scaled = (data_array - mean_scaler) / std_scaler
                        prediction_class = model.predict(data_scaled)[0]
                        probs = model.predict_proba(data_scaled)[0]
                        confidence = float(max(probs))
                    
                    results.append({
                        'sample_id': idx + 1,
                        'prediction': GESTURE_MAP[prediction_class],
                        'confidence': round(confidence * 100, 2)
                    })
                    
                except (ValueError, IndexError):
                    continue
            
            if not results:
                return render_template('batch_predict.html',
                                     error='No valid samples found in CSV')
            
            # Calculate summary
            gesture_counts = {}
            for r in results:
                gesture = r['prediction']
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            
            summary = {
                'total_samples': len(results),
                'gesture_distribution': gesture_counts,
                'avg_confidence': round(np.mean([r['confidence'] for r in results]), 2)
            }
            
            return render_template('batch_predict.html',
                                 results=results,
                                 summary=summary)
            
        except Exception as e:
            return render_template('batch_predict.html',
                                 error=f'Error processing file: {str(e)}')
    
    return render_template('batch_predict.html')

# ==================================
# 🚀 9. RUN APP
# ==================================
if __name__ == '__main__':
    app.run(debug=True)