# ==================================
# 📚 1. IMPORTS & KONFIGURASI
# ==================================
import joblib
import numpy as np
import random 
from flask import Flask, render_template, request, redirect, url_for, session

# --- KONSTANTA & VARIABEL GLOBAL ---
MODEL_PATH = 'model_svm_emg.joblib'
GESTURE_MAP = {0: 'Rock ✊', 1: 'Scissors ✌️', 2: 'Paper ✋', 3: 'OK 👌'} 

# --- PLACEHOLDERS (Digunakan jika Mode Simulasi Aktif) ---
model = None 
mean_scaler = np.zeros(64)   
std_scaler = np.ones(64)     
akurasi_test = 0.0           
parameter_terbaik = {'kernel': 'Simulasi', 'C': 'N/A', 'gamma': 'N/A'} 
sample_data = None  # Dictionary berisi sample data per kelas
MODE_SIMULASI = True # Default: Berjalan dalam mode simulasi

# ==================================
# ⚙️ 2. INISIALISASI APLIKASI
# ==================================
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'  # Untuk session

# ==================================
# 💾 3. LOAD MODEL DENGAN FALLBACK
# ==================================
try:
    # Model Anda menyimpan model, scaler, dan metadata dalam satu dictionary
    data_model = joblib.load(MODEL_PATH)
    
    # Inisialisasi dengan data yang dimuat
    model = data_model['model']
    mean_scaler = data_model['mean_scaler']
    std_scaler = data_model['std_scaler']
    
    # Pastikan GESTURE_MAP berbentuk Dictionary
    if isinstance(data_model['nama_gesture'], dict):
        GESTURE_MAP = data_model['nama_gesture']
    else:
        # Jika bukan dict, buat mapping dari 0, 1, 2, 3
        GESTURE_MAP = dict(enumerate(data_model['nama_gesture']))

    akurasi_test = data_model.get('akurasi_test', 0.0)
    parameter_terbaik = data_model.get('parameter_terbaik', parameter_terbaik)
    
    # Load sample data jika tersedia
    sample_data = data_model.get('sample_data', None)
    
    MODE_SIMULASI = False
    
    print("✓ Model, Scaler, dan Metadata berhasil dimuat. Mode: Produksi.")
    if sample_data is not None:
        print(f"✓ Sample data tersedia untuk {len(sample_data)} kelas.")
    else:
        print("⚠ Sample data tidak tersedia. Random sample akan menggunakan synthetic data.")

except Exception as e:
    # Tangkap semua jenis error (FileNotFound, KeyError, dll.)
    print(f"!!! GAGAL MEMUAT MODEL ({e}). Aplikasi berjalan dalam mode SIMULASI.")
    print("!!! Prediksi akan mengembalikan hasil acak/default. Silakan masukkan model yang valid.")


# ==================================
# 🎲 HELPER: GENERATE BALANCED RANDOM SAMPLE
# ==================================
def generate_balanced_random_sample(gesture_class=None):
    """
    Generate random EMG sample dengan distribusi yang lebih balanced.
    
    Args:
        gesture_class (int, optional): Kelas gestur spesifik (0-3). 
                                       Jika None, pilih random dari semua kelas.
    
    Returns:
        tuple: (emg_data_array, gesture_class)
    """
    
    # 1. Jika sample_data tersedia, ambil dari situ (BEST OPTION)
    if sample_data is not None and not MODE_SIMULASI:
        # Pilih kelas
        if gesture_class is None:
            gesture_class = random.choice(list(sample_data.keys()))
        
        # Validasi kelas
        if gesture_class not in sample_data or len(sample_data[gesture_class]) == 0:
            gesture_class = random.choice(list(sample_data.keys()))
        
        # Ambil random sample dari kelas tersebut
        samples = sample_data[gesture_class]
        random_idx = random.randint(0, len(samples) - 1)
        random_sample = samples[random_idx]
        
        print(f"✓ Using real sample for class {gesture_class}: {GESTURE_MAP[gesture_class]}")
        return random_sample, gesture_class
    
    # 2. Fallback: Generate synthetic data dengan pola berbeda per kelas
    print("⚠ Using synthetic data (no sample_data available)")
    
    if gesture_class is None:
        gesture_class = random.randint(0, 3)
    
    # IMPROVED: Karakteristik yang lebih berbeda antar kelas
    # Menggunakan pola spasial berbeda untuk setiap gestur
    
    if gesture_class == 0:  # Rock ✊ - Aktivitas tinggi di semua sensor
        base_values = np.random.uniform(20, 35, 64)
        noise = np.random.normal(0, 5, 64)
        
    elif gesture_class == 1:  # Scissors ✌️ - Aktivitas tinggi di sensor tertentu (pola V)
        base_values = np.random.uniform(5, 15, 64)
        # Tingkatkan aktivitas di sensor 2, 3 (jari telunjuk & tengah)
        for i in range(8):
            base_values[i*8 + 2] += random.uniform(15, 25)
            base_values[i*8 + 3] += random.uniform(15, 25)
        noise = np.random.normal(0, 3, 64)
        
    elif gesture_class == 2:  # Paper ✋ - Aktivitas merata tapi lebih rendah
        base_values = np.random.uniform(8, 18, 64)
        noise = np.random.normal(0, 4, 64)
        
    else:  # OK 👌 - Aktivitas tinggi di ibu jari & telunjuk
        base_values = np.random.uniform(10, 20, 64)
        # Tingkatkan aktivitas di sensor 1, 2 (ibu jari & telunjuk)
        for i in range(8):
            base_values[i*8 + 1] += random.uniform(10, 20)
            base_values[i*8 + 2] += random.uniform(10, 20)
        noise = np.random.normal(0, 4, 64)
    
    emg_data = base_values + noise
    
    # CRITICAL FIX: Jangan scale ulang jika model butuh data raw!
    # Hanya normalize jika model trained dengan normalized data
    if not MODE_SIMULASI and np.std(std_scaler) > 0.1:
        # Model menggunakan standardization, jadi kita perlu match distribusinya
        # Tapi JANGAN denormalize, biarkan dalam rentang raw values
        pass  # Data sudah dalam rentang yang sesuai
    
    print(f"✓ Generated synthetic sample for class {gesture_class}: {GESTURE_MAP[gesture_class]}")
    print(f"  Sample stats - Mean: {np.mean(emg_data):.2f}, Std: {np.std(emg_data):.2f}")
    
    return emg_data, gesture_class


# ==================================
# --- ROUTE APLIKASI ---
# ==================================

@app.route('/')
def landing():
    """Halaman Awal (Landing Page) dengan tombol Masuk."""
    return render_template('landing.html') 

@app.route('/dashboard')
def index():
    """Halaman Utama/Dashboard (Overview)."""
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    """Halaman Informasi Detail Model (Metrik)."""
    # Info dikumpulkan di sini, karena memang dibutuhkan di template model_info.html
    info = {
        'akurasi': akurasi_test,
        'parameter': parameter_terbaik,
        'kernel': parameter_terbaik.get('kernel', 'N/A'),
        'C': parameter_terbaik.get('C', 'N/A'),
        'gamma': parameter_terbaik.get('gamma', 'N/A'),
        'data_info': '11.678 samples, Split 80/20',
        'simulasi_mode': MODE_SIMULASI 
    }
    return render_template('model_info.html', info=info)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Halaman Input Data EMG dan Prediksi."""
    
    # Cek apakah ada parameter URL yang memaksa simulasi (dari generate-random-sample)
    is_forced_simulasi = request.args.get('force_sim', 'false') == 'true'
    expected_gesture = request.args.get('expected_gesture', None)
    
    if request.method == 'POST':
        # 1. Ambil data dari formulir
        emg_data_str = request.form.get('emg_data')
        
        # 2. Pre-processing Input (String 64 Angka)
        try:
            data_list = [float(x.strip()) for x in emg_data_str.split(',')]
        except ValueError:
            return render_template('predict.html', error="Input harus berupa 64 nilai numerik yang dipisahkan koma.")
            
        # 3. Validasi Jumlah Fitur
        if len(data_list) != 64:
            return render_template('predict.html', error=f"Input harus tepat 64 nilai. Anda memasukkan {len(data_list)} nilai.")

        # --- LOGIKA PREDIKSI ---
        # Gunakan simulasi jika MODE_SIMULASI aktif ATAU jika simulasi dipaksa
        if MODE_SIMULASI or is_forced_simulasi: 
            # Mode Simulasi: Berikan hasil acak dan probabilitas merata
            prediction_class = random.choice(list(GESTURE_MAP.keys())) 
            probabilities = np.ones(len(GESTURE_MAP)) / len(GESTURE_MAP)
            
            result_gesture = f"{GESTURE_MAP.get(prediction_class, 'Kelas Tidak Diketahui')} (SIMULASI)"
            
        else:
            # Mode Produksi: Gunakan Model yang Valid
            final_input_raw = np.array(data_list).reshape(1, -1)
            final_input_scaled = (final_input_raw - mean_scaler) / std_scaler 

            # Prediksi (kelas dan probabilitas)
            prediction_class = model.predict(final_input_scaled)[0]
            
            # Cek apakah model mendukung predict_proba
            if hasattr(model, 'predict_proba'):
                 probabilities = model.predict_proba(final_input_scaled)[0]
            else:
                 probabilities = np.ones(len(GESTURE_MAP)) * 0.0
                 probabilities[prediction_class] = 1.0

            result_gesture = GESTURE_MAP.get(prediction_class, "Kelas Tidak Diketahui")
        
        # Render halaman hasil
        return render_template('result.html', 
                               gesture=result_gesture, 
                               probabilities=probabilities.tolist(),
                               gesture_map=GESTURE_MAP,
                               input_data_64=data_list,
                               expected_gesture=expected_gesture) 

    # Jika method adalah GET, tampilkan formulir input
    emg_data_random = request.args.get('emg_data_random', '')
    return render_template('predict.html', emg_data_default=emg_data_random)


@app.route('/generate-random-sample', methods=['POST'])
def generate_random_sample():
    """
    Menghasilkan 64 nilai EMG yang lebih balanced dan mengirimkannya ke /predict.
    Mendukung pemilihan gestur spesifik jika diminta.
    """
    
    # Ambil parameter gesture_class jika ada (dari tombol spesifik)
    gesture_class_str = request.form.get('gesture_class', None)
    
    if gesture_class_str is not None:
        try:
            gesture_class = int(gesture_class_str)
        except ValueError:
            gesture_class = None
    else:
        gesture_class = None
    
    # Generate sample yang lebih balanced
    emg_data_array, selected_gesture = generate_balanced_random_sample(gesture_class)
    
    # Konversi ke string format CSV
    emg_data_str = ', '.join([f"{x:.2f}" for x in emg_data_array.tolist()])
    
    # Kirim data dengan informasi gesture yang diharapkan
    return redirect(url_for('predict', 
                           emg_data_random=emg_data_str, 
                           force_sim='false',
                           expected_gesture=GESTURE_MAP.get(selected_gesture, 'Unknown')))


# ==================================
# 🆕 ROUTE BARU: DIAGNOSTIC TOOLS
# ==================================
@app.route('/diagnostic')
def diagnostic():
    """Halaman diagnostic untuk check model & data."""
    
    diagnostics = {
        'mode': 'SIMULASI' if MODE_SIMULASI else 'PRODUKSI',
        'model_loaded': model is not None,
        'sample_data_available': sample_data is not None,
        'scaler_stats': {
            'mean_range': f"{np.min(mean_scaler):.2f} to {np.max(mean_scaler):.2f}",
            'std_range': f"{np.min(std_scaler):.2f} to {np.max(std_scaler):.2f}",
        },
        'gesture_map': GESTURE_MAP,
        'test_results': []
    }
    
    # Test prediction dengan synthetic data untuk setiap kelas
    if not MODE_SIMULASI and model is not None:
        for class_id in range(4):
            # Generate sample
            test_data, expected_class = generate_balanced_random_sample(class_id)
            
            # Predict
            test_input = test_data.reshape(1, -1)
            test_scaled = (test_input - mean_scaler) / std_scaler
            predicted_class = model.predict(test_scaled)[0]
            
            diagnostics['test_results'].append({
                'expected': GESTURE_MAP[expected_class],
                'predicted': GESTURE_MAP[predicted_class],
                'match': expected_class == predicted_class,
                'data_stats': f"mean={np.mean(test_data):.2f}, std={np.std(test_data):.2f}"
            })
    
    return render_template('diagnostic.html', diagnostics=diagnostics)


# ==================================
# --- ROUTE APLIKASI ---
# ==================================
@app.route('/generate-specific-sample/<int:gesture_class>', methods=['POST'])
def generate_specific_sample(gesture_class):
    """Generate sample untuk gestur spesifik."""
    
    if gesture_class not in GESTURE_MAP:
        return redirect(url_for('predict'))
    
    # Generate sample untuk kelas spesifik
    emg_data_array, _ = generate_balanced_random_sample(gesture_class)
    
    # Konversi ke string format CSV
    emg_data_str = ', '.join([f"{x:.2f}" for x in emg_data_array.tolist()])
    
    # Kirim data dengan informasi gesture yang diharapkan
    return redirect(url_for('predict', 
                           emg_data_random=emg_data_str, 
                           force_sim='false',
                           expected_gesture=GESTURE_MAP.get(gesture_class, 'Unknown')))


# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)