import joblib
import numpy as np
import random 
from flask import Flask, render_template, request, redirect, url_for

# 1. Inisialisasi Aplikasi Flask
app = Flask(__name__)

# 2. Load Model, Scaler, dan Metadata (dengan mekanisme Fallback/Simulasi)
# Definisikan nilai default (placeholder) jika model gagal dimuat
model = None 
mean_scaler = np.zeros(64)   # Placeholder: Rata-rata 0
std_scaler = np.ones(64)     # Placeholder: Standar Deviasi 1
GESTURE_MAP = {0: 'Rock ✊', 1: 'Scissors ✌️', 2: 'Paper ✋', 3: 'OK 👌'} 
akurasi_test = 0.0           # Placeholder
parameter_terbaik = {'kernel': 'Simulasi', 'C': 'N/A', 'gamma': 'N/A'} # Placeholder
MODE_SIMULASI = True

try:
    # Model Anda menyimpan model, scaler, dan metadata dalam satu dictionary
    data_model = joblib.load('model_svm_emg.joblib')
    
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
    MODE_SIMULASI = False
    
    print("✓ Model, Scaler, dan Metadata berhasil dimuat. Mode: Produksi.")

except Exception as e:
    # Tangkap semua jenis error (FileNotFound, KeyError, dll.)
    print(f"!!! GAGAL MEMUAT MODEL ({e}). Aplikasi berjalan dalam mode SIMULASI.")
    print("!!! Prediksi akan mengembalikan hasil acak/default. Silakan masukkan model yang valid.")


# --- ROUTE APLIKASI ---

@app.route('/')
def index():
    """Halaman Utama/Index (Dashboard Proper)."""
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    """Halaman Informasi Detail Model (Metrik)."""
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
        if MODE_SIMULASI:
            # Mode Simulasi: Berikan hasil acak dan probabilitas merata
            
            # --- Perbaikan: Pilih kelas secara ACAL setiap kali POST ---
            prediction_class = random.choice(list(GESTURE_MAP.keys())) 
            # Probabilitas merata untuk simulasi (mis. 0.25, 0.25, 0.25, 0.25)
            probabilities = np.ones(len(GESTURE_MAP)) / len(GESTURE_MAP)
            
            result_gesture = f"{GESTURE_MAP.get(prediction_class, 'Kelas Tidak Diketahui')} (SIMULASI)"
            
        else:
            # Mode Produksi: Gunakan Model SC yang Valid
            final_input_raw = np.array(data_list).reshape(1, -1)
            final_input_scaled = (final_input_raw - mean_scaler) / std_scaler 

            # Prediksi (kelas dan probabilitas)
            prediction_class = model.predict(final_input_scaled)[0]
            probabilities = model.predict_proba(final_input_scaled)[0] 

            result_gesture = GESTURE_MAP.get(prediction_class, "Kelas Tidak Diketahui")
        
        # 8. Render halaman hasil (result.html)
        return render_template('result.html', 
                               gesture=result_gesture, 
                               probabilities=probabilities.tolist(),
                               gesture_map=GESTURE_MAP,
                               input_data_64=data_list) 

    # Jika method adalah GET, tampilkan formulir input
    return render_template('predict.html')


@app.route('/generate-random-sample', methods=['POST'])
def generate_random_sample():
    """
    Menghasilkan 64 nilai EMG acak dan mengirimkannya ke /predict.
    Asumsi rentang data EMG sekitar -50.0 hingga 50.0.
    """
    # Menghasilkan 64 float acak dalam rentang yang wajar
    random_data = [random.uniform(-50.0, 50.0) for _ in range(64)]
    
    # Konversi ke string format CSV
    emg_data_str = ', '.join([f"{x:.2f}" for x in random_data])
    
    # Kirim data ini sebagai POST request ke route /predict
    return redirect(url_for('predict', emg_data_random=emg_data_str))

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)