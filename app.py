import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# 1. Inisialisasi Aplikasi Flask
app = Flask(__name__)

# --- INI ADALAH BAGIAN KRUSIAL YANG DIREVISI ---
# 2. Load Model, Scaler, dan Metadata (dengan mekanisme Fallback/Simulasi)

# 2a. Definisikan nilai default (placeholder) jika model gagal dimuat
model = None 
mean_scaler = np.zeros(64)   # Placeholder
std_scaler = np.ones(64)     # Placeholder
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
    GESTURE_MAP = data_model['nama_gesture'] 
    akurasi_test = data_model['akurasi_test']
    parameter_terbaik = data_model['parameter_terbaik']
    MODE_SIMULASI = False
    
    print("✓ Model, Scaler, dan Metadata berhasil dimuat. Mode: Produksi.")

except Exception as e:
    # Tangkap semua jenis error (FileNotFound, KeyError, dll.)
    print(f"!!! GAGAL MEMUAT MODEL ({e}). Aplikasi berjalan dalam mode SIMULASI.")
    # Logika predict() akan crash saat mencoba predict_proba jika model=None,
    # jadi kita harus modifikasi fungsi predict() juga agar mengembalikan hasil simulasi.

# --- ROUTE APLIKASI ---

@app.route('/')
def index():
    """Halaman Utama/Index (Dashboard Proper)."""
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    """Halaman Informasi Detail Model (Metrik)."""
    # Siapkan data untuk ditampilkan di model_info.html
    info = {
        'akurasi': akurasi_test,
        'parameter': parameter_terbaik,
        # Menggunakan .get untuk penanganan yang lebih baik jika parameter terbaik tidak lengkap
        'kernel': parameter_terbaik.get('kernel', 'N/A'),
        'C': parameter_terbaik.get('C', 'N/A'),
        'gamma': parameter_terbaik.get('gamma', 'N/A'),
        'data_info': '11,000 samples (asumsi dari Kaggle), Split 80/20',
        'simulasi_mode': MODE_SIMULASI # Kirim info mode simulasi
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
            # Mengubah string input menjadi list of float
            data_list = [float(x.strip()) for x in emg_data_str.split(',')]
        except ValueError:
            return render_template('predict.html', error="Input harus berupa 64 nilai numerik yang dipisahkan koma.")
            
        # 3. Validasi Jumlah Fitur
        if len(data_list) != 64:
            return render_template('predict.html', error=f"Input harus tepat 64 nilai. Anda memasukkan {len(data_list)} nilai.")

        # --- LOGIKA PREDIKSI ---
        if MODE_SIMULASI:
            # Mode Simulasi: Bypass Model SC dan berikan hasil acak
            # Hasil kelas: 0-3
            prediction_class = np.random.randint(0, 4) 
            
            # Probabilitas: Semua sama (0.25) atau acak
            probabilities = np.ones(len(GESTURE_MAP)) / len(GESTURE_MAP)
            
            result_gesture = f"{GESTURE_MAP.get(prediction_class, 'Kelas Tidak Diketahui')} (SIMULASI)"

        else:
            # Mode Produksi: Gunakan Model SC yang Valid
            
            # 4. Ubah ke format yang dibutuhkan model (Array 2D: 1 sample, 64 fitur)
            final_input_raw = np.array(data_list).reshape(1, -1)

            # 5. WAJIB: Standardisasi data input user menggunakan scaler dari training
            final_input_scaled = (final_input_raw - mean_scaler) / std_scaler 

            # 6. Prediksi (kelas dan probabilitas)
            prediction_class = model.predict(final_input_scaled)[0]
            probabilities = model.predict_proba(final_input_scaled)[0] # Array 4 elemen

            # 7. Mapping Hasil Prediksi
            result_gesture = GESTURE_MAP.get(prediction_class, "Kelas Tidak Diketahui")
        
        # 8. Render halaman hasil (result.html)
        return render_template('result.html', 
                               gesture=result_gesture, 
                               probabilities=probabilities,
                               gesture_map=GESTURE_MAP,
                               input_data_64=data_list) # Kirim 64 data untuk visualisasi grid

    # Jika method adalah GET, tampilkan formulir input
    return render_template('predict.html')


# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)