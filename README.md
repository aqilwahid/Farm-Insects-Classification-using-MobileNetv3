# Farm Insects Classification using MobileNetV3

Proyek PyTorch Lightning untuk mengklasifikasikan 15 jenis serangga peternakan menggunakan MobileNetV3. Proyek ini bertujuan untuk membantu dalam mengidentifikasi serangga yang berpotensi berbahaya di lingkungan pertanian, memungkinkan pengguna untuk mengambil tindakan yang tepat guna melindungi peternakan mereka.

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Perbaikan yang Dilakukan](#perbaikan-yang-dilakukan)
- [Persyaratan](#persyaratan)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Hasil](#hasil)
- [Visualisasi](#visualisasi)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)
- [Ucapan Terima Kasih](#ucapan-terima-kasih)

## Gambaran Umum

**Klasifikasi Serangga Peternakan** menggunakan dataset gambar yang dikurasi, menampilkan 15 jenis serangga yang umum ditemukan di lingkungan pertanian. Dengan memanfaatkan model MobileNetV3 yang telah dilatih sebelumnya dan melakukan fine-tuning pada dataset ini, proyek ini menyediakan metode yang efisien dan akurat untuk identifikasi serangga.

## Dataset

Dataset terdiri dari gambar berkualitas tinggi yang mewakili berbagai serangga, menyoroti fitur, warna, dan pola mereka yang unik. Setiap kelas serangga memiliki beberapa gambar untuk memastikan model belajar fitur yang robust.

- **Jumlah Kelas**: 15
- **Resolusi Gambar**: 224x224 piksel
- **Pembagian Data**:
  - Data Latih: 80%
  - Data Validasi: 10%
  - Data Uji: 10%

## Arsitektur Model

- **Model Dasar**: MobileNetV3 Large (pra-latih pada ImageNet)
- **Modifikasi**:
  - Mengganti lapisan classifier terakhir untuk menghasilkan 15 kelas.
  - Menambahkan teknik augmentasi dan normalisasi data.
  - Menerapkan penjadwalan learning rate dan early stopping.

## Perbaikan yang Dilakukan

Perbaikan berikut diterapkan untuk meningkatkan performa model dan kualitas output:

1. **Normalisasi Data**: Menerapkan normalisasi mean dan standar deviasi berdasarkan statistik ImageNet agar sesuai dengan ekspektasi model yang telah dilatih sebelumnya.

2. **Strategi Augmentasi**: Memisahkan transformasi untuk dataset latih dan validasi/pengujian agar augmentasi hanya diterapkan selama pelatihan.

3. **Penyesuaian Pembagian Data**: Meningkatkan ukuran data validasi menjadi 10% untuk evaluasi model yang lebih baik.

4. **Metode Akurasi**: Mengintegrasikan perhitungan akurasi selama fase pelatihan, validasi, dan pengujian.

5. **Scheduler Learning Rate**: Menerapkan scheduler `ReduceLROnPlateau` untuk menyesuaikan learning rate secara dinamis.

6. **Optimasi Perangkat**: Memastikan penggunaan GPU secara otomatis jika tersedia.

7. **Peningkatan Visualisasi**: Memperbaiki fungsi untuk menampilkan gambar dengan benar setelah invers normalisasi.

8. **Tuning Hyperparameter**: Mengoptimalkan learning rate dan weight decay untuk konvergensi yang lebih baik.

## Persyaratan

- Python 3.7 atau lebih tinggi
- PyTorch
- PyTorch Lightning
- torchvision
- wandb (Weights & Biases)
- matplotlib
- numpy
- scikit-learn

## Instalasi

1. **Kloning Repository**

   ```bash
   git clone https://github.com/username/klasifikasi-serangga-peternakan.git
   cd klasifikasi-serangga-peternakan
   ```

2. **Buat Lingkungan Virtual**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Pada Windows gunakan `venv\Scripts\activate`
   ```

3. **Instal Dependensi**

   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi Weights & Biases**

   Daftar untuk akun [Weights & Biases](https://wandb.ai/) dan login:

   ```bash
   wandb login
   ```

## Penggunaan

1. **Persiapkan Dataset**

   Letakkan dataset Anda dalam direktori dengan struktur berikut:

   ```plaintext
   data_path/
   ├── kelas1/
   │   ├── img1.jpg
   │   ├── img2.jpg
   │   └── ...
   ├── kelas2/
   │   ├── img1.jpg
   │   ├── img2.jpg
   │   └── ...
   └── ...
   ```

   Perbarui variabel `data_path` dalam kode untuk menunjuk ke lokasi dataset Anda.

2. **Jalankan Skrip Pelatihan**

   ```bash
   python train.py
   ```

   - Skrip akan secara otomatis menangani pelatihan, validasi, dan pengujian.
   - Log pelatihan dan metrik akan tersedia di dashboard Weights & Biases Anda.

3. **Evaluasi Model**

   - Setelah pelatihan, model akan dievaluasi pada set data uji.
   - Laporan klasifikasi dan metrik akurasi akan ditampilkan.

4. **Visualisasi Hasil**

   - Skrip menyertakan fungsi untuk memvisualisasikan sampel pelatihan dan hasil uji.
   - Plot akan ditampilkan menggunakan `matplotlib`.

## Hasil

Setelah melatih model dengan konfigurasi yang ditingkatkan, diperoleh hasil sebagai berikut:

- **Akurasi Uji**: *misalnya, 92%*

- **Laporan Klasifikasi**:

  ```plaintext
                    precision    recall  f1-score   support

         Kelas 1       0.93      0.90      0.92        20
         Kelas 2       0.95      0.94      0.94        18
         ...

        accuracy                           0.92       150
       macro avg       0.92      0.92      0.92       150
    weighted avg       0.92      0.92      0.92       150
  ```

## Visualisasi

### Contoh Gambar Pelatihan

![Gambar Pelatihan](images/training_samples.png)

### Contoh Hasil Uji

![Hasil Uji](images/test_results.png)

*Catatan: Gambar di atas menunjukkan label sebenarnya dan prediksi untuk sampel gambar dari dataset.*

## Kontribusi

Kontribusi sangat dihargai! Silakan ikuti langkah-langkah berikut:

1. **Fork** repository ini.

2. **Buat branch baru**:

   ```bash
   git checkout -b fitur/nama-fitur-anda
   ```

3. **Commit perubahan Anda**:

   ```bash
   git commit -m 'Menambahkan fitur tertentu'
   ```

4. **Push ke branch**:

   ```bash
   git push origin fitur/nama-fitur-anda
   ```

5. **Buat pull request**.
