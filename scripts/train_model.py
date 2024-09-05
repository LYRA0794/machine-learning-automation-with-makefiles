import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

def train_model(data_file, model_file, valid_file_prefix):
    # Membaca data yang telah diproses
    data = pd.read_csv(data_file)
    
    # Memisahkan fitur dan target
    X = data.drop('Species', axis=1)
    y = data['Species']
    
    # Membagi data menjadi data pelatihan dan data sementara (untuk validasi dan pengujian)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=10)
    
    # Membagi data sementara menjadi data validasi dan data pengujian
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=10)
    
    # Simpan data validasi dan pengujian untuk digunakan nanti
    X_val.to_csv(f'{valid_file_prefix}_X_val.csv', index=False)
    y_val.to_csv(f'{valid_file_prefix}_y_val.csv', index=False)
    X_test.to_csv(f'{valid_file_prefix}_X_test.csv', index=False)
    y_test.to_csv(f'{valid_file_prefix}_y_test.csv', index=False)
    
    # Inisialisasi dan pelatihan model Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    # Mengevaluasi model menggunakan data validasi
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Menampilkan hasil evaluasi pada data validasi
    print("Akurasi Model pada Data Validasi:", val_accuracy)

    # Menyimpan model ke file
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model disimpan di {model_file}")

if __name__ == "__main__":
    data_file = '../data/Iris_Cleaned.csv'  # Path ke file data yang telah diproses
    model_file = '../models/trained_model.pkl'        # Path ke file tempat menyimpan model
    valid_file_prefix = '../data/valid'        # Prefix untuk nama file data validasi dan pengujian
    
    train_model(data_file, model_file, valid_file_prefix)
