import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import pickle

def evaluate_model(test_data_file, model_file):
    # Membaca data pengujian
    X_test = pd.read_csv(test_data_file + '_X_test.csv')
    y_test = pd.read_csv(test_data_file + '_y_test.csv')
    
    # Memuat model dari file
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Menampilkan hasil evaluasi
    print("Akurasi Model:", accuracy)
    print("Laporan Klasifikasi:")
    print(report)

if __name__ == "__main__":
    test_data_file = '../data/valid'   # Prefix file data pengujian
    model_file = '../models/trained_model.pkl'  # Path ke file model
    
    evaluate_model(test_data_file, model_file)
