import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file, output_file):
    # Membaca dataset dari file CSV
    data = pd.read_csv(input_file)
    
    # Menghapus kolom yang tidak diperlukan
    if 'Id' in data.columns:
        data.drop(['Id'], axis=1, inplace=True)
    
    # Menghapus data yang duplikat
    data.drop_duplicates(inplace=True)
    
    # Pengkodean kolom kategorikal (contoh)
    if 'Species' in data.columns:
        label_encoder = LabelEncoder()
        data['Species'] = label_encoder.fit_transform(data['Species'])
        
    # Menyimpan data yang telah diproses
    data.to_csv(output_file, index=False)
    print(f"Data diproses dan disimpan di {output_file}")

if __name__ == "__main__":
    input_file = '../data/Iris.csv'       # Path ke file data mentah
    output_file = '../data/Iris_Cleaned.csv' # Path ke file data yang telah diproses
    
    preprocess_data(input_file, output_file)
