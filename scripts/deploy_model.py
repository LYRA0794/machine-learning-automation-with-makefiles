import pickle
import shutil

def deploy_model(model, destination_file):
    # Menyimpan model ke format pickle
    with open(destination_file, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    model_file = '../models/trained_model.pkl'  # Nama file model yang sudah dilatih
    destination_file = '../models/deployment_model.pkl'
    
    # Muat model dari file
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    
    deploy_model(model, destination_file)

