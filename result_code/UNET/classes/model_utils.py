from keras.models import model_from_json


class ModelUtils:

    @staticmethod
    def load_model(path_to_model):
        json_file = open(path_to_model + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(path_to_model + '.h5')
        print("Loaded model from disk")
        return model

    @staticmethod
    def save_model(path_to_model, model):
        model_json = model.to_json()
        with open(path_to_model + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(path_to_model + ".h5")
        print("Saved model to disk")