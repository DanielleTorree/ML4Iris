import pickle

class Model:

    def __init__(self):
        """
            Inicia o model
        """ 
        self.model = None

    def carrega_model(self, path):
        if path.endswith('.pkl'):
            with open(path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            raise Exception('Formato de arquivo não suportado.')
        return self.model

    def preditor(self, X_input):
        """
            Realiza a predição de iris com base no modelo treinado 
        """
        if self.model is None:
            raise Exception("Modelo não carregado.")
        diagnosis = self.model.predict(X_input)
        return diagnosis