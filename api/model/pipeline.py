import pickle

class Pipeline:
    def __init__(self):
        self.pipeline = None

    def carrega_pipeline(self, path):
        """
            é carregado o pipe no momento de fase de treinamento
        """
        with open(path, 'rb') as file:
            self.pipeline = pickle.load(file)
        return self.pipeline