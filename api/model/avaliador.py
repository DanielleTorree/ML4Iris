from sklearn.metrics import accuracy_score

class Avaliador:

    def __init__(self): 
        """
            Inicia o avaliador
        """
        pass
    
    def avaliador(self, model, X_test, Y_test):
        """
            Realiza uma predição e avalia o modelo
        """
        predicoes = model.predict(X_test)

        return accuracy_score(Y_test, predicoes)