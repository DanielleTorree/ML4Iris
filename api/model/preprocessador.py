from sklearn.model_selection import train_test_split
import pickle
import numpy as np

class PreProcessador:
    def __init__(self):
        pass

    def separa_teste_treino(self, dataset, percentual_teste, seed=7):
        # Divisão em treino e teste
        X_train, X_test, Y_train, Y_test = self.__preparar_holdout(dataset, percentual_teste, seed)

        # Normalização e padronização
        return (X_train, X_test, Y_train, Y_test)

    def __preparar_holdout(self, dataset, percentual_teste, seed):
        """
            Divide os dados em treino e teste.
            Considera que a última coluna é target.
        """
        dados = dataset.values
        X = dados[:, 0:-1]
        Y = dados[:, -1]
        return train_test_split(X, Y, test_size=percentual_teste, random_state=seed)
    
    def preparar_form(self, form):
        """
            Prepara os dados recebidos do frontend para serem utilizados no modelo.
        """
        X_input = np.array([
            form.sepal_length_cm,
            form.sepal_width_cm,
            form.petal_length_cm,
            form.petal_width_cm
        ])

        # Reshape para o modelo entende que está sendo passado
        X_input = X_input.reshape(1, -1)

        return  X_input

    def scaler(self, X_train):
        """
            Normalização de dados
        """
        scaler = pickle.load(open('./MachineLearning/scalers_minmax_scaler_iris.pkl', 'rb'))
        reescaled_X_train = scaler.transform(X_train)
        return reescaled_X_train