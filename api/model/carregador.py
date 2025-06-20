import pandas as pd

class Carregador:

    def __init__(self):
        """
            Inicia o carregador
        """

    def carregar_dados(self, url: str, atributos: list):
        """
            Carrega e retorna um dataframe
        """
        return pd.read_csv(url, names=atributos, header=0,
                           skiprows=1, delimiter=',')