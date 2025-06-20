from sqlalchemy import Column, String, Integer, DateTime, Float
from datetime import datetime
from typing import Union

from model import Base

#Colunas = SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species

class Iris(Base): 
    __tablename__ = 'iris'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length_cm = Column(Float, nullable=False)
    sepal_width_cm = Column(Float, nullable=False)
    petal_length_cm = Column(Float, nullable=False)
    petal_width_cm = Column(Float, nullable=False)
    species = Column(String, nullable=False) 
    date = Column("Date", DateTime, default=datetime.now)

    def __init__(self, 
                 sepal_length_cm:float,
                 sepal_width_cm:float,
                 petal_length_cm:float,
                 petal_width_cm:float,
                 species:str,
                 date:Union[DateTime, None] = None): 
        """
        Cria um objeto Iris

        Arguments:
            sepal_length_cm: comprimento da sépala em centímetros (número decimal)
            sepal_width_cm: largura da sépala em centímetros (número decimal)
            petal_length_cm: comprimento da pétala em centímetros (número decimal)
            petal_width_cm: largura da pétala em centímetros (número decimal)
            species: espécie (texto)
            date: data (usada para registrar data/hora, tipo datetime)
        """

        self.sepal_length_cm=sepal_length_cm
        self.sepal_width_cm=sepal_width_cm
        self.petal_length_cm=petal_length_cm
        self.petal_width_cm=petal_width_cm
        self.species=species
        self.date=date or datetime.now()