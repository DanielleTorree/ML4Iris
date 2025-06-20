from pydantic import BaseModel
from typing import List
from model.iris import Iris

class IrisSchema(BaseModel):
    """
        Representação de iris a ser inserido na base de dados
    """

    sepal_length_cm: float = 5.1
    sepal_width_cm: float = 3.5
    petal_length_cm: float = 1.4
    petal_width_cm: float = 0.2

class IrisViewSchema(BaseModel):
    """
        Representação de iris a ser retornada
    """
    id: int = 1
    sepal_length_cm: float = 5.1
    sepal_width_cm: float = 3.5
    petal_length_cm: float = 1.4
    petal_width_cm: float = 0.2
    specie: str = 'Iris-setosa'

class SearchIrisSchema(BaseModel):
    """
        Representação de busca do grupo de iris
    """
    specie: str = "Iris-setosa"

class ListIrisSchema(BaseModel):
    """
        Representação da lista de retorno de iris
    """
    iris: List[IrisSchema]

class DelIrisSchema(BaseModel):
    """
        Representação de exclusão de iris
    """

    id: int = 1

def show_iris(iris: Iris):
    """
        Representação do retorno de um iris
    """
    return {
        "id": iris.id,
        "sepal_length_cm": iris.sepal_length_cm,
        "sepal_width_cm": iris.sepal_width_cm,
        "petal_length_cm": iris.petal_length_cm,
        "petal_width_cm": iris.petal_width_cm,
        "specie": iris.species,
        "date": iris.date
    }

def show_iris_list(iris: List[Iris]):
    """
        Representação do retorno de uma lista de iris
    """
    result = []
    for i in iris:
        result.append({
            "id": i.id,
            "sepal_length_cm": i.sepal_length_cm,
            "sepal_width_cm": i.sepal_width_cm,
            "petal_length_cm": i.petal_length_cm,
            "petal_width_cm": i.petal_width_cm,
            "specie": i.species
        })
    return {"iris": result}