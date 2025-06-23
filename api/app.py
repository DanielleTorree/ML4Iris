from flask_openapi3 import OpenAPI, Info, Tag
from flask import redirect
from flask import request
from urllib.parse import unquote

from model import *
from logger import logger
from schemas import *
from flask_cors import CORS

from sqlalchemy import and_

# Instância de API 
info = Info(title="My API", version="1.0.0")
app = OpenAPI(__name__, info=info, static_folder='../front', static_url_path='/front')
CORS(app)

# Tags para agrupamento das rotas
home_tag = Tag(name="Documentação", description="Seleção de documentação: Swagger, Redoc ou RapiDoc")
iris_tag = Tag(name="Iris", description="Adição, visualização, remoção e predição de iris")

@app.get('/', tags=[home_tag])
def home():
    """
        Redireciona para o index.html do front.
    """
    return redirect('/front/index.html')

@app.get('/docs', tags=[home_tag])
def docs():
    """
        Redireciona para /openapi
    """
    return redirect('/openapi')

@app.get('/list-iris', tags=[iris_tag],
         responses={"200": IrisViewSchema, "404": ErrorSchema})
def get_iris_list(): 
    """
        Lista todas iris cadastradas na base
    """
    logger.debug("Coletando dados sobre as iris")
    
    # Conecxão com a base de dados
    session = Session()
    iris_list = session.query(Iris).all()

    if not iris_list:
        return {"iris": []}, 200
    else:
        logger.debug("%d iris encontradas" % len(iris_list))
        print(iris_list)
        return show_iris_list(iris_list), 200
    
# Rota de adição de iris
@app.post('/iris', tags=[iris_tag],
          responses={"200": IrisViewSchema, "400": ErrorSchema, "409": ErrorSchema})
def predict(form: IrisSchema):
    """
        Adiciona uma nova iris à base dados 
        e retorna uma representação das iris com a predição associada
    """

    preprocessador = PreProcessador()
    pipeline = Pipeline()
    
    # Recupera os dados do form
    sepal_length_cm = form.sepal_length_cm 
    sepal_width_cm = form.sepal_width_cm
    petal_length_cm = form.petal_length_cm
    petal_width_cm = form.petal_width_cm

    # Prepara os dados para o modelo
    X_input = preprocessador.preparar_form(form)

    # Carrega o modelo
    model_path = 'api/MachineLearning/pipelines/rf_iris_pipeline.pkl'
    modelo = pipeline.carrega_pipeline(model_path)

    # Realiza a predição
    species = modelo.predict(X_input)[0]

    iris = Iris(
        sepal_length_cm = sepal_length_cm,
        sepal_width_cm = sepal_width_cm,
        petal_length_cm = petal_length_cm,
        petal_width_cm = petal_width_cm,
        species = species
    )
    
    logger.debug(f"Adiciona iris com características: \
                '{iris.sepal_length_cm}' - \
                '{iris.sepal_width_cm}' - \
                '{iris.petal_length_cm}' - \
                '{iris.petal_width_cm}'")
    
    try:
        # Cria conexão com a base de dados
        session = Session()
        
        # Adiciona iris
        session.add(iris)
        # Commit
        session.commit()
        # Conclui transação
        logger.debug(f"Adicionada iris com características: \
                '{iris.sepal_length_cm}' - \
                '{iris.sepal_width_cm}' - \
                '{iris.petal_length_cm}' - \
                '{iris.petal_width_cm}'")
    
        return show_iris(iris), 200
    except Exception as e:
        error_message = "Não foi possível salver nova iris"
        logger.warning(f"Erro ao adicionar nova iris {error_message}")
        return {"message", error_message}, 400
    
# Rota de exclusão de iris
@app.delete('/iris', tags=[iris_tag],
            responses={"200": IrisViewSchema, "400": ErrorSchema, "409": ErrorSchema})
def delete_iris(query: DelIrisSchema):
    """
        Exclui uma iris a partir do id
    """
    iris_id = query.id
    logger.debug(f"Excluindo dados sobre iris de id #{id}")

    try:
        # Cria conexão com a base de dados
        session = Session()

        # Busca iris
        iris = session.query(Iris).filter(Iris.id == iris_id).first()

        if not iris:
            error_message = "Iris não encontrada na base de dados"
            logger.warning(f"Erro ao excluir iris de id '{iris_id}'")
            return {"message": error_message}, 404
        
        session.delete(iris)
        session.commit()
        logger.debug(f"Iris excluída id: #{iris_id} e espécie: #{iris.species}")
        return {"message": f"Iris => id: #{iris_id} e espécie: #{iris.species} removida com sucesso"}, 200
    except Exception as e:
        error_message = "Não foi possível excluir iris"
        logger.warning(f"Erro ao tentar excluir iris {error_message}")
        return {"message", error_message}, 400

if __name__ == '__main__':
    app.run(debug=True)