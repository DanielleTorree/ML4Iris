from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os

from model.base import Base 
from model.iris import Iris
from model.modelo import Model
from model.pipeline import Pipeline
from model.preprocessador import PreProcessador
from model.avaliador import Avaliador
from model.carregador import Carregador

db_path = 'database/'
if not os.path.exists(db_path):
    # Cria o diretório
    os.makedirs(db_path)

# Url de acesso ao repo local
db_url = 'sqlite:///%s/iris.sqlite3' % db_path

engine = create_engine(db_url, echo=False)

Session = sessionmaker(bind=engine)

if not database_exists(engine.url):
    # Cria o banco de dados
    create_database(engine.url)

# Cria as tabelas
Base.metadata.create_all(engine)
