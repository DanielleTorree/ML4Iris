from pydantic import BaseModel

class ErrorSchema(BaseModel):
    """
        Representação de mensagem de erro
    """
    message: str