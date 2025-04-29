from pydantic import BaseModel


class Dataset(BaseModel):
    """Dataset Schema"""
   
    def to_json(self) -> str:
            """ export dataset to json """
            return ""