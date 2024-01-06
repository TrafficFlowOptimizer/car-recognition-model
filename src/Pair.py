from pydantic import BaseModel


class Pair(BaseModel):
    first: int
    second: int
