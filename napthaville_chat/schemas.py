from pydantic import BaseModel


class InputSchema(BaseModel):
    init_persona: str
    target_persona: str
