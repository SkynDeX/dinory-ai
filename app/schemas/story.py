# app/schemas/story.py
from pydantic import BaseModel, Field, AliasChoices, ConfigDict

class NextSceneRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    completion_id: int = Field(
        ...,
        validation_alias=AliasChoices('completionId', 'completion_id')
    )
    scene: int = Field(..., validation_alias=AliasChoices('scene', 'scene'))
    choice: str = Field(..., validation_alias=AliasChoices('choice', 'choice'))
