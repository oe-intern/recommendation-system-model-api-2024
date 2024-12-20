from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, List, Any


class RecommedationBase(BaseModel):
    number_of_items: int
    products: Any
    type_scores: Any
    total: int
    product_scores: Any

    model_config = ConfigDict(from_attribute=True, extra="ignore")


# request
class PreRecommendationRequest(RecommedationBase):
    pass


class RecommendationRequest(RecommedationBase):
    pass

class NewProductRecommendationRequest(BaseModel):
    number_of_items: int
    products: Any
    product_id: str

    model_config = ConfigDict(from_attribute=True, extra="ignore")