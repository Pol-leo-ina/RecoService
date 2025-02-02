import os
from typing import List

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKey
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UnAuthorizedError, UserNotFoundError
from service.log import app_logger

# Load environment variables from .env file (with api token)
load_dotenv()


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
bearer = HTTPBearer()
API_KEY = os.getenv("API_KEY")


async def verify_token(
    key: HTTPAuthorizationCredentials = Security(bearer),
) -> str:
    if key.credentials == API_KEY:
        return key.credentials
    raise UnAuthorizedError()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        401: {"description": "Invalid api token"},
        404: {"description": "Invalid model name or user not found"},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_key: APIKey = Depends(verify_token),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    # Write your code here
    if model_name == "range_test":
        reco = list(range(10))
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    reco = list(range(k_recs))
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
