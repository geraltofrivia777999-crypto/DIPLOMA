"""Authentication router."""
from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends

from ..auth import create_access_token
from ..schemas import TokenOut
from ..settings import settings

router = APIRouter()


@router.post("/login", response_model=TokenOut)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username != settings.admin_username or form.password != settings.admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = create_access_token(form.username)
    return TokenOut(access_token=token)
