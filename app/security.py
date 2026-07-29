import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import settings


def verify_api_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
        )

    scheme, separator, supplied_key = authorization.partition(" ")

    if separator != " " or scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Use a Bearer API key.",
        )

    if not hmac.compare_digest(
        supplied_key,
        settings.api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The supplied API key is invalid.",
        )