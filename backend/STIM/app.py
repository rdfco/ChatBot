import logging
from typing import Any, AsyncContextManager, Callable, Mapping, Self

import fastapi
from fastapi import Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from STIM.api.auth.router import router as auth_router
from STIM.api.connection.router import router as connection_router
from STIM.api.conversation.router import router as conversation_router
from STIM.api.result.router import router as result_router
from STIM.api.settings.router import router as settings_router
from STIM.auth import authenticate
from STIM.config import config
from STIM.errors import UserFacingError, ValidationError
from STIM.repositories.base import NotFoundError, NotUniqueError

logger = logging.getLogger(__name__)


def handle_exceptions(request: Request, e: Exception) -> JSONResponse:
    if isinstance(e, NotFoundError):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": e.message})
    elif isinstance(e, NotUniqueError):
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": e.message})
    elif isinstance(e, ValidationError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(e)})
    elif isinstance(e, UserFacingError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(e)})

    logger.exception(e)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(e)})


class App(fastapi.FastAPI):
    def __init__(  # type: ignore[misc]
        self,
        lifespan: Callable[[Self], AsyncContextManager[Mapping[str, Any]]] | None = None,
    ) -> None:
        super().__init__(title="STIM API", lifespan=lifespan)
        self.add_middleware(
            CORSMiddleware,
            allow_origins=config.allowed_origins.split(",") if config.has_auth else ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        common_dependencies = []
        if config.has_auth:
            common_dependencies = [Depends(authenticate)]

            # Add route for login
            self.include_router(auth_router)

        self.include_router(settings_router, dependencies=common_dependencies)
        self.include_router(connection_router, dependencies=common_dependencies)
        self.include_router(conversation_router, dependencies=common_dependencies)
        self.include_router(result_router, dependencies=common_dependencies)

        # Handle 500s separately to play well with TestClient and allow re-raising in tests
        self.add_exception_handler(NotFoundError, handle_exceptions)
        self.add_exception_handler(NotUniqueError, handle_exceptions)
        self.add_exception_handler(ValidationError, handle_exceptions)
        self.add_exception_handler(UserFacingError, handle_exceptions)
        self.add_exception_handler(Exception, handle_exceptions)
