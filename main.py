import uvicorn
from fastapi import FastAPI

from api import router as api_router
from services import RecognitionService


def create_app() -> FastAPI:
    app = FastAPI(title="Face Recognition API", version="1.0.0")

    @app.on_event("startup")
    def _startup() -> None:
        # pass
        # Initialize singleton service so first request isn't slow
        RecognitionService.instance()

    app.include_router(api_router, prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
