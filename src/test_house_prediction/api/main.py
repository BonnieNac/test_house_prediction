"""Main entry point for the FastAPI application."""

from fastapi import FastAPI

from test_house_prediction.api.middlewares import LimitUploadSizeMiddleware
from test_house_prediction.api.routers import base, greetings, predict, system
from test_house_prediction.core.utils import ensure_dirs_exist, get_project_version

# Initialize project directories on startup
ensure_dirs_exist()

app = FastAPI(
    title="test_house_prediction",
    description="API for test_house_prediction package",
    version=get_project_version(),
)

# Global upload size limit middleware
app.add_middleware(LimitUploadSizeMiddleware)


app.include_router(base.router)
app.include_router(system.router)
app.include_router(greetings.router)
app.include_router(predict.router)


def main() -> None:  # pragma: no cover
    """Run the FastAPI application using uvicorn."""
    import uvicorn

    uvicorn.run("test_house_prediction.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
