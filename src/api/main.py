import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.seller_assessment import router as seller_assessment_router
from src.api.seller_intelligence import router as seller_intelligence_router


app = FastAPI(
    title="AI Seller Risk Intelligence API",
    description="API for seller delivery and review risk analysis.",
    version="1.0.0",
)


# Keep local development origins available.
allowed_origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]

# Add the deployed frontend URL through an environment variable.
frontend_urls = os.getenv("FRONTEND_URL", "")

for url in frontend_urls.split(","):
    url = url.strip().rstrip("/")
    if url and url not in allowed_origins:
        allowed_origins.append(url)


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)
app.include_router(seller_assessment_router)
app.include_router(seller_intelligence_router)