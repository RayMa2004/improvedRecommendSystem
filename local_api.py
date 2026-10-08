"""一个适合学习的本地推荐接口。

这个文件复用根目录 main.py 中的 CSV 版推荐系统，不连接 MySQL。
请从项目根目录启动：

    python -m uvicorn local_api:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import RecommenderSystem


recommender: Optional[RecommenderSystem] = None


class RecommendationRequest(BaseModel):
    """客户端提交的推荐请求。"""

    user_id: str = Field(min_length=1, description="例如 U00001")
    hour: int = Field(ge=0, le=23, description="当前小时，范围 0-23")
    is_weekend: bool
    is_holiday: bool


class RecommendationResponse(BaseModel):
    status: str
    user_id: str
    recommendations: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时加载一次模型，之后请求直接复用内存中的模型。"""

    global recommender
    recommender = RecommenderSystem()
    recommender.fit_with_weights(
        "model_weights/improved_twin_towers_model.pth",
        "model_weights/lightgcn.pth",
        "model_weights/three_towers_model.pth",
        "model_weights/multi_task_model.pth",
    )
    yield
    recommender = None


app = FastAPI(
    title="Local Recommendation Service",
    description="CSV-backed recommendation service for learning",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """检查推荐模型是否已经加载。"""

    return {"status": "healthy" if recommender is not None else "starting"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    """根据用户和当前场景返回推荐物品 ID。"""

    if recommender is None:
        raise HTTPException(status_code=503, detail="recommendation model is not ready")

    known_user = any(user.user_id == request.user_id for user in recommender.users)
    if not known_user:
        raise HTTPException(status_code=404, detail=f"unknown user: {request.user_id}")

    recommendations = recommender.recommend(
        request.user_id,
        request.hour,
        request.is_weekend,
        request.is_holiday,
    )
    return RecommendationResponse(
        status="success",
        user_id=request.user_id,
        recommendations=recommendations,
    )

