from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from routers.id_parser import router as id_parser_router
from routers.metrics import router as metrics_router
from routers.optimize import router as optimize_router
from routers.video import router as video_router

app = FastAPI(title="MIDO-Chat")
templates = Jinja2Templates(directory="templates")

app.include_router(id_parser_router)
app.include_router(metrics_router)
app.include_router(optimize_router)
app.include_router(video_router)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request}) # ✅ 新版标准写法
