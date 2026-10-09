from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from agent import shopping_chain

app = FastAPI(
    title="AI Shopping Assistant",
    version="1.0",
    description="AI Shopping Assistant using LangChain and LangServe"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_routes(
    app,
    shopping_chain,
    path="/shopping-agent"
)

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "healthy"}
