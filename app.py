from fastapi import FastAPI
from clay_pipeline import router as clay_router

app = FastAPI()

# include clay pipeline routes
app.include_router(clay_router)

@app.get("/")
def root():
    return {"message": "Clay API is running"}
