from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from Workflow import app as workflow_app
from queries import validate_conversation


app = FastAPI(
    title="Medical RAG API",
    description="API for the AI Medical Assistant using LangGraph and RAG",
    version="1.0.0"
)


class SymptomRequest(BaseModel):
    user_id: str
    conversation_id: str
    symptoms: str


class AnalysisResponse(BaseModel):
    final_advice: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_symptoms(request: SymptomRequest):

    if not request.symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms cannot be empty.")

    if not request.user_id.strip():
        raise HTTPException(status_code=400, detail="User ID cannot be empty.")

    if not request.conversation_id.strip():
        raise HTTPException(status_code=400, detail="Conversation ID cannot be empty.")

    # Validate conversation belongs to user
    if not validate_conversation(request.user_id, request.conversation_id):
        raise HTTPException(
            status_code=403,
            detail="Conversation does not belong to this user."
        )

    try:

        result = workflow_app.invoke({
            "symptoms": request.symptoms,
            "user_id": request.user_id,
            "conversation_id": request.conversation_id
        })

        final_advice = result.get("final_advice", "No advice could be generated.")

        return AnalysisResponse(final_advice=final_advice)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating symptoms: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)