import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ===============================
# Get user medications
# ===============================
def get_user_medications(user_id: str):

    response = (
        supabase
        .table("medications")
        .select("name")
        .eq("user_id", user_id)
        .execute()
    )

    if not response.data:
        return []

    return [m["name"] for m in response.data]


# ===============================
# Check conversation ownership
# ===============================
def validate_conversation(user_id: str, conversation_id: int):

    response = (
        supabase
        .table("conversations")
        .select("conv_id")
        .eq("conv_id", conversation_id)
        .eq("user_id", user_id)
        .execute()
    )

    return bool(response.data)


# ===============================
# Get conversation history
# ===============================
def get_conversation_history(conversation_id: int):

    response = (
        supabase
        .table("Conversation_Memory")
        .select("role,content")
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
    )

    if not response.data:
        return []

    return response.data


# ===============================
# Save message
# ===============================
def save_message(conversation_id: int, role: str, content: str):

    supabase.table("Conversation_Memory").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content
    }).execute()