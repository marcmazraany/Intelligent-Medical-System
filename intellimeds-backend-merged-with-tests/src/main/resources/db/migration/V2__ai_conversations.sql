CREATE TABLE ai_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_ai_conversations_user_id ON ai_conversations(user_id);
CREATE INDEX idx_ai_conversations_updated_at ON ai_conversations(updated_at DESC);

CREATE TABLE ai_conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES ai_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_ai_message_role CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE INDEX idx_ai_conversation_messages_conversation_id
    ON ai_conversation_messages(conversation_id);

CREATE INDEX idx_ai_conversation_messages_created_at
    ON ai_conversation_messages(created_at ASC);

CREATE OR REPLACE FUNCTION set_ai_conversations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ai_conversations_updated_at
BEFORE UPDATE ON ai_conversations
FOR EACH ROW
EXECUTE FUNCTION set_ai_conversations_updated_at();