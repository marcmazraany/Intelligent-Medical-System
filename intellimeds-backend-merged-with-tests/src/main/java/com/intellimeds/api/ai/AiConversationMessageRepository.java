package com.intellimeds.api.ai;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

public interface AiConversationMessageRepository extends JpaRepository<AiConversationMessageEntity, UUID> {
    List<AiConversationMessageEntity> findAllByConversationIdOrderByCreatedAtAsc(UUID conversationId);
    List<AiConversationMessageEntity> findTop6ByConversationIdOrderByCreatedAtDesc(UUID conversationId);
}