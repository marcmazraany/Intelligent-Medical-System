package com.intellimeds.api.ai;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface AiConversationRepository extends JpaRepository<AiConversationEntity, UUID> {
    List<AiConversationEntity> findAllByUserIdOrderByUpdatedAtDesc(UUID userId);
    Optional<AiConversationEntity> findByIdAndUserId(UUID id, UUID userId);
    void deleteByIdAndUserId(UUID id, UUID userId);
}