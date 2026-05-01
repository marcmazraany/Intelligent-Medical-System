package com.intellimeds.api.ai;

import com.intellimeds.api.ai.dto.AiConversationDto;
import com.intellimeds.api.ai.dto.AiMessageDto;
import com.intellimeds.api.ai.dto.CreateAiConversationRequest;
import com.intellimeds.api.common.NotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;

@Service
public class AiConversationService {

    private final AiConversationRepository conversationRepository;
    private final AiConversationMessageRepository messageRepository;

    public AiConversationService(
            AiConversationRepository conversationRepository,
            AiConversationMessageRepository messageRepository
    ) {
        this.conversationRepository = conversationRepository;
        this.messageRepository = messageRepository;
    }

    @Transactional(readOnly = true)
    public List<AiConversationDto> list(UUID userId) {
        return conversationRepository.findAllByUserIdOrderByUpdatedAtDesc(userId)
                .stream()
                .map(this::toConversationDto)
                .toList();
    }

    @Transactional
    public AiConversationDto create(UUID userId, CreateAiConversationRequest request) {
        String title = (request != null && request.title() != null && !request.title().isBlank())
                ? request.title().trim()
                : "New conversation";

        AiConversationEntity saved = conversationRepository.save(
                AiConversationEntity.builder()
                        .userId(userId)
                        .title(title)
                        .build()
        );

        return toConversationDto(saved);
    }

    @Transactional(readOnly = true)
    public List<AiMessageDto> getMessages(UUID userId, UUID conversationId) {
        ensureConversationOwnedByUser(userId, conversationId);

        return messageRepository.findAllByConversationIdOrderByCreatedAtAsc(conversationId)
                .stream()
                .map(this::toMessageDto)
                .toList();
    }

    @Transactional(readOnly = true)
    public List<AiMessageDto> getLastMessages(UUID userId, UUID conversationId, int limit) {
        ensureConversationOwnedByUser(userId, conversationId);

        List<AiConversationMessageEntity> latest =
                messageRepository.findAllByConversationIdOrderByCreatedAtAsc(conversationId);

        int safeLimit = Math.max(limit, 1);
        int fromIndex = Math.max(latest.size() - safeLimit, 0);

        return latest.subList(fromIndex, latest.size())
                .stream()
                .map(this::toMessageDto)
                .toList();
    }

    @Transactional
    public AiMessageDto addMessage(UUID userId, UUID conversationId, String role, String content) {
        AiConversationEntity conversation = ensureConversationOwnedByUser(userId, conversationId);

        AiConversationMessageEntity saved = messageRepository.save(
                AiConversationMessageEntity.builder()
                        .conversationId(conversation.getId())
                        .role(role)
                        .content(content)
                        .build()
        );

        conversation.setTitle(
                conversation.getTitle() == null || conversation.getTitle().isBlank()
                        ? "New conversation"
                        : conversation.getTitle()
        );
        conversationRepository.save(conversation);

        return toMessageDto(saved);
    }

    @Transactional
    public void delete(UUID userId, UUID conversationId) {
        ensureConversationOwnedByUser(userId, conversationId);
        conversationRepository.deleteByIdAndUserId(conversationId, userId);
    }

    private AiConversationEntity ensureConversationOwnedByUser(UUID userId, UUID conversationId) {
        return conversationRepository.findByIdAndUserId(conversationId, userId)
                .orElseThrow(() -> new NotFoundException("AI conversation not found"));
    }

    private AiConversationDto toConversationDto(AiConversationEntity e) {
        return AiConversationDto.builder()
                .id(e.getId().toString())
                .title(e.getTitle())
                .createdAt(e.getCreatedAt())
                .updatedAt(e.getUpdatedAt())
                .build();
    }

    private AiMessageDto toMessageDto(AiConversationMessageEntity e) {
        return AiMessageDto.builder()
                .id(e.getId().toString())
                .role(e.getRole())
                .content(e.getContent())
                .createdAt(e.getCreatedAt())
                .build();
    }
}