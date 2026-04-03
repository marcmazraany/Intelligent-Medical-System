package com.intellimeds.api.ai;

import com.intellimeds.api.ai.dto.*;
import com.intellimeds.api.security.AuthUser;
import jakarta.validation.Valid;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/ai")
public class AiConversationController {

    private final AiConversationService conversationService;
    private final AiChatService aiChatService;

    public AiConversationController(
            AiConversationService conversationService,
            AiChatService aiChatService
    ) {
        this.conversationService = conversationService;
        this.aiChatService = aiChatService;
    }

    @GetMapping("/conversations")
    public List<AiConversationDto> list(@AuthenticationPrincipal AuthUser user) {
        return conversationService.list(user.userId());
    }

    @PostMapping("/conversations")
    public AiConversationDto create(
            @AuthenticationPrincipal AuthUser user,
            @RequestBody(required = false) CreateAiConversationRequest request
    ) {
        return conversationService.create(user.userId(), request);
    }

    @GetMapping("/conversations/{conversationId}/messages")
    public List<AiMessageDto> messages(
            @AuthenticationPrincipal AuthUser user,
            @PathVariable UUID conversationId
    ) {
        return conversationService.getMessages(user.userId(), conversationId);
    }

    @DeleteMapping("/conversations/{conversationId}")
    public Map<String, String> delete(
            @AuthenticationPrincipal AuthUser user,
            @PathVariable UUID conversationId
    ) {
        conversationService.delete(user.userId(), conversationId);
        return Map.of("status", "deleted");
    }

    @PostMapping("/analyze")
    public AiAnalyzeResponse analyze(
            @AuthenticationPrincipal AuthUser user,
            @Valid @RequestBody AiAnalyzeRequest request
    ) {
        return aiChatService.analyze(user.userId(), request);
    }
}