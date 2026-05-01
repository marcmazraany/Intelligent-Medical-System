package com.intellimeds.api.ai;

import com.intellimeds.api.ai.dto.*;
import com.intellimeds.api.common.BadRequestException;
import com.intellimeds.api.common.NotFoundException;
import com.intellimeds.api.medications.MedicationEntity;
import com.intellimeds.api.medications.MedicationRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

@Service
public class AiChatService {

    private final AiConversationRepository conversationRepository;
    private final AiConversationService conversationService;
    private final MedicationRepository medicationRepository;
    private final PythonAiClient pythonAiClient;

    public AiChatService(
            AiConversationRepository conversationRepository,
            AiConversationService conversationService,
            MedicationRepository medicationRepository,
            PythonAiClient pythonAiClient
    ) {
        this.conversationRepository = conversationRepository;
        this.conversationService = conversationService;
        this.medicationRepository = medicationRepository;
        this.pythonAiClient = pythonAiClient;
    }

    @Transactional
    public AiAnalyzeResponse analyze(UUID userId, AiAnalyzeRequest request) {
        if (request.message() == null || request.message().isBlank()) {
            throw new BadRequestException("message is required");
        }

        UUID conversationId = resolveConversationId(userId, request);

        conversationService.addMessage(userId, conversationId, "user", request.message().trim());

        List<AiHistoryItem> history = conversationService.getLastMessages(userId, conversationId, 6)
                .stream()
                .map(msg -> new AiHistoryItem(msg.role(), msg.content()))
                .toList();

        List<AiMedicationItem> medications = medicationRepository.findAllByUserIdOrderByCreatedAtDesc(userId)
                .stream()
                .map(this::toMedicationItem)
                .toList();

        PythonAnalyzeRequest pythonRequest = new PythonAnalyzeRequest(
                request.message().trim(),
                medications,
                history
        );

        String reply = pythonAiClient.analyze(pythonRequest);

        conversationService.addMessage(userId, conversationId, "assistant", reply);

        return AiAnalyzeResponse.builder()
                .conversationId(conversationId.toString())
                .reply(reply)
                .build();
    }

    private UUID resolveConversationId(UUID userId, AiAnalyzeRequest request) {
        if (request.conversationId() == null || request.conversationId().isBlank()) {
            AiConversationDto created = conversationService.create(
                    userId,
                    new CreateAiConversationRequest("New session conversation")
            );
            return UUID.fromString(created.id());
        }

        UUID conversationId;
        try {
            conversationId = UUID.fromString(request.conversationId());
        } catch (IllegalArgumentException ex) {
            throw new BadRequestException("conversationId is invalid");
        }

        conversationRepository.findByIdAndUserId(conversationId, userId)
                .orElseThrow(() -> new NotFoundException("AI conversation not found"));

        return conversationId;
    }

    private AiMedicationItem toMedicationItem(MedicationEntity med) {
        return new AiMedicationItem(
                med.getName(),
                med.getDosage(),
                null,
                med.getFrequency()
        );
    }
}