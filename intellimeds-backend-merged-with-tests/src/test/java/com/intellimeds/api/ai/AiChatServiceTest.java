package com.intellimeds.api.ai;

import com.intellimeds.api.ai.dto.*;
import com.intellimeds.api.medications.MedicationEntity;
import com.intellimeds.api.medications.MedicationRepository;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("AI Chat – AiChatService Unit Tests")
class AiChatServiceTest {

    @Mock private AiConversationRepository conversationRepository;
    @Mock private AiConversationService conversationService;
    @Mock private MedicationRepository medicationRepository;
    @Mock private PythonAiClient pythonAiClient;

    @InjectMocks
    private AiChatService aiChatService;

    private static final UUID USER_ID = UUID.fromString("11111111-1111-1111-1111-111111111111");
    private static final UUID CONVERSATION_ID = UUID.fromString("22222222-2222-2222-2222-222222222222");

    @Nested
    @DisplayName("analyze")
    class Analyze {
        @Test
        @DisplayName("auto-creates a conversation when no conversationId is provided")
        void analyze_autoCreatesConversationWhenConversationIdMissing() {
            when(conversationService.create(eq(USER_ID), any(CreateAiConversationRequest.class)))
                    .thenReturn(AiConversationDto.builder()
                            .id(CONVERSATION_ID.toString())
                            .title("New session conversation")
                            .createdAt(Instant.now())
                            .updatedAt(Instant.now())
                            .build());
            when(conversationService.getLastMessages(USER_ID, CONVERSATION_ID, 6)).thenReturn(List.of());
            when(medicationRepository.findAllByUserIdOrderByCreatedAtDesc(USER_ID)).thenReturn(List.of());
            when(pythonAiClient.analyze(any(PythonAnalyzeRequest.class))).thenReturn("Take paracetamol if appropriate.");

            AiAnalyzeResponse result = aiChatService.analyze(
                    USER_ID,
                    new AiAnalyzeRequest("I have a headache", null)
            );

            assertThat(result.conversationId()).isEqualTo(CONVERSATION_ID.toString());
            assertThat(result.reply()).isEqualTo("Take paracetamol if appropriate.");
            verify(conversationService).create(eq(USER_ID), any(CreateAiConversationRequest.class));
            verify(conversationService, times(2)).addMessage(eq(USER_ID), eq(CONVERSATION_ID), anyString(), anyString());
        }

        @Test
        @DisplayName("uses an existing conversationId when it belongs to the user")
        void analyze_usesExistingConversationId() {
            when(conversationRepository.findByIdAndUserId(CONVERSATION_ID, USER_ID))
                    .thenReturn(Optional.of(AiConversationEntity.builder()
                            .id(CONVERSATION_ID)
                            .userId(USER_ID)
                            .title("Existing")
                            .build()));
            when(conversationService.getLastMessages(USER_ID, CONVERSATION_ID, 6)).thenReturn(List.of());
            when(medicationRepository.findAllByUserIdOrderByCreatedAtDesc(USER_ID)).thenReturn(List.of());
            when(pythonAiClient.analyze(any(PythonAnalyzeRequest.class))).thenReturn("Advice.");

            aiChatService.analyze(USER_ID, new AiAnalyzeRequest("Can I take this?", CONVERSATION_ID.toString()));

            verify(conversationService, never()).create(any(), any());
            verify(conversationRepository).findByIdAndUserId(CONVERSATION_ID, USER_ID);
        }

        @Test
        @DisplayName("injects user medications and conversation history into the Python request")
        void analyze_injectsMedicationAndHistoryContext() {
            MedicationEntity med = new MedicationEntity();
            med.setName("Panadol");
            med.setDosage("500mg");
            med.setFrequency("twice daily");

            AiMessageDto previousMessage = AiMessageDto.builder()
                    .id(UUID.randomUUID().toString())
                    .role("assistant")
                    .content("Previous advice")
                    .createdAt(Instant.now())
                    .build();

            when(conversationRepository.findByIdAndUserId(CONVERSATION_ID, USER_ID))
                    .thenReturn(Optional.of(AiConversationEntity.builder().id(CONVERSATION_ID).userId(USER_ID).build()));
            when(conversationService.getLastMessages(USER_ID, CONVERSATION_ID, 6)).thenReturn(List.of(previousMessage));
            when(medicationRepository.findAllByUserIdOrderByCreatedAtDesc(USER_ID)).thenReturn(List.of(med));

            ArgumentCaptor<PythonAnalyzeRequest> requestCaptor = ArgumentCaptor.forClass(PythonAnalyzeRequest.class);
            when(pythonAiClient.analyze(requestCaptor.capture())).thenReturn("Context-aware reply");

            aiChatService.analyze(USER_ID, new AiAnalyzeRequest("What can I take?", CONVERSATION_ID.toString()));

            PythonAnalyzeRequest sent = requestCaptor.getValue();
            assertThat(sent.symptoms()).isEqualTo("What can I take?");
            assertThat(sent.medications())
                    .singleElement()
                    .satisfies(item -> {
                        assertThat(item.name()).isEqualTo("Panadol");
                        assertThat(item.dosage()).isEqualTo("500mg");
                        assertThat(item.frequency()).isEqualTo("twice daily");
                    });
            assertThat(sent.history())
                    .singleElement()
                    .satisfies(item -> {
                        assertThat(item.role()).isEqualTo("assistant");
                        assertThat(item.content()).isEqualTo("Previous advice");
                    });
        }

        @Test
        @DisplayName("stores the user message before calling AI and stores the assistant reply after")
        void analyze_persistsBothMessages() {
            when(conversationRepository.findByIdAndUserId(CONVERSATION_ID, USER_ID))
                    .thenReturn(Optional.of(AiConversationEntity.builder().id(CONVERSATION_ID).userId(USER_ID).build()));
            when(conversationService.getLastMessages(USER_ID, CONVERSATION_ID, 6)).thenReturn(List.of());
            when(medicationRepository.findAllByUserIdOrderByCreatedAtDesc(USER_ID)).thenReturn(List.of());
            when(pythonAiClient.analyze(any(PythonAnalyzeRequest.class))).thenReturn("Here is advice.");

            aiChatService.analyze(USER_ID, new AiAnalyzeRequest("Need advice", CONVERSATION_ID.toString()));

            verify(conversationService).addMessage(USER_ID, CONVERSATION_ID, "user", "Need advice");
            verify(conversationService).addMessage(USER_ID, CONVERSATION_ID, "assistant", "Here is advice.");
        }
    }
}
