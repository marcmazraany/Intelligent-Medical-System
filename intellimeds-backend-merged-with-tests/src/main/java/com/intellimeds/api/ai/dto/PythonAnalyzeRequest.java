package com.intellimeds.api.ai.dto;

import java.util.List;

public record PythonAnalyzeRequest(
        String symptoms,
        List<AiMedicationItem> medications,
        List<AiHistoryItem> history
) {}