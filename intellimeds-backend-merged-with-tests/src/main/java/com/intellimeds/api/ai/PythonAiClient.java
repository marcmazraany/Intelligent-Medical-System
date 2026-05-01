package com.intellimeds.api.ai;

import com.intellimeds.api.ai.dto.PythonAnalyzeRequest;
import com.intellimeds.api.ai.dto.PythonAnalyzeResponse;
import com.intellimeds.api.common.BadRequestException;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

@Component
public class PythonAiClient {

    private final RestTemplate restTemplate;
    private final AiServiceProperties properties;

    public PythonAiClient(RestTemplate restTemplate, AiServiceProperties properties) {
        this.restTemplate = restTemplate;
        this.properties = properties;
    }

    public String analyze(PythonAnalyzeRequest request) {
        String url = properties.baseUrl() + properties.analyzePath();

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<PythonAnalyzeRequest> entity = new HttpEntity<>(request, headers);

        try {
            ResponseEntity<PythonAnalyzeResponse> response =
                    restTemplate.exchange(url, HttpMethod.POST, entity, PythonAnalyzeResponse.class);

            if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
                throw new BadRequestException("AI service returned an invalid response");
            }

            String reply = response.getBody().reply();
            if (reply == null || reply.isBlank()) {
                throw new BadRequestException("AI service returned an empty reply");
            }

            return reply.trim();
        } catch (Exception ex) {
            throw new BadRequestException("Failed to connect to AI service: " + ex.getMessage());
        }
    }
}