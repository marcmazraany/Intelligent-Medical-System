package com.intellimeds.api.medications;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.intellimeds.api.common.BadRequestException;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Component
public class MedicationScannerClient {

    private final RestTemplate restTemplate;
    private final MedicationScannerProperties properties;
    private final ObjectMapper objectMapper;

    public MedicationScannerClient(
            RestTemplate restTemplate,
            MedicationScannerProperties properties,
            ObjectMapper objectMapper
    ) {
        this.restTemplate = restTemplate;
        this.properties = properties;
        this.objectMapper = objectMapper;
    }

    public Map<String, Object> scan(MultipartFile file) {
        String url = properties.baseUrl() + properties.scanPath();

        try {
            ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename() != null ? file.getOriginalFilename() : "upload.jpg";
                }
            };

            HttpHeaders fileHeaders = new HttpHeaders();
            MediaType contentType = file.getContentType() != null
                    ? MediaType.parseMediaType(file.getContentType())
                    : MediaType.APPLICATION_OCTET_STREAM;
            fileHeaders.setContentType(contentType);

            HttpEntity<ByteArrayResource> fileEntity = new HttpEntity<>(resource, fileHeaders);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", fileEntity);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<MultiValueMap<String, Object>> requestEntity =
                    new HttpEntity<>(body, headers);

            ResponseEntity<String> response =
                    restTemplate.exchange(url, HttpMethod.POST, requestEntity, String.class);

            if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
                throw new BadRequestException("Scanner service returned an invalid response");
            }

            return objectMapper.readValue(response.getBody(), new TypeReference<>() {});
        } catch (Exception ex) {
            throw new BadRequestException("Failed to scan medication image: " + ex.getMessage());
        }
    }
}