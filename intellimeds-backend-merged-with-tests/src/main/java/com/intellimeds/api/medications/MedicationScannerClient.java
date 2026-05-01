package com.intellimeds.api.medications;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.intellimeds.api.common.BadRequestException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class MedicationScannerClient {

    private static final Logger log = LoggerFactory.getLogger(MedicationScannerClient.class);

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

    /**
     * Scans medication image with automatic barcode → OCR fallback.
     *
     * NEW: Returns special error when barcode fails so frontend can prompt
     * user to take a new photo of the medication NAME (not barcode).
     */
    public Map<String, Object> scan(MultipartFile file) {
        byte[] imageBytes;
        try {
            imageBytes = file.getBytes();
        } catch (Exception ex) {
            throw new BadRequestException("Failed to read image file: " + ex.getMessage());
        }

        // Step 1: Try Barcode Detection First (Port 8000)
        log.info("Attempting barcode detection at {}...", properties.barcodeUrl());
        boolean barcodeAttempted = true;

        try {
            Map<String, Object> barcodeResult = callApi(
                    properties.barcodeUrl(),
                    imageBytes,
                    file.getOriginalFilename(),
                    file.getContentType()
            );

            // Barcode succeeded
            log.info("✅ Barcode detection successful");

            // Add metadata to indicate no fallback was used
            barcodeResult.put("fallback_used", false);
            barcodeResult.put("detection_method", "barcode");

            return barcodeResult;

        } catch (HttpStatusCodeException ex) {
            // Barcode API returned an error (404 = not found, 500 = server error)
            log.warn("Barcode detection failed with status {}: {}", ex.getStatusCode(), ex.getMessage());

            // NEW: Instead of auto-falling back to OCR on the same image,
            // return an error that tells the frontend to ask user for a TEXT photo
            if (ex.getStatusCode() == HttpStatus.NOT_FOUND) {
                log.info("Barcode not recognized in database. Frontend should prompt user for medication text photo.");

                // Return a special response that frontend can detect
                Map<String, Object> errorResponse = new LinkedHashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("error_type", "barcode_not_found");
                errorResponse.put("message", "Barcode not found in database");
                errorResponse.put("user_action_required", true);
                errorResponse.put("user_message", "Barcode not recognized. Please take a clear photo of the medication name and details.");
                errorResponse.put("next_step", "request_text_photo");

                throw new BarcodeNotFoundException(errorResponse);
            }

        } catch (Exception ex) {
            // Other barcode errors (network, timeout, etc.)
            log.warn("Barcode detection failed: {}", ex.getMessage());
        }

        // Step 2: If we reach here without throwing BarcodeNotFoundException,
        // it means barcode detection had a technical error (not "not found")
        // In this case, we DON'T want OCR fallback automatically

        log.error("Barcode detection failed due to technical error, not trying OCR");
        throw new BadRequestException(
                "Could not scan barcode. Please ensure the barcode is clearly visible and try again."
        );
    }

    /**
     * NEW: Separate method for OCR-only scanning (called when user takes text photo)
     */
    public Map<String, Object> scanText(MultipartFile file) {
        byte[] imageBytes;
        try {
            imageBytes = file.getBytes();
        } catch (Exception ex) {
            throw new BadRequestException("Failed to read image file: " + ex.getMessage());
        }

        log.info("OCR text detection at {}...", properties.ocrUrl());

        try {
            Map<String, Object> ocrResult = callApi(
                    properties.ocrUrl(),
                    imageBytes,
                    file.getOriginalFilename(),
                    file.getContentType()
            );

            // OCR succeeded
            log.info("✅ OCR detection successful");

            // Add metadata
            ocrResult.put("fallback_used", false);  // Not fallback, intentional OCR scan
            ocrResult.put("detection_method", "ocr");

            return ocrResult;

        } catch (Exception ex) {
            log.error("OCR detection failed: {}", ex.getMessage());
            throw new BadRequestException(
                    "Could not detect medication text. Please ensure the medication name is clearly visible and try again."
            );
        }
    }

    /**
     * Generic method to call either barcode or OCR API
     */
    private Map<String, Object> callApi(String url, byte[] imageBytes, String filename, String contentType) throws Exception {
        ByteArrayResource resource = new ByteArrayResource(imageBytes) {
            @Override
            public String getFilename() {
                return filename != null ? filename : "upload.jpg";
            }
        };

        HttpHeaders fileHeaders = new HttpHeaders();
        MediaType mediaType = contentType != null
                ? MediaType.parseMediaType(contentType)
                : MediaType.APPLICATION_OCTET_STREAM;
        fileHeaders.setContentType(mediaType);

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
            throw new BadRequestException("Detection service returned an invalid response");
        }

        return objectMapper.readValue(response.getBody(), new TypeReference<>() {});
    }

    /**
     * Custom exception for barcode not found - allows special handling
     */
    public static class BarcodeNotFoundException extends RuntimeException {
        private final Map<String, Object> response;

        public BarcodeNotFoundException(Map<String, Object> response) {
            super((String) response.get("message"));
            this.response = response;
        }

        public Map<String, Object> getResponse() {
            return response;
        }
    }
}