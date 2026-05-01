package com.intellimeds.api.medications;

import com.intellimeds.api.common.BadRequestException;
import com.intellimeds.api.medications.dto.MedicationDto;
import com.intellimeds.api.medications.dto.MedicationScanBatchResponse;
import com.intellimeds.api.medications.dto.MedicationScanResponse;
import com.intellimeds.api.medications.dto.MedicationScanSaveItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDate;
import java.util.*;

@Service
public class MedicationScanService {

    private static final Logger log = LoggerFactory.getLogger(MedicationScanService.class);

    private final MedicationScannerClient scannerClient;
    private final MedicationRepository medicationRepository;

    public MedicationScanService(
            MedicationScannerClient scannerClient,
            MedicationRepository medicationRepository
    ) {
        this.scannerClient = scannerClient;
        this.medicationRepository = medicationRepository;
    }

    @Transactional
    public MedicationScanBatchResponse scanBarcode(UUID userId, MultipartFile file) {
        validateFile(file);

        try {
            Map<String, Object> raw = scannerClient.scan(file);
            return processAndSave(userId, raw, "barcode");
        } catch (Exception ex) {
            log.error("Error in scanBarcode: {}", ex.getMessage(), ex);
            throw ex;
        }
    }

    @Transactional
    public MedicationScanBatchResponse scanText(UUID userId, MultipartFile file) {
        validateFile(file);

        try {
            log.info("Starting OCR text scan...");
            Map<String, Object> raw = scannerClient.scanText(file);
            log.info("OCR scan completed, processing results...");

            MedicationScanBatchResponse response = processAndSave(userId, raw, "ocr");
            log.info("OCR processing completed successfully");

            return response;
        } catch (Exception ex) {
            log.error("❌ ERROR in scanText: {}", ex.getMessage(), ex);
            log.error("❌ Exception type: {}", ex.getClass().getName());
            log.error("❌ Stack trace:", ex);
            throw ex;
        }
    }

    private MedicationScanBatchResponse processAndSave(UUID userId, Map<String, Object> raw, String expectedSource) {
        try {
            log.info("Processing medication scan data, source: {}", expectedSource);

            String detectionMethod = value(raw, "detection_method");

            if ("barcode".equals(expectedSource)) {
                log.info("Barcode detection was successful");
            } else {
                log.info("OCR detection was successful");
            }

            List<MedicationScanResponse> parsedItems = normalizeAll(raw);
            log.info("Parsed {} medication items", parsedItems.size());

            if (parsedItems.isEmpty()) {
                throw new BadRequestException("Scanner did not return any valid medications");
            }

            List<MedicationScanSaveItem> savedItems = new ArrayList<>();

            for (MedicationScanResponse parsed : parsedItems) {
                log.info("Processing medication: {}", parsed.name());

                if (parsed.name() == null || parsed.name().isBlank()) {
                    log.warn("Skipping medication with blank name");
                    continue;
                }

                // Handle expiry date - OCR might not have it
                LocalDate expiryDate = null;
                boolean expiryDateMissing = false;

                if (parsed.expiryDate() != null && !parsed.expiryDate().isBlank()) {
                    try {
                        expiryDate = LocalDate.parse(parsed.expiryDate());
                        log.info("Parsed expiry date: {}", expiryDate);
                    } catch (Exception ex) {
                        log.warn("Could not parse expiry date '{}' for medication '{}'",
                                parsed.expiryDate(), parsed.name());
                        expiryDateMissing = true;
                    }
                } else {
                    log.info("No expiry date provided for '{}'", parsed.name());
                    expiryDateMissing = true;
                }

                String dosage = cleanValue(parsed.dosage());
                if (dosage == null || dosage.isBlank() || "nan".equalsIgnoreCase(dosage)) {
                    dosage = "Unknown";
                }
                log.info("Dosage: {}", dosage);

                String frequency = "";

                MedicationEntity existing = null;

                try {
                    if (expiryDate != null) {
                        log.info("Looking for existing medication by name and expiry...");
                        existing = medicationRepository.findByUserIdAndNameIgnoreCaseAndExpiryDate(
                                userId,
                                parsed.name().trim(),
                                expiryDate
                        ).orElse(null);
                    } else {
                        log.info("Looking for existing medication by name only...");
                        List<MedicationEntity> matchingByName = medicationRepository
                                .findByUserIdAndNameIgnoreCase(userId, parsed.name().trim());

                        if (!matchingByName.isEmpty()) {
                            existing = matchingByName.stream()
                                    .max(Comparator.comparing(MedicationEntity::getCreatedAt))
                                    .orElse(null);

                            log.info("Found existing medication '{}' by name only", parsed.name());
                        }
                    }
                } catch (Exception ex) {
                    log.error("Error looking up existing medication: {}", ex.getMessage(), ex);
                    throw new BadRequestException("Database error: " + ex.getMessage());
                }

                MedicationEntity saved;
                String action;

                try {
                    if (existing != null) {
                        log.info("Updating existing medication");
                        MedicationEntity med = existing;
                        med.setQuantity((med.getQuantity() == null ? 1 : med.getQuantity()) + 1);

                        if ((med.getDosage() == null || med.getDosage().isBlank() || med.getDosage().equalsIgnoreCase("Unknown"))
                                && !dosage.equalsIgnoreCase("Unknown")) {
                            med.setDosage(dosage);
                        }

                        if (med.getExpiryDate() == null && expiryDate != null) {
                            med.setExpiryDate(expiryDate);
                        }

                        saved = medicationRepository.save(med);
                        action = "incremented";

                    } else {
                        log.info("Creating new medication");
                        LocalDate finalExpiryDate = expiryDate;
                        MedicationStatus status = MedicationStatus.available;

                        if (expiryDateMissing) {
                            finalExpiryDate = LocalDate.now().plusYears(1);
                            status = MedicationStatus.needs_review;
                            log.info("Expiry date missing for '{}', set to default with needs_review status", parsed.name());
                        }

                        String notes = buildScanNotes(parsed, detectionMethod, expiryDateMissing);
                        log.info("Notes: {}", notes);

                        MedicationEntity med = MedicationEntity.builder()
                                .userId(userId)
                                .name(parsed.name().trim())
                                .dosage(dosage)
                                .expiryDate(finalExpiryDate)
                                .frequency(frequency)
                                .quantity(1)
                                .reminderTimes(List.of())
                                .status(status)
                                .notes(notes)
                                .build();

                        log.info("Saving medication to database...");
                        saved = medicationRepository.save(med);
                        log.info("Medication saved with ID: {}", saved.getId());
                        action = "created";
                    }
                } catch (Exception ex) {
                    log.error("❌ Error saving medication to database: {}", ex.getMessage(), ex);
                    throw new BadRequestException("Failed to save medication: " + ex.getMessage());
                }

                try {
                    log.info("Converting medication to DTO...");
                    MedicationDto dto = toDto(saved);

                    savedItems.add(
                            MedicationScanSaveItem.builder()
                                    .action(action)
                                    .medication(dto)
                                    .scan(parsed)
                                    .build()
                    );
                    log.info("Medication item added to saved items");
                } catch (Exception ex) {
                    log.error("❌ Error creating saved item: {}", ex.getMessage(), ex);
                    throw new BadRequestException("Failed to create response: " + ex.getMessage());
                }
            }

            if (savedItems.isEmpty()) {
                throw new BadRequestException("Scanner returned results, but none could be saved");
            }

            Integer detectedCount = asInteger(raw.get("detected_count"));
            if (detectedCount == null) {
                detectedCount = parsedItems.size();
            }

            String source = firstNonBlank(
                    value(raw, "source"),
                    value(raw, "detection_method"),
                    "unknown"
            );

            log.info("Building final response with {} items", savedItems.size());

            return MedicationScanBatchResponse.builder()
                    .source(source)
                    .detectedCount(detectedCount)
                    .items(savedItems)
                    .raw(new LinkedHashMap<>(raw))
                    .build();

        } catch (Exception ex) {
            log.error("❌ Error in processAndSave: {}", ex.getMessage(), ex);
            throw ex;
        }
    }

    private List<MedicationScanResponse> normalizeAll(Map<String, Object> raw) {
        List<Map<String, Object>> entries = extractDrugDetailsList(raw);
        List<MedicationScanResponse> result = new ArrayList<>();

        for (Map<String, Object> item : entries) {
            try {
                MedicationScanResponse parsed = MedicationScanResponse.builder()
                        .source(firstNonBlank(
                                value(raw, "source"),
                                value(raw, "detection_method"),
                                inferSource(raw)
                        ))
                        .gtin(firstNonBlank(
                                value(item, "GTIN"),
                                value(item, "gtin")
                        ))
                        .name(firstNonBlank(
                                value(item, "Brand Name"),
                                value(item, "Brand name"),
                                value(item, "brand_name"),
                                value(item, "name")
                        ))
                        .manufacturer(firstNonBlank(
                                value(item, "Manufacturer"),
                                value(item, "manufacturer")
                        ))
                        .dosage(firstNonBlank(
                                value(item, "Dosage"),
                                value(item, "dosage"),
                                value(item, "Strength"),
                                value(item, "strength")
                        ))
                        .quantity(firstNonBlank(
                                value(item, "Quantity"),
                                value(item, "quantity"),
                                value(item, "Presentation")
                        ))
                        .form(firstNonBlank(
                                value(item, "Form"),
                                value(item, "form")
                        ))
                        .expiryDate(firstNonBlank(
                                value(item, "Expiry Date"),
                                value(item, "expiry_date"),
                                value(item, "expiry")
                        ))
                        .raw(new LinkedHashMap<>(item))
                        .build();

                if (parsed.name() != null && !parsed.name().isBlank()) {
                    result.add(parsed);
                }
            } catch (Exception ex) {
                log.error("Error parsing medication item: {}", ex.getMessage(), ex);
                // Continue with other items
            }
        }

        return result;
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> extractDrugDetailsList(Map<String, Object> raw) {
        Object drugDetails = raw.get("Drug Details");

        if (drugDetails instanceof List<?> list) {
            List<Map<String, Object>> out = new ArrayList<>();
            for (Object obj : list) {
                if (obj instanceof Map<?, ?> map) {
                    out.add((Map<String, Object>) map);
                }
            }
            return out;
        }

        if (drugDetails instanceof Map<?, ?> map) {
            return List.of((Map<String, Object>) map);
        }

        return List.of(raw);
    }

    private void validateFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BadRequestException("file is required");
        }

        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            throw new BadRequestException("Only image files are supported");
        }
    }

    private String buildScanNotes(MedicationScanResponse parsed, String detectionMethod, boolean expiryDateMissing) {
        StringBuilder sb = new StringBuilder();

        if ("ocr".equalsIgnoreCase(detectionMethod) || "text_detection".equalsIgnoreCase(parsed.source())) {
            sb.append("Added by OCR scan");
        } else {
            sb.append("Added by barcode scan");
        }

        if (expiryDateMissing) {
            sb.append(" | ⚠️ Expiry date not detected - please verify and update");
        }

        // SAFE null checks for all fields
        if (parsed.gtin() != null && !parsed.gtin().isBlank() && !"null".equalsIgnoreCase(parsed.gtin())) {
            sb.append(" | GTIN: ").append(parsed.gtin());
        }
        if (parsed.manufacturer() != null && !parsed.manufacturer().isBlank() && !"null".equalsIgnoreCase(parsed.manufacturer())) {
            sb.append(" | Manufacturer: ").append(parsed.manufacturer());
        }
        if (parsed.form() != null && !parsed.form().isBlank() && !"null".equalsIgnoreCase(parsed.form())) {
            sb.append(" | Form: ").append(parsed.form());
        }

        return sb.toString();
    }

    private String value(Map<String, Object> raw, String key) {
        if (raw == null || key == null) return null;
        Object v = raw.get(key);
        return cleanValue(v == null ? null : String.valueOf(v));
    }

    private String cleanValue(String value) {
        if (value == null) return null;

        String trimmed = value.trim();

        if (trimmed.isEmpty()
                || "null".equalsIgnoreCase(trimmed)
                || "nan".equalsIgnoreCase(trimmed)
                || "none".equalsIgnoreCase(trimmed)
                || "undefined".equalsIgnoreCase(trimmed)) {
            return null;
        }

        return trimmed;
    }

    private String firstNonBlank(String... values) {
        if (values == null) return null;

        for (String value : values) {
            String cleaned = cleanValue(value);
            if (cleaned != null && !cleaned.isBlank()) {
                return cleaned;
            }
        }
        return null;
    }

    private String inferSource(Map<String, Object> raw) {
        if (raw == null) return "unknown";

        if (raw.containsKey("detected_texts")) {
            return "text_detection";
        }
        if (raw.containsKey("GTIN") || raw.containsKey("gtin")) {
            return "barcode";
        }
        return "unknown";
    }

    private Integer asInteger(Object value) {
        try {
            if (value == null) return null;
            String str = String.valueOf(value).trim();
            if ("nan".equalsIgnoreCase(str) || "null".equalsIgnoreCase(str)) {
                return null;
            }
            return Integer.valueOf(str);
        } catch (Exception ex) {
            log.warn("Could not convert value to integer: {}", value);
            return null;
        }
    }

    private MedicationDto toDto(MedicationEntity e) {
        if (e == null) {
            throw new IllegalArgumentException("Cannot convert null entity to DTO");
        }

        return MedicationDto.builder()
                .id(e.getId() != null ? e.getId().toString() : null)
                .name(e.getName())
                .dosage(e.getDosage())
                .expiryDate(e.getExpiryDate())
                .frequency(e.getFrequency())
                .quantity(e.getQuantity())
                .reminderTimes(e.getReminderTimes() != null ? e.getReminderTimes() : List.of())
                .status(e.getStatus() != null ? e.getStatus().name() : MedicationStatus.available.name())
                .notes(e.getNotes())
                .build();
    }
}