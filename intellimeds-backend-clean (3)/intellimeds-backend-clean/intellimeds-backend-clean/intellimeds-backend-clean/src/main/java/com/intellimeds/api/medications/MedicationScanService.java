package com.intellimeds.api.medications;

import com.intellimeds.api.common.BadRequestException;
import com.intellimeds.api.medications.dto.MedicationDto;
import com.intellimeds.api.medications.dto.MedicationScanBatchResponse;
import com.intellimeds.api.medications.dto.MedicationScanResponse;
import com.intellimeds.api.medications.dto.MedicationScanSaveItem;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDate;
import java.util.*;

@Service
public class MedicationScanService {

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
    public MedicationScanBatchResponse scanAndSave(UUID userId, MultipartFile file) {
        validateFile(file);

        Map<String, Object> raw = scannerClient.scan(file);
        List<MedicationScanResponse> parsedItems = normalizeAll(raw);

        if (parsedItems.isEmpty()) {
            throw new BadRequestException("Scanner did not return any valid medications");
        }

        List<MedicationScanSaveItem> savedItems = new ArrayList<>();

        for (MedicationScanResponse parsed : parsedItems) {
            if (parsed.name() == null || parsed.name().isBlank()) {
                continue;
            }

            if (parsed.expiryDate() == null || parsed.expiryDate().isBlank()) {
                continue;
            }

            LocalDate expiryDate;
            try {
                expiryDate = LocalDate.parse(parsed.expiryDate());
            } catch (Exception ex) {
                continue;
            }

            String dosage = parsed.dosage() != null && !parsed.dosage().isBlank()
                    ? parsed.dosage()
                    : "Unknown";

            String frequency = "";

            var existing = medicationRepository.findByUserIdAndNameIgnoreCaseAndExpiryDate(
                    userId,
                    parsed.name().trim(),
                    expiryDate
            );

            MedicationEntity saved;
            String action;

            if (existing.isPresent()) {
                MedicationEntity med = existing.get();
                med.setQuantity((med.getQuantity() == null ? 1 : med.getQuantity()) + 1);

                if ((med.getDosage() == null || med.getDosage().isBlank() || med.getDosage().equalsIgnoreCase("Unknown"))
                        && !dosage.equalsIgnoreCase("Unknown")) {
                    med.setDosage(dosage);
                }

                saved = medicationRepository.save(med);
                action = "incremented";
            } else {
                MedicationEntity med = MedicationEntity.builder()
                        .userId(userId)
                        .name(parsed.name().trim())
                        .dosage(dosage)
                        .expiryDate(expiryDate)
                        .frequency(frequency)
                        .quantity(1)
                        .reminderTimes(List.of())
                        .status(MedicationStatus.available)
                        .notes(buildScanNotes(parsed))
                        .build();

                saved = medicationRepository.save(med);
                action = "created";
            }

            savedItems.add(
                    MedicationScanSaveItem.builder()
                            .action(action)
                            .medication(toDto(saved))
                            .scan(parsed)
                            .build()
            );
        }

        if (savedItems.isEmpty()) {
            throw new BadRequestException("Scanner returned results, but none could be saved");
        }

        Integer detectedCount = asInteger(raw.get("detected_count"));
        if (detectedCount == null) {
            detectedCount = parsedItems.size();
        }

        return MedicationScanBatchResponse.builder()
                .source(firstNonBlank(value(raw, "source"), "barcode"))
                .detectedCount(detectedCount)
                .items(savedItems)
                .raw(new LinkedHashMap<>(raw))
                .build();
    }

    private List<MedicationScanResponse> normalizeAll(Map<String, Object> raw) {
        List<Map<String, Object>> entries = extractDrugDetailsList(raw);
        List<MedicationScanResponse> result = new ArrayList<>();

        for (Map<String, Object> item : entries) {
            MedicationScanResponse parsed = MedicationScanResponse.builder()
                    .source(firstNonBlank(
                            value(raw, "source"),
                            inferSource(raw)
                    ))
                    .gtin(firstNonBlank(
                            value(item, "GTIN"),
                            value(item, "gtin")
                    ))
                    .name(firstNonBlank(
                            value(item, "Brand Name"),
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
                            value(item, "strength")
                    ))
                    .quantity(firstNonBlank(
                            value(item, "Quantity"),
                            value(item, "quantity")
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

    private String buildScanNotes(MedicationScanResponse parsed) {
        StringBuilder sb = new StringBuilder("Added by barcode scan");
        if (parsed.gtin() != null && !parsed.gtin().isBlank()) {
            sb.append(" | GTIN: ").append(parsed.gtin());
        }
        if (parsed.manufacturer() != null && !parsed.manufacturer().isBlank()) {
            sb.append(" | Manufacturer: ").append(parsed.manufacturer());
        }
        if (parsed.form() != null && !parsed.form().isBlank()) {
            sb.append(" | Form: ").append(parsed.form());
        }
        return sb.toString();
    }

    private String value(Map<String, Object> raw, String key) {
        Object v = raw.get(key);
        return v == null ? null : String.valueOf(v).trim();
    }

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank() && !"null".equalsIgnoreCase(value)) {
                return value;
            }
        }
        return null;
    }

    private String inferSource(Map<String, Object> raw) {
        if (raw.containsKey("GTIN") || raw.containsKey("gtin") || raw.containsKey("Drug Details")) {
            return "barcode";
        }
        return "unknown";
    }

    private Integer asInteger(Object value) {
        try {
            if (value == null) return null;
            return Integer.valueOf(String.valueOf(value));
        } catch (Exception ex) {
            return null;
        }
    }

    private MedicationDto toDto(MedicationEntity e) {
        return MedicationDto.builder()
                .id(e.getId().toString())
                .name(e.getName())
                .dosage(e.getDosage())
                .expiryDate(e.getExpiryDate())
                .frequency(e.getFrequency())
                .quantity(e.getQuantity())
                .reminderTimes(e.getReminderTimes())
                .status(e.getStatus().name())
                .notes(e.getNotes())
                .build();
    }
}