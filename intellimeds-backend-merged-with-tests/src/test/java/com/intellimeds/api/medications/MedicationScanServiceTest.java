package com.intellimeds.api.medications;

import com.intellimeds.api.common.BadRequestException;
import com.intellimeds.api.medications.dto.MedicationScanBatchResponse;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.time.Instant;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("Medication Scanning – MedicationScanService Unit Tests")
class MedicationScanServiceTest {

    @Mock private MedicationScannerClient scannerClient;
    @Mock private MedicationRepository medicationRepository;

    @InjectMocks
    private MedicationScanService scanService;

    private static final UUID USER_ID = UUID.fromString("11111111-1111-1111-1111-111111111111");

    private MultipartFile imageFile() {
        return new MockMultipartFile("file", "med.jpg", "image/jpeg", "fake-image".getBytes());
    }

    private Map<String, Object> rawMedication(String name, String dosage, String expiryDate) {
        Map<String, Object> item = new HashMap<>();
        item.put("Brand Name", name);
        item.put("Dosage", dosage);
        if (expiryDate != null) {
            item.put("Expiry Date", expiryDate);
        }

        Map<String, Object> raw = new HashMap<>();
        raw.put("detection_method", "barcode");
        raw.put("source", "barcode");
        raw.put("detected_count", 1);
        raw.put("Drug Details", item);
        return raw;
    }

    @Nested
    @DisplayName("file validation")
    class FileValidation {
        @Test
        @DisplayName("throws when the file is missing or empty")
        void scanBarcode_throwsForEmptyFile() {
            MultipartFile empty = new MockMultipartFile("file", "empty.jpg", "image/jpeg", new byte[0]);

            assertThatThrownBy(() -> scanService.scanBarcode(USER_ID, empty))
                    .isInstanceOf(BadRequestException.class)
                    .hasMessageContaining("file is required");
        }

        @Test
        @DisplayName("throws when the file is not an image")
        void scanBarcode_throwsForNonImageFile() {
            MultipartFile pdf = new MockMultipartFile("file", "file.pdf", "application/pdf", "data".getBytes());

            assertThatThrownBy(() -> scanService.scanBarcode(USER_ID, pdf))
                    .isInstanceOf(BadRequestException.class)
                    .hasMessageContaining("Only image files are supported");
        }
    }

    @Nested
    @DisplayName("scan and save")
    class ScanAndSave {
        @Test
        @DisplayName("when the same medication and expiry already exist, quantity is incremented")
        void scanBarcode_incrementsQuantityForDuplicateMedication() {
            LocalDate expiry = LocalDate.of(2026, 6, 30);
            MedicationEntity existing = new MedicationEntity();
            existing.setId(UUID.randomUUID());
            existing.setUserId(USER_ID);
            existing.setName("Paracetamol");
            existing.setDosage("500mg");
            existing.setExpiryDate(expiry);
            existing.setFrequency("");
            existing.setQuantity(1);
            existing.setStatus(MedicationStatus.available);
            existing.setCreatedAt(Instant.now());
            existing.setReminderTimes(List.of());

            when(scannerClient.scan(any(MultipartFile.class)))
                    .thenReturn(rawMedication("Paracetamol", "500mg", "2026-06-30"));
            when(medicationRepository.findByUserIdAndNameIgnoreCaseAndExpiryDate(USER_ID, "Paracetamol", expiry))
                    .thenReturn(Optional.of(existing));
            when(medicationRepository.save(any(MedicationEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));

            MedicationScanBatchResponse response = scanService.scanBarcode(USER_ID, imageFile());

            assertThat(existing.getQuantity()).isEqualTo(2);
            assertThat(response.items()).singleElement().satisfies(item -> {
                assertThat(item.action()).isEqualTo("incremented");
                assertThat(item.medication().quantity()).isEqualTo(2);
                assertThat(item.medication().name()).isEqualTo("Paracetamol");
            });
            verify(medicationRepository).save(existing);
        }

        @Test
        @DisplayName("when no matching medication exists, a new medication is created")
        void scanBarcode_createsNewMedicationWhenNoDuplicateExists() {
            LocalDate expiry = LocalDate.of(2025, 12, 31);
            when(scannerClient.scan(any(MultipartFile.class)))
                    .thenReturn(rawMedication("Ibuprofen", "400mg", "2025-12-31"));
            when(medicationRepository.findByUserIdAndNameIgnoreCaseAndExpiryDate(USER_ID, "Ibuprofen", expiry))
                    .thenReturn(Optional.empty());
            when(medicationRepository.save(any(MedicationEntity.class))).thenAnswer(invocation -> {
                MedicationEntity saved = invocation.getArgument(0);
                saved.setId(UUID.randomUUID());
                return saved;
            });

            MedicationScanBatchResponse response = scanService.scanBarcode(USER_ID, imageFile());

            assertThat(response.source()).isEqualTo("barcode");
            assertThat(response.detectedCount()).isEqualTo(1);
            assertThat(response.items()).singleElement().satisfies(item -> {
                assertThat(item.action()).isEqualTo("created");
                assertThat(item.medication().name()).isEqualTo("Ibuprofen");
                assertThat(item.medication().status()).isEqualTo("available");
            });
        }

        @Test
        @DisplayName("OCR scan without expiry creates needs_review medication with a placeholder expiry")
        void scanText_missingExpirySetsNeedsReview() {
            Map<String, Object> raw = rawMedication("SomeDrug", "10mg", null);
            raw.put("detection_method", "ocr");
            raw.put("source", "ocr");

            when(scannerClient.scanText(any(MultipartFile.class))).thenReturn(raw);
            when(medicationRepository.findByUserIdAndNameIgnoreCase(USER_ID, "SomeDrug")).thenReturn(List.of());
            when(medicationRepository.save(any(MedicationEntity.class))).thenAnswer(invocation -> {
                MedicationEntity saved = invocation.getArgument(0);
                saved.setId(UUID.randomUUID());
                return saved;
            });

            MedicationScanBatchResponse response = scanService.scanText(USER_ID, imageFile());

            assertThat(response.items()).singleElement().satisfies(item -> {
                assertThat(item.action()).isEqualTo("created");
                assertThat(item.medication().status()).isEqualTo("needs_review");
                assertThat(item.medication().expiryDate())
                        .isAfterOrEqualTo(LocalDate.now().plusYears(1).minusDays(1))
                        .isBeforeOrEqualTo(LocalDate.now().plusYears(1).plusDays(1));
                assertThat(item.medication().notes()).contains("Expiry date not detected");
            });
        }

        @Test
        @DisplayName("throws when scanner returns no valid medication names")
        void scanBarcode_throwsWhenScannerReturnsNoValidMedication() {
            when(scannerClient.scan(any(MultipartFile.class))).thenReturn(Map.of("detected_count", 0));

            assertThatThrownBy(() -> scanService.scanBarcode(USER_ID, imageFile()))
                    .isInstanceOf(BadRequestException.class)
                    .hasMessageContaining("Scanner did not return any valid medications");
        }
    }

    @Nested
    @DisplayName("key normalisation")
    class KeyNormalisation {
        @Test
        @DisplayName("Title Case key 'Brand Name' maps to medication name")
        void titleCaseBrandNameMapsToMedicationName() {
            when(scannerClient.scan(any(MultipartFile.class)))
                    .thenReturn(rawMedication("Solpadeine", "500mg", "2026-03-01"));
            when(medicationRepository.findByUserIdAndNameIgnoreCaseAndExpiryDate(
                    USER_ID, "Solpadeine", LocalDate.of(2026, 3, 1)))
                    .thenReturn(Optional.empty());
            when(medicationRepository.save(any(MedicationEntity.class))).thenAnswer(invocation -> {
                MedicationEntity saved = invocation.getArgument(0);
                saved.setId(UUID.randomUUID());
                return saved;
            });

            MedicationScanBatchResponse response = scanService.scanBarcode(USER_ID, imageFile());

            assertThat(response.items()).singleElement()
                    .satisfies(item -> assertThat(item.medication().name()).isEqualTo("Solpadeine"));
        }

        @Test
        @DisplayName("lowercase key 'name' also maps to medication name")
        void lowercaseNameMapsToMedicationName() {
            Map<String, Object> raw = Map.of(
                    "detection_method", "barcode",
                    "Drug Details", Map.of(
                            "name", "Panadol",
                            "dosage", "500mg",
                            "expiry_date", "2026-01-01"
                    )
            );
            when(scannerClient.scan(any(MultipartFile.class))).thenReturn(raw);
            when(medicationRepository.findByUserIdAndNameIgnoreCaseAndExpiryDate(
                    USER_ID, "Panadol", LocalDate.of(2026, 1, 1)))
                    .thenReturn(Optional.empty());
            when(medicationRepository.save(any(MedicationEntity.class))).thenAnswer(invocation -> {
                MedicationEntity saved = invocation.getArgument(0);
                saved.setId(UUID.randomUUID());
                return saved;
            });

            MedicationScanBatchResponse response = scanService.scanBarcode(USER_ID, imageFile());

            assertThat(response.items()).singleElement()
                    .satisfies(item -> assertThat(item.medication().name()).isEqualTo("Panadol"));
        }
    }
}
