package com.intellimeds.api.medications;

import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface MedicationRepository extends JpaRepository<MedicationEntity, UUID> {
    List<MedicationEntity> findAllByUserIdOrderByUpdatedAtDesc(UUID userId);
    List<MedicationEntity> findAllByUserIdOrderByCreatedAtDesc(UUID userId);
    Optional<MedicationEntity> findByIdAndUserId(UUID id, UUID userId);
    Optional<MedicationEntity> findByUserIdAndNameIgnoreCaseAndExpiryDate(UUID userId, String name, LocalDate expiryDate);

    /**
     * Find medications by user ID and name only (case-insensitive).
     * Used for OCR fallback when expiry date is not detected.
     * Returns all matching medications so we can find the most recent one.
     */
    List<MedicationEntity> findByUserIdAndNameIgnoreCase(UUID userId, String name);

    void deleteByIdAndUserId(UUID id, UUID userId);
}