package com.intellimeds.api.medications;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "medications")
@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MedicationEntity {

    @Id
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Column(name = "user_id", nullable = false)
    private UUID userId;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private String dosage;

    @Column(name = "expiry_date", nullable = false)
    private LocalDate expiryDate;

    @Column(nullable = false)
    private String frequency;

    @Column(nullable = false)
    private Integer quantity;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "reminder_times", nullable = false, columnDefinition = "jsonb")
    @Builder.Default
    private List<String> reminderTimes = new ArrayList<>();

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private MedicationStatus status;

    @Column(columnDefinition = "text")
    private String notes;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    @PrePersist
    void prePersist() {
        if (id == null) id = UUID.randomUUID();
        if (createdAt == null) createdAt = Instant.now();
        updatedAt = Instant.now();
        if (status == null) status = MedicationStatus.available;
        if (frequency == null) frequency = "";
        if (quantity == null || quantity < 1) quantity = 1;
        if (reminderTimes == null) reminderTimes = new ArrayList<>();
    }

    @PreUpdate
    void preUpdate() {
        updatedAt = Instant.now();
    }
}