package com.intellimeds.api.alerts;

import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "alerts")
@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AlertEntity {

    @Id
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Column(name = "user_id", nullable = false)
    private UUID userId;

    @Column(name = "medication_name", nullable = false)
    private String medicationName;

    @Column(name = "max_price", nullable = false)
    private BigDecimal maxPrice;

    @Column(name = "email_enabled", nullable = false)
    private boolean emailEnabled;

    @Column(name = "created_date", nullable = false, updatable = false)
    private Instant createdDate;

    @Column(name = "last_notified")
    private Instant lastNotified;

    @Column(nullable = false)
    private String status;

    @Builder.Default
    @Column(nullable = false)
    private boolean active = true;

    /**
     * Optional km radius — if set, only notify when a pharmacy within
     * this distance has the medication in stock.
     * Added by V4 migration. Nullable = no distance filter.
     */
    @Column(name = "max_distance")
    private Double maxDistance;

    @PrePersist
    void prePersist() {
        if (id == null) id = UUID.randomUUID();
        if (createdDate == null) createdDate = Instant.now();
        if (status == null) status = "active";
    }
}
