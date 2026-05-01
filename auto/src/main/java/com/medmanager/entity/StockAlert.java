package com.medmanager.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDateTime;

@Entity
@Table(name = "stock_alerts")
@Data
@NoArgsConstructor
public class StockAlert {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String userEmail;

    private String userPhone;

    @Column(nullable = false)
    private String medicationName;

    private Long medicationId;      // kept for backward compatibility, not used

    private Double maxPrice;
    private Double maxDistance;

    @Column(nullable = false)
    private boolean active;

    @Column(nullable = false)
    private boolean notifyByEmail;

    @Column(nullable = false)
    private boolean notifyBySMS;

    private LocalDateTime createdAt;
    private LocalDateTime lastNotified;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        active = true;
    }
}