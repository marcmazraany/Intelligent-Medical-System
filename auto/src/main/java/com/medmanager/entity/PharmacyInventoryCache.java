package com.medmanager.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDateTime;

@Entity
@Table(
        name = "pharmacy_inventory_cache",
        uniqueConstraints = {
                // One row per pharmacy + medication name.
                // The DB enforces this so concurrent writes can never create duplicates.
                @UniqueConstraint(
                        name = "uq_pharmacy_medication",
                        columnNames = {"pharmacyNodeId", "medicationName"}
                )
        }
)
@Data
@NoArgsConstructor
public class PharmacyInventoryCache {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long pharmacyNodeId;

    @Column(nullable = false)
    private String pharmacyName;

    private Long medicationId;      // nullable — name is the primary key now

    @Column(nullable = false)
    private String medicationName;

    @Column(nullable = false)
    private Integer stockQuantity;

    @Column(nullable = false)
    private Double price;

    private String currency = "LBP";

    @Column(nullable = false)
    private Double pharmacyLatitude;

    @Column(nullable = false)
    private Double pharmacyLongitude;

    @Column(nullable = false)
    private LocalDateTime lastUpdated;

    @Column(nullable = false)
    private LocalDateTime nextCheckTime;

    // OUT_OF_STOCK / HIGH / MEDIUM / LOW
    @Column(nullable = false)
    private String priority;

    @Column(nullable = false)
    private boolean available;
}