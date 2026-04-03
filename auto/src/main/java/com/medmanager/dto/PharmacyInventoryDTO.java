package com.medmanager.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class PharmacyInventoryDTO {
    private Long medicationId;
    private String medicationName;
    private String dosage;
    private int stockQuantity;
    private double price;
    private String currency;
    private boolean available;
}