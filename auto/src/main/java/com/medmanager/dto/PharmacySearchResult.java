package com.medmanager.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class PharmacySearchResult {
    private String pharmacyName;
    private String address;
    private double latitude;
    private double longitude;
    private double distanceKm;
    private Integer travelTimeMinutes;
    private int stockQuantity;
    private double price;
    private String currency;
    private boolean inStock;
    private String googleMapsUrl;
    private String lastUpdated;
    private String pharmacyPhone;
}