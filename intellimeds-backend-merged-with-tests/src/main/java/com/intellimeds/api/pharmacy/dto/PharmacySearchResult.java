package com.intellimeds.api.pharmacy.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Mirrors the PharmacySearchResult the auto system returns.
 * JsonIgnoreProperties so unknown extra fields don't break deserialization.
 */
@Data
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class PharmacySearchResult {
    private String  pharmacyName;
    private String  address;
    private double  latitude;
    private double  longitude;
    private double  distanceKm;
    private Integer travelTimeMinutes;
    private int     stockQuantity;
    private double  price;
    private String  currency;
    private boolean inStock;
    private String  googleMapsUrl;
    private String  lastUpdated;
    private String  pharmacyPhone;
}
