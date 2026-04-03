package com.intellimeds.api.pharmacy.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Mirrors the MedicationAvailabilityResponse the auto system returns.
 * IntelliMeds receives this, optionally enriches it, and forwards to the frontend.
 */
@Data
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class MedicationAvailabilityResponse {
    private String medicationName;
    private String dosage;
    private int    totalPharmaciesChecked;
    private int    totalPharmaciesWithStock;
    private List<PharmacySearchResult> pharmacies;
}
