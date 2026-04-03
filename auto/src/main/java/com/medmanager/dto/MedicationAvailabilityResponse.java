package com.medmanager.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
@NoArgsConstructor
public class MedicationAvailabilityResponse {
    private String medicationName;
    private String dosage;
    private int totalPharmaciesChecked;
    private int pharmaciesWithStock;
    private List<PharmacySearchResult> pharmacies;
}