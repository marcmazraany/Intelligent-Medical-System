package com.medmanager.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
@NoArgsConstructor
public class PingAllResponse {
    private String medicationName;
    private String pharmaciesContacted;
    private List<PharmacySearchResult> pharmaciesWithStock;
    private double searchRadiusKm;
    private String verifiedAt;
}