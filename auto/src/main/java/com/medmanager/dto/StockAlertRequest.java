package com.medmanager.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class StockAlertRequest {
    private String userEmail;
    private String userPhone;
    private String medicationName;
    private Double maxPrice;
    private Double maxDistance;
    private boolean notifyByEmail;
    private boolean notifyBySMS;
}