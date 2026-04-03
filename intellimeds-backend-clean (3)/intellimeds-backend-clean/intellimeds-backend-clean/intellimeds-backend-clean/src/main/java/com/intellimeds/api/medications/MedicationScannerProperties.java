package com.intellimeds.api.medications;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "scanner.service")
public record MedicationScannerProperties(
        String baseUrl,
        String scanPath
) {}