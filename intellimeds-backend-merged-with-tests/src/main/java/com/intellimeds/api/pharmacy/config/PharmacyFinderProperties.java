package com.intellimeds.api.pharmacy.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Binds to application.properties prefix "pharmacy.finder".
 *
 * Required entries (add to your .env / application.properties):
 *
 *   pharmacy.finder.base-url=http://localhost:8090
 *   pharmacy.finder.default-latitude=33.8886
 *   pharmacy.finder.default-longitude=35.4955
 */
@ConfigurationProperties(prefix = "pharmacy.finder")
public record PharmacyFinderProperties(
        String baseUrl,
        double defaultLatitude,
        double defaultLongitude
) {
    public PharmacyFinderProperties {
        if (baseUrl == null || baseUrl.isBlank())
            baseUrl = "http://localhost:8090";
        if (defaultLatitude  == 0) defaultLatitude  = 33.8886;
        if (defaultLongitude == 0) defaultLongitude = 35.4955;
    }
}
