package com.intellimeds.api.pharmacy.service;

import com.intellimeds.api.pharmacy.config.PharmacyFinderProperties;
import com.intellimeds.api.pharmacy.dto.MedicationAvailabilityResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.util.Collections;

/**
 * Thin HTTP client that forwards requests to the pharmacy-finder / auto system.
 * IntelliMeds never reads pharmacy node data directly — it delegates to this class.
 */
@Component
@Slf4j
public class PharmacyFinderClient {

    private final RestTemplate    http;
    private final PharmacyFinderProperties props;

    public PharmacyFinderClient(
            @Qualifier("pharmacyRestTemplate") RestTemplate http,
            PharmacyFinderProperties props) {
        this.http  = http;
        this.props = props;
    }

    /**
     * Search for a medication by name.
     * Maps to GET /api/medications/search on the auto system.
     */
    public MedicationAvailabilityResponse search(
            String name, double lat, double lon, int maxResults) {

        String url = UriComponentsBuilder
                .fromHttpUrl(props.baseUrl() + "/api/medications/search")
                .queryParam("name",       name)
                .queryParam("latitude",   lat)
                .queryParam("longitude",  lon)
                .queryParam("maxResults", maxResults)
                .toUriString();

        try {
            MedicationAvailabilityResponse result =
                    http.getForObject(url, MedicationAvailabilityResponse.class);

            if (result == null) {
                return emptyResponse(name);
            }
            log.info("✅ Auto system returned {} pharmacies for '{}'",
                    result.getTotalPharmaciesWithStock(), name);
            return result;

        } catch (Exception e) {
            log.error("❌ Auto system unreachable for '{}': {}", name, e.getMessage());
            return emptyResponse(name);
        }
    }

    /**
     * Force a live ping, bypassing the auto system's cache.
     * Maps to POST /api/medications/ping on the auto system.
     */
    public MedicationAvailabilityResponse forcePing(
            String name, double lat, double lon, double maxDistanceKm) {

        String url = UriComponentsBuilder
                .fromHttpUrl(props.baseUrl() + "/api/medications/ping")
                .queryParam("name",           name)
                .queryParam("latitude",       lat)
                .queryParam("longitude",      lon)
                .queryParam("maxDistanceKm",  maxDistanceKm)
                .toUriString();

        try {
            MedicationAvailabilityResponse result =
                    http.postForObject(url, null, MedicationAvailabilityResponse.class);

            return result != null ? result : emptyResponse(name);

        } catch (Exception e) {
            log.error("❌ Auto system ping failed for '{}': {}", name, e.getMessage());
            return emptyResponse(name);
        }
    }

    // ── helpers ───────────────────────────────────────────────────────
    private MedicationAvailabilityResponse emptyResponse(String name) {
        MedicationAvailabilityResponse r = new MedicationAvailabilityResponse();
        r.setMedicationName(name);
        r.setDosage("");
        r.setTotalPharmaciesChecked(0);
        r.setTotalPharmaciesWithStock(0);
        r.setPharmacies(Collections.emptyList());
        return r;
    }
}
