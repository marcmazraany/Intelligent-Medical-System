package com.intellimeds.api.pharmacy.controller;

import com.intellimeds.api.pharmacy.config.PharmacyFinderProperties;
import com.intellimeds.api.pharmacy.dto.MedicationAvailabilityResponse;
import com.intellimeds.api.pharmacy.service.PharmacyFinderClient;
import com.intellimeds.api.security.AuthUser;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

/**
 * Pharmacy Finder endpoints exposed to the frontend.
 * All requests require a valid JWT (enforced by SecurityConfig).
 *
 * Frontend calls IntelliMeds (8080) → IntelliMeds calls auto system (8090).
 * Frontend never talks to the auto system directly.
 *
 * GET  /api/pharmacy-finder/search?name=Panadol
 * POST /api/pharmacy-finder/ping?name=Panadol
 */
@RestController
@RequestMapping("/api/pharmacy-finder")
@RequiredArgsConstructor
@Slf4j
public class PharmacyFinderController {

    private final PharmacyFinderClient     client;
    private final PharmacyFinderProperties props;

    /**
     * Cache-first search.
     * The auto system checks its cache first; on a miss it pings pharmacies live.
     */
    @GetMapping("/search")
    public MedicationAvailabilityResponse search(
            @AuthenticationPrincipal AuthUser user,
            @RequestParam String name,
            @RequestParam(required = false) Double latitude,
            @RequestParam(required = false) Double longitude,
            @RequestParam(defaultValue = "10") int maxResults) {

        double lat = latitude  != null ? latitude  : props.defaultLatitude();
        double lon = longitude != null ? longitude : props.defaultLongitude();

        log.info("🔍 [{}] Search: '{}'", user.email(), name);
        return client.search(name, lat, lon, maxResults);
    }

    /**
     * Force live ping, bypassing the auto system's cache.
     * Use this when the user explicitly refreshes results.
     */
    @PostMapping("/ping")
    public MedicationAvailabilityResponse ping(
            @AuthenticationPrincipal AuthUser user,
            @RequestParam String name,
            @RequestParam(required = false) Double latitude,
            @RequestParam(required = false) Double longitude,
            @RequestParam(defaultValue = "50") double maxDistanceKm) {

        double lat = latitude  != null ? latitude  : props.defaultLatitude();
        double lon = longitude != null ? longitude : props.defaultLongitude();

        log.info("🔔 [{}] Force ping: '{}'", user.email(), name);
        return client.forcePing(name, lat, lon, maxDistanceKm);
    }
}
