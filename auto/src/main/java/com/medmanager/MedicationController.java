package com.medmanager.controller;

import com.medmanager.dto.*;
import com.medmanager.entity.*;
import com.medmanager.repository.*;
import com.medmanager.service.*;
import com.medmanager.service.GoogleMapsDistanceService.DistanceResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/medications")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
@Slf4j
public class MedicationController {

    private final MedicationRepository medicationRepository;
    private final PharmacyInventoryCacheRepository cacheRepository;
    private final StockAlertRepository alertRepository;
    private final PharmacyPingService pingService;
    private final GoogleMapsDistanceService distanceService;

    @Value("${user.default.latitude}")
    private double defaultLat;

    @Value("${user.default.longitude}")
    private double defaultLon;

    // ── GET all medications in the system ────────────────────────────────────
    @GetMapping
    public ResponseEntity<List<Medication>> getAllMedications() {
        return ResponseEntity.ok(medicationRepository.findAll());
    }

    // ── MAIN SEARCH ──────────────────────────────────────────────────────────
    @GetMapping("/search")
    public ResponseEntity<MedicationAvailabilityResponse> search(
            @RequestParam String name,
            @RequestParam(required = false) Double latitude,
            @RequestParam(required = false) Double longitude,
            @RequestParam(defaultValue = "10") int maxResults) {

        double userLat = latitude  != null ? latitude  : defaultLat;
        double userLon = longitude != null ? longitude : defaultLon;

        log.info("🔍 Search: '{}' from ({}, {})", name, userLat, userLon);

        // ── Step 1: check cache ──────────────────────────────────────────────
        List<PharmacyInventoryCache> cached =
                cacheRepository.findByMedicationNameIgnoreCaseAndAvailableTrue(name);

        if (cached.isEmpty()) {
            cached = cacheRepository.findByMedicationNameContainingIgnoreCaseAndAvailableTrue(name);
        }

        if (!cached.isEmpty()) {
            log.info("✅ Cache hit: {} pharmacies have '{}'", cached.size(), name);
            return ResponseEntity.ok(buildResponse(name, cached, userLat, userLon, maxResults));
        }

        // ── Step 2: cache miss — ping only pingable pharmacies live ──────────
        // Pharmacies with supportsLivePing=false (agent-based local DBs) are
        // skipped here. Their data will appear once the agent syncs.
        log.info("❌ Cache miss for '{}' — pinging live pharmacies...", name);

        List<PharmacySearchResult> liveResults =
                pingService.pingAllPharmaciesByName(name, userLat, userLon, 50.0);

        // Update cache with whatever we just found
        pingService.updateCacheAfterPingByName(name);

        if (!liveResults.isEmpty()) {
            // Re-read from cache so response is consistent
            cached = cacheRepository.findByMedicationNameIgnoreCaseAndAvailableTrue(name);
            if (cached.isEmpty()) {
                // Cache not yet populated — return live results directly
                return ResponseEntity.ok(buildResponseFromLive(name, liveResults, maxResults));
            }
            return ResponseEntity.ok(buildResponse(name, cached, userLat, userLon, maxResults));
        }

        // ── Step 3: nothing found anywhere ──────────────────────────────────
        log.info("🚫 '{}' not found in any reachable pharmacy", name);
        MedicationAvailabilityResponse empty = new MedicationAvailabilityResponse();
        empty.setMedicationName(name);
        empty.setDosage("");
        empty.setTotalPharmaciesChecked(0);
        empty.setPharmaciesWithStock(0);
        empty.setPharmacies(List.of());
        return ResponseEntity.ok(empty);
    }

    // ── FORCE PING (bypass cache) ────────────────────────────────────────────
    @PostMapping("/ping")
    public ResponseEntity<PingAllResponse> forcePing(
            @RequestParam String name,
            @RequestParam(required = false) Double latitude,
            @RequestParam(required = false) Double longitude,
            @RequestParam(defaultValue = "50") double maxDistanceKm) {

        double userLat = latitude  != null ? latitude  : defaultLat;
        double userLon = longitude != null ? longitude : defaultLon;

        List<PharmacySearchResult> results =
                pingService.pingAllPharmaciesByName(name, userLat, userLon, maxDistanceKm);

        pingService.updateCacheAfterPingByName(name);

        PingAllResponse response = new PingAllResponse();
        response.setMedicationName(name);
        response.setPharmaciesContacted(results.size() + " pharmacies responded");
        response.setPharmaciesWithStock(results);
        response.setSearchRadiusKm(maxDistanceKm);
        response.setVerifiedAt(LocalDateTime.now().toString());
        return ResponseEntity.ok(response);
    }

    // ── ALERTS ───────────────────────────────────────────────────────────────
    @PostMapping("/alerts")
    public ResponseEntity<StockAlert> createAlert(@RequestBody StockAlertRequest request) {
        StockAlert alert = new StockAlert();
        alert.setUserEmail(request.getUserEmail());
        alert.setUserPhone(request.getUserPhone());
        alert.setMedicationName(request.getMedicationName());
        alert.setMaxPrice(request.getMaxPrice());
        alert.setMaxDistance(request.getMaxDistance());
        alert.setNotifyByEmail(request.isNotifyByEmail());
        alert.setNotifyBySMS(request.isNotifyBySMS());
        log.info("🔔 Alert created for '{}' → {}", request.getMedicationName(), request.getUserEmail());
        return ResponseEntity.ok(alertRepository.save(alert));
    }

    @GetMapping("/alerts")
    public ResponseEntity<List<StockAlert>> getAlerts(@RequestParam String email) {
        return ResponseEntity.ok(alertRepository.findByUserEmailAndActiveTrue(email));
    }

    @DeleteMapping("/alerts/{id}")
    public ResponseEntity<Void> deleteAlert(@PathVariable Long id) {
        alertRepository.findById(id).ifPresent(a -> {
            a.setActive(false);
            alertRepository.save(a);
        });
        return ResponseEntity.noContent().build();
    }

    // ── Deprecated ID-based endpoint kept for backward compatibility ─────────
    @GetMapping("/{medicationId}/availability")
    @Deprecated
    public ResponseEntity<MedicationAvailabilityResponse> searchById(
            @PathVariable Long medicationId,
            @RequestParam(required = false) Double latitude,
            @RequestParam(required = false) Double longitude,
            @RequestParam(defaultValue = "10") int maxResults) {

        return medicationRepository.findById(medicationId)
                .map(med -> search(med.getName(), latitude, longitude, maxResults))
                .orElse(ResponseEntity.notFound().build());
    }

    // ── Response builders ────────────────────────────────────────────────────
    private MedicationAvailabilityResponse buildResponse(
            String name, List<PharmacyInventoryCache> cached,
            double userLat, double userLon, int maxResults) {

        List<PharmacySearchResult> results = cached.stream()
                .map(c -> {
                    DistanceResult dist = distanceService.calculateDistance(
                            userLat, userLon, c.getPharmacyLatitude(), c.getPharmacyLongitude());
                    PharmacySearchResult r = new PharmacySearchResult();
                    r.setPharmacyName(c.getPharmacyName());
                    r.setLatitude(c.getPharmacyLatitude());
                    r.setLongitude(c.getPharmacyLongitude());
                    r.setDistanceKm(dist.distanceKm);
                    r.setTravelTimeMinutes(dist.travelTimeMinutes);
                    r.setStockQuantity(c.getStockQuantity());
                    r.setPrice(c.getPrice());
                    r.setCurrency(c.getCurrency());
                    r.setInStock(c.isAvailable());
                    r.setGoogleMapsUrl(String.format(
                            "https://www.google.com/maps/search/?api=1&query=%f,%f",
                            c.getPharmacyLatitude(), c.getPharmacyLongitude()));
                    r.setLastUpdated(timeAgo(c.getLastUpdated()));
                    return r;
                })
                .sorted(Comparator.comparingDouble(PharmacySearchResult::getDistanceKm))
                .limit(maxResults)
                .collect(Collectors.toList());

        MedicationAvailabilityResponse resp = new MedicationAvailabilityResponse();
        resp.setMedicationName(cached.get(0).getMedicationName());
        resp.setDosage("");
        resp.setTotalPharmaciesChecked(cached.size());
        resp.setPharmaciesWithStock(results.size());
        resp.setPharmacies(results);
        return resp;
    }

    private MedicationAvailabilityResponse buildResponseFromLive(
            String name, List<PharmacySearchResult> live, int maxResults) {
        MedicationAvailabilityResponse resp = new MedicationAvailabilityResponse();
        resp.setMedicationName(name);
        resp.setDosage("");
        resp.setTotalPharmaciesChecked(live.size());
        resp.setPharmaciesWithStock(live.size());
        resp.setPharmacies(live.stream().limit(maxResults).collect(Collectors.toList()));
        return resp;
    }

    private String timeAgo(LocalDateTime dt) {
        long mins = ChronoUnit.MINUTES.between(dt, LocalDateTime.now());
        if (mins < 1)  return "just now";
        if (mins < 60) return mins + " min ago";
        long hrs = mins / 60;
        if (hrs < 24)  return hrs + " hr ago";
        return (hrs / 24) + " days ago";
    }
}