package com.medmanager.service;

import com.medmanager.dto.PharmacyInventoryDTO;
import com.medmanager.dto.PharmacySearchResult;
import com.medmanager.entity.PharmacyInventoryCache;
import com.medmanager.entity.PharmacyNode;
import com.medmanager.repository.PharmacyInventoryCacheRepository;
import com.medmanager.repository.PharmacyNodeRepository;
import com.medmanager.service.GoogleMapsDistanceService.DistanceResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class PharmacyPingService {

    private final PharmacyNodeRepository pharmacyRepository;
    private final PharmacyInventoryCacheRepository cacheRepository;
    private final GoogleMapsDistanceService distanceService;
    private final PharmacyDataSourceService dataSourceService;

    // Dedicated thread pool for parallel pings
    private final ExecutorService pingPool = Executors.newFixedThreadPool(20);

    // ── Main entry point ─────────────────────────────────────────────────────
    // Only pings pharmacies where supportsLivePing = true.
    // Agent-based (local DB/Excel) pharmacies are skipped — their cache
    // is populated by the agent on its own schedule.
    public List<PharmacySearchResult> pingAllPharmaciesByName(
            String medicationName, double userLat, double userLon, double maxDistanceKm) {

        log.info("🔔 PING: searching within {}km for '{}'", maxDistanceKm, medicationName);

        // Only pingable pharmacies
        List<PharmacyNode> pingable = pharmacyRepository.findByActiveTrueAndSupportsLivePingTrue();

        // Cheap Haversine pre-filter — avoid Google Maps calls for far pharmacies
        List<PharmacyNode> nearby = pingable.stream()
                .filter(p -> distanceService.haversineKm(
                        userLat, userLon, p.getLatitude(), p.getLongitude()) <= maxDistanceKm * 1.3)
                .collect(Collectors.toList());

        log.info("📍 {} pingable pharmacies in range", nearby.size());

        List<PharmacySearchResult> results = Collections.synchronizedList(new ArrayList<>());

        List<CompletableFuture<Void>> futures = nearby.stream()
                .map(pharmacy -> CompletableFuture.runAsync(() -> {
                    try {
                        // KEY FIX: read via dataSourceService, not hardcoded WebClient
                        // This means REST, Supabase, MySQL, Excel, CSV all work here
                        List<PharmacyInventoryDTO> inventory = dataSourceService.readInventory(pharmacy);

                        if (inventory == null || inventory.isEmpty()) return;

                        inventory.stream()
                                .filter(item -> item.getMedicationName() != null &&
                                        item.getMedicationName().toLowerCase()
                                                .contains(medicationName.toLowerCase()))
                                .filter(item -> item.isAvailable() && item.getStockQuantity() > 0)
                                .findFirst()
                                .ifPresent(item -> {
                                    DistanceResult dist = distanceService.calculateDistance(
                                            userLat, userLon, pharmacy.getLatitude(), pharmacy.getLongitude());

                                    if (dist.distanceKm > maxDistanceKm) return;

                                    PharmacySearchResult result = new PharmacySearchResult();
                                    result.setPharmacyName(pharmacy.getName());
                                    result.setAddress(pharmacy.getAddress());
                                    result.setLatitude(pharmacy.getLatitude());
                                    result.setLongitude(pharmacy.getLongitude());
                                    result.setDistanceKm(dist.distanceKm);
                                    result.setTravelTimeMinutes(dist.travelTimeMinutes);
                                    result.setStockQuantity(item.getStockQuantity());
                                    result.setPrice(item.getPrice());
                                    result.setCurrency(item.getCurrency() != null ? item.getCurrency() : "LBP");
                                    result.setInStock(true);
                                    result.setGoogleMapsUrl(mapsUrl(pharmacy.getLatitude(), pharmacy.getLongitude()));
                                    result.setLastUpdated("Verified just now");
                                    result.setPharmacyPhone(pharmacy.getPhoneNumber());
                                    results.add(result);
                                    log.info("✅ {} has '{}' ({} units)",
                                            pharmacy.getName(), item.getMedicationName(), item.getStockQuantity());
                                });
                    } catch (Exception e) {
                        log.warn("⚠️ {} ping failed: {}", pharmacy.getName(), e.getMessage());
                    }
                }, pingPool))
                .collect(Collectors.toList());

        try {
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                    .get(12, TimeUnit.SECONDS);
        } catch (Exception e) {
            log.warn("⏰ Some pings timed out — returning partial results");
        }

        results.sort(Comparator.comparingDouble(PharmacySearchResult::getDistanceKm));
        log.info("🎯 PING DONE: {}/{} have '{}'", results.size(), nearby.size(), medicationName);
        return results;
    }

    // ── Cache upsert after ping ───────────────────────────────────────────────
    @Transactional
    public void updateCacheAfterPingByName(String medicationName) {
        List<PharmacyNode> pingable = pharmacyRepository.findByActiveTrueAndSupportsLivePingTrue();
        for (PharmacyNode pharmacy : pingable) {
            try {
                List<PharmacyInventoryDTO> inventory = dataSourceService.readInventory(pharmacy);
                upsertCache(pharmacy, inventory, medicationName);
            } catch (Exception e) {
                log.warn("Cache update failed for {}: {}", pharmacy.getName(), e.getMessage());
            }
        }
    }

    // ── Upsert — safe against concurrent writes ───────────────────────────────
    @Transactional
    public void upsertCache(PharmacyNode pharmacy, List<PharmacyInventoryDTO> inventory,
                            String medicationName) {
        LocalDateTime now = LocalDateTime.now();
        inventory.stream()
                .filter(item -> item.getMedicationName() != null &&
                        item.getMedicationName().equalsIgnoreCase(medicationName))
                .findFirst()
                .ifPresent(item -> {
                    List<PharmacyInventoryCache> existing = cacheRepository
                            .findByPharmacyNodeIdAndMedicationNameIgnoreCase(
                                    pharmacy.getId(), medicationName);

                    PharmacyInventoryCache entry = existing.isEmpty()
                            ? new PharmacyInventoryCache() : existing.get(0);

                    if (existing.size() > 1) {
                        cacheRepository.deleteAll(existing.subList(1, existing.size()));
                    }

                    entry.setPharmacyNodeId(pharmacy.getId());
                    entry.setPharmacyName(pharmacy.getName());
                    entry.setMedicationId(item.getMedicationId());
                    entry.setMedicationName(item.getMedicationName());
                    entry.setStockQuantity(item.getStockQuantity());
                    entry.setPrice(item.getPrice());
                    entry.setCurrency(item.getCurrency() != null ? item.getCurrency() : "LBP");
                    entry.setPharmacyLatitude(pharmacy.getLatitude());
                    entry.setPharmacyLongitude(pharmacy.getLongitude());
                    entry.setLastUpdated(now);
                    entry.setAvailable(item.isAvailable() && item.getStockQuantity() > 0);
                    entry.setPriority(priority(item.getStockQuantity()));
                    entry.setNextCheckTime(nextCheck(now, item.getStockQuantity()));
                    cacheRepository.save(entry);
                });
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    private String priority(int qty) {
        if (qty <= 0) return "OUT_OF_STOCK";
        if (qty <= 5) return "HIGH";
        if (qty <= 20) return "MEDIUM";
        return "LOW";
    }

    private LocalDateTime nextCheck(LocalDateTime now, int qty) {
        if (qty <= 0)  return now.plusMinutes(10);
        if (qty <= 5)  return now.plusMinutes(3);
        if (qty <= 20) return now.plusMinutes(6);
        return now.plusMinutes(10);
    }

    private String mapsUrl(double lat, double lon) {
        return String.format("https://www.google.com/maps/search/?api=1&query=%f,%f", lat, lon);
    }
}