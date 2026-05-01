package com.medmanager.service;

import com.medmanager.dto.PharmacyInventoryDTO;
import com.medmanager.entity.*;
import com.medmanager.repository.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class SmartPharmacyScanner {

    private final PharmacyNodeRepository pharmacyNodeRepository;
    private final PharmacyInventoryCacheRepository cacheRepository;
    private final PharmacyDataSourceService dataSourceService;
    private final PharmacyPingService pingService;
    private final StockAlertRepository alertRepository;
    private final NotificationService notificationService;

    @Value("${pharmacy.network.enabled:true}")
    private boolean enabled;

    // =========================================================
    // FULL SCAN — every 5 minutes
    // Reads ALL medications from ALL active pharmacies
    // =========================================================
    @Scheduled(fixedDelay = 300_000)
    @Transactional
    public void fullInventoryScan() {
        if (!enabled) return;

        log.info("═══════════════════════════════════════");
        log.info("🔄 FULL SCAN STARTED");

        List<PharmacyNode> pharmacies = pharmacyNodeRepository.findByActiveTrue();
        int scanned = 0, updated = 0;

        for (PharmacyNode pharmacy : pharmacies) {
            try {
                log.info("📡 Scanning: {}", pharmacy.getName());
                List<PharmacyInventoryDTO> inventory = dataSourceService.readInventory(pharmacy);

                if (inventory.isEmpty()) {
                    log.warn("⚠️ No data from {}", pharmacy.getName());
                    continue;
                }

                for (PharmacyInventoryDTO item : inventory) {
                    upsertCacheEntry(pharmacy, item);
                    updated++;
                }

                checkAlertsForPharmacy(pharmacy, inventory);
                scanned++;

            } catch (Exception e) {
                log.error("❌ Error scanning {}: {}", pharmacy.getName(), e.getMessage());
            }
        }

        log.info("✅ FULL SCAN DONE — {} pharmacies, {} items", scanned, updated);
        log.info("═══════════════════════════════════════");
    }

    // =========================================================
    // SMART SCAN — every 1 minute
    // Only re-checks items whose nextCheckTime has passed
    // (low stock items get checked more frequently)
    // =========================================================
    @Scheduled(fixedDelay = 60_000)
    @Transactional
    public void smartScan() {
        if (!enabled) return;

        List<PharmacyInventoryCache> due = cacheRepository.findByNextCheckTimeBefore(LocalDateTime.now());
        if (due.isEmpty()) return;

        log.info("🔍 SMART SCAN: {} items due for refresh", due.size());

        // Group by pharmacy to batch reads
        Map<Long, List<PharmacyInventoryCache>> byPharmacy = due.stream()
                .collect(Collectors.groupingBy(PharmacyInventoryCache::getPharmacyNodeId));

        int totalUpdated = 0;
        for (Map.Entry<Long, List<PharmacyInventoryCache>> entry : byPharmacy.entrySet()) {
            PharmacyNode pharmacy = pharmacyNodeRepository.findById(entry.getKey()).orElse(null);
            if (pharmacy == null || !pharmacy.isActive()) continue;

            try {
                List<PharmacyInventoryDTO> inventory = dataSourceService.readInventory(pharmacy);
                if (inventory.isEmpty()) continue;

                for (PharmacyInventoryCache cached : entry.getValue()) {
                    inventory.stream()
                            .filter(i -> i.getMedicationName() != null &&
                                    i.getMedicationName().equalsIgnoreCase(cached.getMedicationName()))
                            .findFirst()
                            .ifPresent(item -> {
                                upsertCacheEntry(pharmacy, item);
                                checkSingleAlert(pharmacy, item);
                            });
                    totalUpdated++;
                }
            } catch (Exception e) {
                log.error("❌ Smart scan error for {}: {}", pharmacy.getName(), e.getMessage());
            }
        }
        log.info("✅ SMART SCAN DONE — {} items refreshed", totalUpdated);
    }

    // Manual trigger (used by TestController)
    public void triggerSmartScan() {
        smartScan();
    }

    // =========================================================
    // CACHE UPSERT — shared logic, no delete-then-insert gap
    // =========================================================
    @Transactional
    protected void upsertCacheEntry(PharmacyNode pharmacy, PharmacyInventoryDTO item) {
        if (item.getMedicationName() == null || item.getMedicationName().isBlank()) return;

        LocalDateTime now = LocalDateTime.now();

        List<PharmacyInventoryCache> existing = cacheRepository
                .findByPharmacyNodeIdAndMedicationNameIgnoreCase(
                        pharmacy.getId(), item.getMedicationName());

        PharmacyInventoryCache entry = existing.isEmpty()
                ? new PharmacyInventoryCache() : existing.get(0);

        // Clean duplicates if any
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
    }

    // =========================================================
    // ALERT CHECKING
    // =========================================================
    private void checkAlertsForPharmacy(PharmacyNode pharmacy, List<PharmacyInventoryDTO> inventory) {
        for (PharmacyInventoryDTO item : inventory) {
            if (!item.isAvailable() || item.getStockQuantity() <= 0) continue;
            checkSingleAlert(pharmacy, item);
        }
    }

    private void checkSingleAlert(PharmacyNode pharmacy, PharmacyInventoryDTO item) {
        List<StockAlert> alerts = alertRepository
                .findByMedicationNameIgnoreCaseAndActiveTrue(item.getMedicationName());

        for (StockAlert alert : alerts) {
            if (alert.getMaxPrice() != null && item.getPrice() > alert.getMaxPrice()) continue;
            if (notificationService.canNotify(alert)) {
                notificationService.sendRestockNotification(alert, item, pharmacy);
                alert.setLastNotified(LocalDateTime.now());
                alertRepository.save(alert);
            }
        }
    }

    // =========================================================
    // HELPERS
    // =========================================================
    private String priority(int qty) {
        if (qty <= 0)  return "OUT_OF_STOCK";
        if (qty <= 5)  return "HIGH";
        if (qty <= 20) return "MEDIUM";
        return "LOW";
    }

    private LocalDateTime nextCheck(LocalDateTime now, int qty) {
        if (qty <= 0)  return now.plusMinutes(10);
        if (qty <= 5)  return now.plusMinutes(3);
        if (qty <= 20) return now.plusMinutes(6);
        return now.plusMinutes(10);
    }
}