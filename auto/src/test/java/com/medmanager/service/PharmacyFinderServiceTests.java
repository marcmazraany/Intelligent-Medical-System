package com.medmanager.service;

import com.medmanager.dto.PharmacyInventoryDTO;
import com.medmanager.dto.PharmacySearchResult;
import com.medmanager.entity.PharmacyInventoryCache;
import com.medmanager.entity.PharmacyNode;
import com.medmanager.enums.PharmacyDataSourceType;
import com.medmanager.repository.PharmacyInventoryCacheRepository;
import com.medmanager.repository.PharmacyNodeRepository;
import com.medmanager.repository.StockAlertRepository;
import com.medmanager.service.GoogleMapsDistanceService.DistanceResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.LocalDateTime;
import java.util.List;
import java.util.concurrent.ExecutorService;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("SmartPharmacyScanner Unit Tests")
class SmartPharmacyScannerTest {

    @Mock
    private PharmacyNodeRepository pharmacyNodeRepository;

    @Mock
    private PharmacyInventoryCacheRepository cacheRepository;

    @Mock
    private PharmacyDataSourceService dataSourceService;

    @Mock
    private PharmacyPingService pingService;

    @Mock
    private StockAlertRepository alertRepository;

    @Mock
    private NotificationService notificationService;

    @InjectMocks
    private SmartPharmacyScanner scanner;

    @BeforeEach
    void enableScanner() {
        ReflectionTestUtils.setField(scanner, "enabled", true);
    }

    private PharmacyNode buildNode(Long id, boolean active, boolean livePing) {
        PharmacyNode node = new PharmacyNode();
        node.setId(id);
        node.setName("Test Pharmacy " + id);
        node.setLatitude(33.8886);
        node.setLongitude(35.4955);
        node.setActive(active);
        node.setSupportsLivePing(livePing);
        node.setDataSourceType(PharmacyDataSourceType.REST_API);
        return node;
    }

    private PharmacyInventoryDTO buildInventoryItem(String medicationName, int quantity, double price) {
        PharmacyInventoryDTO item = new PharmacyInventoryDTO();
        item.setMedicationName(medicationName);
        item.setStockQuantity(quantity);
        item.setPrice(price);
        item.setCurrency("LBP");
        item.setAvailable(quantity > 0);
        return item;
    }

    @Test
    @DisplayName("fullInventoryScan: only asks repository for active pharmacies")
    void fullInventoryScan_onlyQueriesActivePharmacies() {
        PharmacyNode activeNode = buildNode(1L, true, true);
        PharmacyNode inactiveNode = buildNode(2L, false, true);

        when(pharmacyNodeRepository.findByActiveTrue()).thenReturn(List.of(activeNode));
        when(dataSourceService.readInventory(activeNode)).thenReturn(List.of());

        scanner.fullInventoryScan();

        verify(dataSourceService).readInventory(activeNode);
        verify(dataSourceService, never()).readInventory(inactiveNode);
    }

    @Test
    @DisplayName("fullInventoryScan: upserts inventory items to cache")
    void fullInventoryScan_upsertsInventoryToCache() {
        PharmacyNode node = buildNode(1L, true, true);
        PharmacyInventoryDTO item = buildInventoryItem("Panadol", 10, 25000.0);

        when(pharmacyNodeRepository.findByActiveTrue()).thenReturn(List.of(node));
        when(dataSourceService.readInventory(node)).thenReturn(List.of(item));
        when(cacheRepository.findByPharmacyNodeIdAndMedicationNameIgnoreCase(1L, "Panadol"))
                .thenReturn(List.of());
        when(alertRepository.findByMedicationNameIgnoreCaseAndActiveTrue("Panadol"))
                .thenReturn(List.of());
        when(cacheRepository.save(any(PharmacyInventoryCache.class))).thenAnswer(inv -> inv.getArgument(0));

        scanner.fullInventoryScan();

        ArgumentCaptor<PharmacyInventoryCache> cacheCaptor = ArgumentCaptor.forClass(PharmacyInventoryCache.class);
        verify(cacheRepository, atLeastOnce()).save(cacheCaptor.capture());

        PharmacyInventoryCache saved = cacheCaptor.getValue();
        assertThat(saved.getMedicationName()).isEqualTo("Panadol");
        assertThat(saved.getPharmacyNodeId()).isEqualTo(1L);
        assertThat(saved.getPriority()).isEqualTo("MEDIUM");
        assertThat(saved.isAvailable()).isTrue();
    }

    @Test
    @DisplayName("fullInventoryScan: saves HIGH priority and short next-check time for low stock")
    void fullInventoryScan_setsHighPriorityForLowStock() {
        PharmacyNode node = buildNode(1L, true, true);
        PharmacyInventoryDTO item = buildInventoryItem("Insulin", 3, 100000.0);

        when(pharmacyNodeRepository.findByActiveTrue()).thenReturn(List.of(node));
        when(dataSourceService.readInventory(node)).thenReturn(List.of(item));
        when(cacheRepository.findByPharmacyNodeIdAndMedicationNameIgnoreCase(1L, "Insulin"))
                .thenReturn(List.of());
        when(alertRepository.findByMedicationNameIgnoreCaseAndActiveTrue("Insulin"))
                .thenReturn(List.of());
        when(cacheRepository.save(any(PharmacyInventoryCache.class))).thenAnswer(inv -> inv.getArgument(0));

        scanner.fullInventoryScan();

        ArgumentCaptor<PharmacyInventoryCache> cacheCaptor = ArgumentCaptor.forClass(PharmacyInventoryCache.class);
        verify(cacheRepository).save(cacheCaptor.capture());

        PharmacyInventoryCache saved = cacheCaptor.getValue();
        assertThat(saved.getPriority()).isEqualTo("HIGH");
        assertThat(saved.getNextCheckTime()).isAfter(saved.getLastUpdated());
        assertThat(saved.getNextCheckTime()).isBeforeOrEqualTo(saved.getLastUpdated().plusMinutes(3).plusSeconds(1));
    }
}

@ExtendWith(MockitoExtension.class)
@DisplayName("PharmacyPingService Unit Tests")
class PharmacyPingServiceTest {

    @Mock
    private PharmacyNodeRepository pharmacyNodeRepository;

    @Mock
    private PharmacyInventoryCacheRepository cacheRepository;

    @Mock
    private GoogleMapsDistanceService distanceService;

    @Mock
    private PharmacyDataSourceService dataSourceService;

    @InjectMocks
    private PharmacyPingService pingService;

    @AfterEach
    void shutDownPingPool() {
        ExecutorService pingPool = (ExecutorService) ReflectionTestUtils.getField(pingService, "pingPool");
        if (pingPool != null) {
            pingPool.shutdownNow();
        }
    }

    private PharmacyNode buildNode(Long id, String name, double lat, double lon) {
        PharmacyNode node = new PharmacyNode();
        node.setId(id);
        node.setName(name);
        node.setLatitude(lat);
        node.setLongitude(lon);
        node.setActive(true);
        node.setSupportsLivePing(true);
        node.setDataSourceType(PharmacyDataSourceType.REST_API);
        return node;
    }

    private PharmacyInventoryDTO buildInventoryItem(String medicationName, int quantity, double price) {
        PharmacyInventoryDTO item = new PharmacyInventoryDTO();
        item.setMedicationName(medicationName);
        item.setStockQuantity(quantity);
        item.setPrice(price);
        item.setCurrency("LBP");
        item.setAvailable(quantity > 0);
        return item;
    }

    @Test
    @DisplayName("pingAllPharmaciesByName: excludes pharmacies beyond 1.3× radius before reading inventory")
    void pingAllPharmaciesByName_excludesFarPharmacies() {
        PharmacyNode farNode = buildNode(1L, "Far Pharmacy", 34.0000, 35.6000);

        when(pharmacyNodeRepository.findByActiveTrueAndSupportsLivePingTrue()).thenReturn(List.of(farNode));
        when(distanceService.haversineKm(anyDouble(), anyDouble(), eq(34.0000), eq(35.6000)))
                .thenReturn(10.0);

        List<PharmacySearchResult> results = pingService.pingAllPharmaciesByName(
                "Panadol", 33.8886, 35.4955, 5.0);

        verify(dataSourceService, never()).readInventory(farNode);
        assertThat(results).isEmpty();
    }

    @Test
    @DisplayName("pingAllPharmaciesByName: returns matching in-stock medications sorted by distance")
    void pingAllPharmaciesByName_returnsMatchingMedication() {
        PharmacyNode nearNode = buildNode(1L, "Near Pharmacy", 33.8890, 35.4960);
        PharmacyInventoryDTO item = buildInventoryItem("Panadol Extra", 12, 35000.0);

        when(pharmacyNodeRepository.findByActiveTrueAndSupportsLivePingTrue()).thenReturn(List.of(nearNode));
        when(distanceService.haversineKm(anyDouble(), anyDouble(), eq(33.8890), eq(35.4960)))
                .thenReturn(0.1);
        when(dataSourceService.readInventory(nearNode)).thenReturn(List.of(item));
        when(distanceService.calculateDistance(anyDouble(), anyDouble(), eq(33.8890), eq(35.4960)))
                .thenReturn(new DistanceResult(0.2, 2));

        List<PharmacySearchResult> results = pingService.pingAllPharmaciesByName(
                "Panadol", 33.8886, 35.4955, 5.0);

        assertThat(results).hasSize(1);
        assertThat(results.get(0).getPharmacyName()).isEqualTo("Near Pharmacy");
        assertThat(results.get(0).getStockQuantity()).isEqualTo(12);
        assertThat(results.get(0).isInStock()).isTrue();
    }

    @Test
    @DisplayName("updateCacheAfterPingByName: saves matching medication from pingable pharmacies")
    void updateCacheAfterPingByName_savesMatchingMedication() {
        PharmacyNode node = buildNode(1L, "Live Pharmacy", 33.8890, 35.4960);
        PharmacyInventoryDTO matchingItem = buildInventoryItem("Panadol", 20, 30000.0);
        PharmacyInventoryDTO otherItem = buildInventoryItem("Aspirin", 15, 25000.0);

        when(pharmacyNodeRepository.findByActiveTrueAndSupportsLivePingTrue()).thenReturn(List.of(node));
        when(dataSourceService.readInventory(node)).thenReturn(List.of(matchingItem, otherItem));
        when(cacheRepository.findByPharmacyNodeIdAndMedicationNameIgnoreCase(1L, "Panadol"))
                .thenReturn(List.of());
        when(cacheRepository.save(any(PharmacyInventoryCache.class))).thenAnswer(inv -> inv.getArgument(0));

        pingService.updateCacheAfterPingByName("Panadol");

        ArgumentCaptor<PharmacyInventoryCache> cacheCaptor = ArgumentCaptor.forClass(PharmacyInventoryCache.class);
        verify(cacheRepository).save(cacheCaptor.capture());
        assertThat(cacheCaptor.getValue().getMedicationName()).isEqualTo("Panadol");
        assertThat(cacheCaptor.getValue().getPriority()).isEqualTo("MEDIUM");
    }
}

@ExtendWith(MockitoExtension.class)
@DisplayName("GoogleMapsDistanceService Unit Tests")
class GoogleMapsDistanceServiceTest {

    private GoogleMapsDistanceService distanceService;

    @BeforeEach
    void setUp() {
        distanceService = new GoogleMapsDistanceService();
        ReflectionTestUtils.setField(distanceService, "enabled", false);
        ReflectionTestUtils.setField(distanceService, "apiKey", "YOUR_API_KEY_HERE");
        distanceService.init();
    }

    @Test
    @DisplayName("haversineKm: returns zero for identical coordinates")
    void haversineKm_zeroForSamePoint() {
        double dist = distanceService.haversineKm(33.8886, 35.4955, 33.8886, 35.4955);
        assertThat(dist).isLessThan(0.001);
    }

    @Test
    @DisplayName("haversineKm: Jounieh to Beirut is approximately 14 km")
    void haversineKm_jouniehToBeirut() {
        double dist = distanceService.haversineKm(33.9810, 35.6172, 33.8938, 35.5018);
        assertThat(dist).isBetween(10.0, 18.0);
    }

    @Test
    @DisplayName("haversineKm: is symmetric")
    void haversineKm_isSymmetric() {
        double ab = distanceService.haversineKm(33.9810, 35.6172, 33.8938, 35.5018);
        double ba = distanceService.haversineKm(33.8938, 35.5018, 33.9810, 35.6172);
        assertThat(ab).isCloseTo(ba, org.assertj.core.data.Offset.offset(0.001));
    }

    @Test
    @DisplayName("calculateDistance: falls back to Haversine when Google Maps is disabled")
    void calculateDistance_fallsBackToHaversineWhenGoogleMapsDisabled() {
        DistanceResult result = distanceService.calculateDistance(33.8886, 35.4955, 33.9810, 35.6172);

        assertThat(result).isNotNull();
        assertThat(result.distanceKm).isGreaterThan(0);
        assertThat(result.travelTimeMinutes).isNull();
    }
}
