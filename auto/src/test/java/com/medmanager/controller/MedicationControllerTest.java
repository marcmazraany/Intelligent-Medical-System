package com.medmanager.controller;

import com.medmanager.config.SimulatedDelayInterceptor;
import com.medmanager.dto.PharmacySearchResult;
import com.medmanager.entity.PharmacyInventoryCache;
import com.medmanager.repository.MedicationRepository;
import com.medmanager.repository.PharmacyInventoryCacheRepository;
import com.medmanager.repository.StockAlertRepository;
import com.medmanager.service.GoogleMapsDistanceService;
import com.medmanager.service.GoogleMapsDistanceService.DistanceResult;
import com.medmanager.service.PharmacyPingService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(controllers = com.medmanager.controller.MedicationController.class)
@TestPropertySource(properties = {
        "user.default.latitude=33.8886",
        "user.default.longitude=35.4955",
        "google.maps.api.key=YOUR_API_KEY_HERE",
        "google.maps.enabled=false"
})
@DisplayName("Pharmacy Finder – MedicationController Endpoint Tests")
class MedicationControllerTest {

    @Autowired
    private MockMvc mvc;

    @MockBean
    private MedicationRepository medicationRepository;

    @MockBean
    private PharmacyInventoryCacheRepository cacheRepository;

    @MockBean
    private StockAlertRepository alertRepository;

    @MockBean
    private PharmacyPingService pingService;

    @MockBean
    private GoogleMapsDistanceService distanceService;

    @MockBean
    private SimulatedDelayInterceptor delayInterceptor;

    @BeforeEach
    void setUp() throws Exception {
        when(distanceService.calculateDistance(anyDouble(), anyDouble(), anyDouble(), anyDouble()))
                .thenReturn(new DistanceResult(0.0, 1));
        when(delayInterceptor.preHandle(
                any(HttpServletRequest.class),
                any(HttpServletResponse.class),
                any(Object.class)))
                .thenReturn(true);
    }

    private PharmacyInventoryCache buildCacheEntry(String medName, boolean available, double price) {
        PharmacyInventoryCache entry = new PharmacyInventoryCache();
        entry.setPharmacyNodeId(1L);
        entry.setPharmacyName("Central Pharmacy");
        entry.setMedicationName(medName);
        entry.setStockQuantity(available ? 20 : 0);
        entry.setPrice(price);
        entry.setCurrency("LBP");
        entry.setAvailable(available);
        entry.setPharmacyLatitude(33.8886);
        entry.setPharmacyLongitude(35.4955);
        entry.setLastUpdated(LocalDateTime.now());
        entry.setPriority(available ? "MEDIUM" : "OUT_OF_STOCK");
        entry.setNextCheckTime(LocalDateTime.now().plusMinutes(6));
        return entry;
    }

    @Nested
    @DisplayName("GET /api/medications/search")
    class SearchEndpoint {

        @Test
        @DisplayName("Returns 200 with results on exact cache hit")
        void search_returns200OnCacheHit() throws Exception {
            when(cacheRepository.findByMedicationNameIgnoreCaseAndAvailableTrue("Panadol"))
                    .thenReturn(List.of(buildCacheEntry("Panadol", true, 30000)));

            mvc.perform(get("/api/medications/search")
                            .param("name", "Panadol")
                            .param("latitude", "33.8886")
                            .param("longitude", "35.4955"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.pharmacies").isArray())
                    .andExpect(jsonPath("$.pharmacies[0].pharmacyName").value("Central Pharmacy"))
                    .andExpect(jsonPath("$.pharmacies[0].price").value(30000.0));
        }

        @Test
        @DisplayName("Returns 200 with an empty pharmacy list when medication is not found")
        void search_returns200EmptyArrayWhenNotFound() throws Exception {
            when(cacheRepository.findByMedicationNameIgnoreCaseAndAvailableTrue(anyString()))
                    .thenReturn(List.of());
            when(cacheRepository.findByMedicationNameContainingIgnoreCaseAndAvailableTrue(anyString()))
                    .thenReturn(List.of());
            when(pingService.pingAllPharmaciesByName(anyString(), anyDouble(), anyDouble(), anyDouble()))
                    .thenReturn(List.of());

            mvc.perform(get("/api/medications/search")
                            .param("name", "VeryRareDrug999")
                            .param("latitude", "33.8886")
                            .param("longitude", "35.4955"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.pharmacies").isArray())
                    .andExpect(jsonPath("$.pharmacies").isEmpty())
                    .andExpect(jsonPath("$.medicationName").value("VeryRareDrug999"));
        }

        @Test
        @DisplayName("Returns 400 when name parameter is missing")
        void search_returns400OnMissingName() throws Exception {
            mvc.perform(get("/api/medications/search"))
                    .andExpect(status().isBadRequest());
        }

        @Test
        @DisplayName("Falls through to live ping on exact and partial cache miss")
        void search_triggersPingOnCacheMiss() throws Exception {
            when(cacheRepository.findByMedicationNameIgnoreCaseAndAvailableTrue(anyString()))
                    .thenReturn(List.of());
            when(cacheRepository.findByMedicationNameContainingIgnoreCaseAndAvailableTrue(anyString()))
                    .thenReturn(List.of());
            when(pingService.pingAllPharmaciesByName(anyString(), anyDouble(), anyDouble(), anyDouble()))
                    .thenReturn(List.of());

            mvc.perform(get("/api/medications/search")
                            .param("name", "ScarceDrug")
                            .param("latitude", "33.8886")
                            .param("longitude", "35.4955"))
                    .andExpect(status().isOk());

            verify(pingService).pingAllPharmaciesByName(eq("ScarceDrug"), anyDouble(), anyDouble(), eq(50.0));
            verify(pingService).updateCacheAfterPingByName("ScarceDrug");
        }

        @Test
        @DisplayName("Partial cache match returns results without live ping")
        void search_partialMatchReturnsWithoutPing() throws Exception {
            when(cacheRepository.findByMedicationNameIgnoreCaseAndAvailableTrue("Panadol Extra"))
                    .thenReturn(List.of());
            when(cacheRepository.findByMedicationNameContainingIgnoreCaseAndAvailableTrue("Panadol Extra"))
                    .thenReturn(List.of(buildCacheEntry("Panadol Extra 500mg", true, 35000)));

            mvc.perform(get("/api/medications/search")
                            .param("name", "Panadol Extra")
                            .param("latitude", "33.8886")
                            .param("longitude", "35.4955"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.pharmacies").isNotEmpty())
                    .andExpect(jsonPath("$.pharmacies[0].pharmacyName").value("Central Pharmacy"));

            verify(pingService, never()).pingAllPharmaciesByName(anyString(), anyDouble(), anyDouble(), anyDouble());
        }

        @Test
        @DisplayName("Response includes totalPharmaciesChecked field")
        void search_responseTotalPharmaciesField() throws Exception {
            when(cacheRepository.findByMedicationNameIgnoreCaseAndAvailableTrue(anyString()))
                    .thenReturn(List.of(buildCacheEntry("Panadol", true, 30000)));

            mvc.perform(get("/api/medications/search")
                            .param("name", "Panadol"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.totalPharmaciesChecked").isNumber())
                    .andExpect(jsonPath("$.pharmaciesWithStock").value(1));
        }
    }

    @Nested
    @DisplayName("POST /api/medications/ping")
    class PingEndpoint {

        @Test
        @DisplayName("Always calls live ping regardless of cache state")
        void ping_alwaysCallsLivePing() throws Exception {
            when(pingService.pingAllPharmaciesByName(anyString(), anyDouble(), anyDouble(), anyDouble()))
                    .thenReturn(List.of());

            mvc.perform(post("/api/medications/ping")
                            .param("name", "Panadol")
                            .param("latitude", "33.8886")
                            .param("longitude", "35.4955"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.medicationName").value("Panadol"));

            verify(pingService).pingAllPharmaciesByName(eq("Panadol"), anyDouble(), anyDouble(), eq(50.0));
        }

        @Test
        @DisplayName("Returns 400 when name parameter is missing")
        void ping_returns400OnMissingName() throws Exception {
            mvc.perform(post("/api/medications/ping"))
                    .andExpect(status().isBadRequest());
        }

        @Test
        @DisplayName("Updates cache after live ping")
        void ping_updatesCacheAfterPing() throws Exception {
            PharmacySearchResult liveResult = new PharmacySearchResult();
            liveResult.setPharmacyName("Live Pharmacy");
            liveResult.setInStock(true);
            liveResult.setPrice(40000);
            liveResult.setStockQuantity(15);

            when(pingService.pingAllPharmaciesByName(anyString(), anyDouble(), anyDouble(), anyDouble()))
                    .thenReturn(List.of(liveResult));

            mvc.perform(post("/api/medications/ping")
                            .param("name", "Panadol")
                            .param("latitude", "33.8886")
                            .param("longitude", "35.4955"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.pharmaciesWithStock[0].pharmacyName").value("Live Pharmacy"));

            verify(pingService).updateCacheAfterPingByName("Panadol");
        }
    }
}
