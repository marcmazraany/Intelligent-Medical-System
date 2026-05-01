package com.medmanager.service;

import com.google.maps.DistanceMatrixApi;
import com.google.maps.GeoApiContext;
import com.google.maps.model.*;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
@Slf4j
public class GoogleMapsDistanceService {

    @Value("${google.maps.api.key}")
    private String apiKey;

    @Value("${google.maps.enabled:true}")
    private boolean enabled;

    private GeoApiContext context;
    private final Map<String, DistanceCache> cache = new ConcurrentHashMap<>();

    @PostConstruct
    public void init() {
        if (enabled && apiKey != null && !apiKey.equals("YOUR_API_KEY_HERE")) {
            context = new GeoApiContext.Builder().apiKey(apiKey).build();
            log.info("✅ Google Maps API initialised");
        } else {
            log.warn("⚠️ Google Maps not configured — using straight-line distance");
        }
    }

    public DistanceResult calculateDistance(double fromLat, double fromLon,
                                            double toLat, double toLon) {
        String key = String.format("%.4f,%.4f-%.4f,%.4f", fromLat, fromLon, toLat, toLon);
        DistanceCache cached = cache.get(key);
        if (cached != null && !cached.isExpired()) {
            return new DistanceResult(cached.distanceKm, cached.travelTimeMinutes);
        }
        if (context != null) {
            try {
                DistanceResult result = getGoogleMapsDistance(fromLat, fromLon, toLat, toLon);
                cache.put(key, new DistanceCache(result.distanceKm, result.travelTimeMinutes));
                return result;
            } catch (Exception e) {
                log.warn("Google Maps failed, falling back to straight-line: {}", e.getMessage());
            }
        }
        return new DistanceResult(haversineKm(fromLat, fromLon, toLat, toLon), null);
    }

    // Public so PharmacyPingService can use it for cheap pre-filtering
    public double haversineKm(double lat1, double lon1, double lat2, double lon2) {
        final double R = 6371;
        double dLat = Math.toRadians(lat2 - lat1);
        double dLon = Math.toRadians(lon2 - lon1);
        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2)
                + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
                * Math.sin(dLon / 2) * Math.sin(dLon / 2);
        return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    private DistanceResult getGoogleMapsDistance(double fromLat, double fromLon,
                                                 double toLat, double toLon) throws Exception {
        DistanceMatrix matrix = DistanceMatrixApi.newRequest(context)
                .origins(fromLat + "," + fromLon)
                .destinations(toLat + "," + toLon)
                .mode(TravelMode.DRIVING)
                .units(Unit.METRIC)
                .await();
        if (matrix.rows.length > 0 && matrix.rows[0].elements.length > 0) {
            DistanceMatrixElement el = matrix.rows[0].elements[0];
            if (el.distance != null && el.duration != null) {
                return new DistanceResult(el.distance.inMeters / 1000.0,
                        (int) (el.duration.inSeconds / 60));
            }
        }
        throw new Exception("No data from Google Maps");
    }

    public static class DistanceResult {
        public final double distanceKm;
        public final Integer travelTimeMinutes;
        public DistanceResult(double distanceKm, Integer travelTimeMinutes) {
            this.distanceKm = distanceKm;
            this.travelTimeMinutes = travelTimeMinutes;
        }
    }

    private static class DistanceCache {
        final double distanceKm;
        final Integer travelTimeMinutes;
        final long timestamp;
        DistanceCache(double d, Integer t) {
            this.distanceKm = d; this.travelTimeMinutes = t;
            this.timestamp = System.currentTimeMillis();
        }
        boolean isExpired() { return System.currentTimeMillis() - timestamp > 3_600_000; }
    }
}