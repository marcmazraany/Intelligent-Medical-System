package com.intellimeds.api.pharmacy.service;

import com.intellimeds.api.alerts.AlertEntity;
import com.intellimeds.api.alerts.AlertRepository;
import com.intellimeds.api.pharmacy.config.PharmacyFinderProperties;
import com.intellimeds.api.pharmacy.dto.MedicationAvailabilityResponse;
import com.intellimeds.api.pharmacy.dto.PharmacySearchResult;
import com.intellimeds.api.users.UserEntity;
import com.intellimeds.api.users.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;

/**
 * Runs every 5 minutes.
 * For every active alert in Supabase it asks the auto system if that
 * medication is currently in stock, then emails the user if it is.
 *
 * The auto system never touches IntelliMeds' DB.
 * IntelliMeds never touches the auto system's DB.
 * They communicate only via HTTP.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class PharmacyAlertNotifier {

    private final AlertRepository          alertRepository;
    private final UserRepository           userRepository;
    private final PharmacyFinderClient     pharmacyClient;
    private final PharmacyFinderProperties props;
    private final JavaMailSender           mailSender;

    // ── Run every 5 minutes ───────────────────────────────────────────
    @Scheduled(fixedDelay = 300_000)
    @Transactional
    public void checkAlerts() {
        List<AlertEntity> activeAlerts = alertRepository.findAllActiveAlerts();

        if (activeAlerts.isEmpty()) return;

        log.info("🔔 Checking {} active pharmacy alerts", activeAlerts.size());

        for (AlertEntity alert : activeAlerts) {
            if (!canNotify(alert)) continue;

            try {
                checkSingleAlert(alert);
            } catch (Exception e) {
                log.error("❌ Error checking alert {} for '{}': {}",
                        alert.getId(), alert.getMedicationName(), e.getMessage());
            }
        }
    }

    // ── Check one alert ───────────────────────────────────────────────
    private void checkSingleAlert(AlertEntity alert) {
        MedicationAvailabilityResponse result = pharmacyClient.search(
                alert.getMedicationName(),
                props.defaultLatitude(),
                props.defaultLongitude(),
                5   // top 5 nearby pharmacies is enough
        );

        if (result.getPharmacies() == null || result.getPharmacies().isEmpty()) return;

        // Find a pharmacy that meets the price constraint (if any)
        PharmacySearchResult match = result.getPharmacies().stream()
                .filter(PharmacySearchResult::isInStock)
                .filter(p -> alert.getMaxPrice() == null
                        || BigDecimal.valueOf(p.getPrice()).compareTo(alert.getMaxPrice()) <= 0)
                .filter(p -> alert.getMaxDistance() == null
                        || p.getDistanceKm() <= alert.getMaxDistance())
                .findFirst()
                .orElse(null);

        if (match == null) return;

        // Resolve user email from Supabase
        if (!alert.isEmailEnabled()) return;

        userRepository.findById(alert.getUserId()).ifPresent(user -> {
            sendEmail(user, alert, match);
            alert.setLastNotified(Instant.now());
            alertRepository.save(alert);
            log.info("📧 Notified {} — '{}' in stock at {}",
                    user.getEmail(), alert.getMedicationName(), match.getPharmacyName());
        });
    }

    // ── Throttle: at most once per hour per alert ─────────────────────
    private boolean canNotify(AlertEntity alert) {
        if (alert.getLastNotified() == null) return true;
        return alert.getLastNotified().isBefore(Instant.now().minusSeconds(3_600));
    }

    // ── Email ─────────────────────────────────────────────────────────
    private void sendEmail(UserEntity user, AlertEntity alert, PharmacySearchResult pharmacy) {
        try {
            SimpleMailMessage msg = new SimpleMailMessage();
            msg.setTo(user.getEmail());
            msg.setSubject(alert.getMedicationName() + " is in stock near you — IntelliMeds");
            msg.setText("""
                    Hi,

                    Good news! %s is currently in stock at a nearby pharmacy.

                    Pharmacy  : %s
                    Address   : %s
                    Distance  : %.1f km away
                    Stock     : %d units
                    Price     : %.0f %s
                    Phone     : %s
                    Maps link : %s

                    Open IntelliMeds to see all available pharmacies nearby.

                    — The IntelliMeds Team
                    """.formatted(
                    alert.getMedicationName(),
                    pharmacy.getPharmacyName(),
                    pharmacy.getAddress()       != null ? pharmacy.getAddress()       : "N/A",
                    pharmacy.getDistanceKm(),
                    pharmacy.getStockQuantity(),
                    pharmacy.getPrice(),
                    pharmacy.getCurrency()      != null ? pharmacy.getCurrency()      : "LBP",
                    pharmacy.getPharmacyPhone() != null ? pharmacy.getPharmacyPhone() : "N/A",
                    pharmacy.getGoogleMapsUrl() != null ? pharmacy.getGoogleMapsUrl() : ""
            ));
            mailSender.send(msg);
        } catch (Exception e) {
            log.error("❌ Email failed to {}: {}", user.getEmail(), e.getMessage());
        }
    }
}
