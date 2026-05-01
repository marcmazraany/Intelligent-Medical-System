package com.intellimeds.api.pharmacy.service;

import com.intellimeds.api.alerts.AlertEntity;
import com.intellimeds.api.alerts.AlertRepository;
import com.intellimeds.api.pharmacy.config.PharmacyFinderProperties;
import com.intellimeds.api.pharmacy.dto.MedicationAvailabilityResponse;
import com.intellimeds.api.pharmacy.dto.PharmacySearchResult;
import com.intellimeds.api.users.UserEntity;
import com.intellimeds.api.users.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("Pharmacy Alerts – PharmacyAlertNotifier Unit Tests")
class PharmacyAlertNotifierTest {

    @Mock private AlertRepository alertRepository;
    @Mock private UserRepository userRepository;
    @Mock private PharmacyFinderClient pharmacyFinderClient;
    @Mock private JavaMailSender mailSender;

    private PharmacyAlertNotifier notifier;

    private static final UUID USER_ID = UUID.fromString("11111111-1111-1111-1111-111111111111");

    @BeforeEach
    void setUp() {
        PharmacyFinderProperties props = new PharmacyFinderProperties("http://localhost:8090", 33.8886, 35.4955);
        notifier = new PharmacyAlertNotifier(alertRepository, userRepository, pharmacyFinderClient, props, mailSender);
    }

    private AlertEntity alert(String medName, BigDecimal maxPrice, boolean emailEnabled, Instant lastNotified) {
        AlertEntity alert = new AlertEntity();
        alert.setId(UUID.randomUUID());
        alert.setUserId(USER_ID);
        alert.setMedicationName(medName);
        alert.setMaxPrice(maxPrice);
        alert.setEmailEnabled(emailEnabled);
        alert.setLastNotified(lastNotified);
        alert.setActive(true);
        alert.setStatus("active");
        return alert;
    }

    private UserEntity user() {
        UserEntity user = new UserEntity();
        user.setId(USER_ID);
        user.setEmail("jana@intellimeds.com");
        return user;
    }

    private PharmacySearchResult pharmacy(boolean inStock, double price, double distanceKm) {
        PharmacySearchResult pharmacy = new PharmacySearchResult();
        pharmacy.setPharmacyName("Nearby Pharmacy");
        pharmacy.setAddress("Beirut");
        pharmacy.setInStock(inStock);
        pharmacy.setPrice(price);
        pharmacy.setDistanceKm(distanceKm);
        pharmacy.setStockQuantity(inStock ? 5 : 0);
        pharmacy.setCurrency("LBP");
        pharmacy.setPharmacyPhone("+9611000000");
        pharmacy.setGoogleMapsUrl("https://maps.example/pharmacy");
        return pharmacy;
    }

    private MedicationAvailabilityResponse responseWith(PharmacySearchResult... pharmacies) {
        MedicationAvailabilityResponse response = new MedicationAvailabilityResponse();
        response.setMedicationName("Panadol");
        response.setPharmacies(List.of(pharmacies));
        response.setTotalPharmaciesWithStock((int) List.of(pharmacies).stream().filter(PharmacySearchResult::isInStock).count());
        response.setTotalPharmaciesChecked(pharmacies.length);
        return response;
    }

    @Nested
    @DisplayName("checkAlerts")
    class CheckAlerts {
        @Test
        @DisplayName("does not query pharmacies or send email when the alert is throttled")
        void checkAlerts_doesNotNotifyWhenThrottled() {
            AlertEntity recentAlert = alert("Panadol", BigDecimal.valueOf(50_000), true,
                    Instant.now().minusSeconds(1_800));
            when(alertRepository.findAllActiveAlerts()).thenReturn(List.of(recentAlert));

            notifier.checkAlerts();

            verifyNoInteractions(pharmacyFinderClient, userRepository, mailSender);
            verify(alertRepository, never()).save(any(AlertEntity.class));
        }

        @Test
        @DisplayName("sends email and updates lastNotified when stock matches after throttle period")
        void checkAlerts_sendsEmailWhenStockMatches() {
            AlertEntity oldAlert = alert("Panadol", BigDecimal.valueOf(50_000), true,
                    Instant.now().minusSeconds(7_200));
            when(alertRepository.findAllActiveAlerts()).thenReturn(List.of(oldAlert));
            when(pharmacyFinderClient.search(eq("Panadol"), eq(33.8886), eq(35.4955), eq(5)))
                    .thenReturn(responseWith(pharmacy(true, 30_000, 2.5)));
            when(userRepository.findById(USER_ID)).thenReturn(Optional.of(user()));

            notifier.checkAlerts();

            ArgumentCaptor<SimpleMailMessage> messageCaptor = ArgumentCaptor.forClass(SimpleMailMessage.class);
            verify(mailSender).send(messageCaptor.capture());
            assertThat(messageCaptor.getValue().getTo()).containsExactly("jana@intellimeds.com");
            assertThat(messageCaptor.getValue().getSubject()).contains("Panadol is in stock");
            assertThat(oldAlert.getLastNotified()).isNotNull();
            verify(alertRepository).save(oldAlert);
        }

        @Test
        @DisplayName("does not send email when the in-stock pharmacy is above maxPrice")
        void checkAlerts_skipsWhenPriceExceedsMax() {
            AlertEntity priceAlert = alert("Panadol", BigDecimal.valueOf(20_000), true,
                    Instant.now().minusSeconds(7_200));
            when(alertRepository.findAllActiveAlerts()).thenReturn(List.of(priceAlert));
            when(pharmacyFinderClient.search(eq("Panadol"), eq(33.8886), eq(35.4955), eq(5)))
                    .thenReturn(responseWith(pharmacy(true, 50_000, 2.0)));

            notifier.checkAlerts();

            verifyNoInteractions(userRepository, mailSender);
            verify(alertRepository, never()).save(any(AlertEntity.class));
        }

        @Test
        @DisplayName("does not send email when emailEnabled is false")
        void checkAlerts_skipsWhenEmailDisabled() {
            AlertEntity noEmailAlert = alert("Panadol", BigDecimal.valueOf(50_000), false,
                    Instant.now().minusSeconds(7_200));
            when(alertRepository.findAllActiveAlerts()).thenReturn(List.of(noEmailAlert));
            when(pharmacyFinderClient.search(eq("Panadol"), eq(33.8886), eq(35.4955), eq(5)))
                    .thenReturn(responseWith(pharmacy(true, 30_000, 2.0)));

            notifier.checkAlerts();

            verifyNoInteractions(userRepository, mailSender);
            verify(alertRepository, never()).save(any(AlertEntity.class));
        }

        @Test
        @DisplayName("does not send email when medication is out of stock")
        void checkAlerts_skipsWhenOutOfStock() {
            AlertEntity activeAlert = alert("Panadol", BigDecimal.valueOf(50_000), true,
                    Instant.now().minusSeconds(7_200));
            when(alertRepository.findAllActiveAlerts()).thenReturn(List.of(activeAlert));
            when(pharmacyFinderClient.search(eq("Panadol"), eq(33.8886), eq(35.4955), eq(5)))
                    .thenReturn(responseWith(pharmacy(false, 30_000, 2.0)));

            notifier.checkAlerts();

            verifyNoInteractions(userRepository, mailSender);
            verify(alertRepository, never()).save(any(AlertEntity.class));
        }

        @Test
        @DisplayName("does nothing when no active alerts exist")
        void checkAlerts_doesNothingWhenNoActiveAlertsExist() {
            when(alertRepository.findAllActiveAlerts()).thenReturn(List.of());

            notifier.checkAlerts();

            verifyNoInteractions(pharmacyFinderClient, userRepository, mailSender);
        }
    }
}
