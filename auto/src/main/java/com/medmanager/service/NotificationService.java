package com.medmanager.service;

import com.medmanager.dto.PharmacyInventoryDTO;
import com.medmanager.entity.PharmacyNode;
import com.medmanager.entity.StockAlert;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationService {

    private final JavaMailSender mailSender;

    @Value("${spring.mail.username:noreply@medmanager.com}")
    private String fromEmail;

    public boolean canNotify(StockAlert alert) {
        if (alert.getLastNotified() == null) return true;
        return alert.getLastNotified().isBefore(LocalDateTime.now().minusHours(1));
    }

    public void sendRestockNotification(StockAlert alert, PharmacyInventoryDTO item, PharmacyNode pharmacy) {
        String body = String.format(
                "Good news! %s is back in stock.\n\n" +
                        "Pharmacy: %s\n" +
                        "Address: %s\n" +
                        "Stock: %d units\n" +
                        "Price: %.0f %s\n" +
                        "Phone: %s\n\n" +
                        "Search again on MedFinder to see all available pharmacies.",
                alert.getMedicationName(),
                pharmacy.getName(),
                pharmacy.getAddress() != null ? pharmacy.getAddress() : "N/A",
                item.getStockQuantity(),
                item.getPrice(),
                item.getCurrency() != null ? item.getCurrency() : "LBP",
                pharmacy.getPhoneNumber() != null ? pharmacy.getPhoneNumber() : "N/A"
        );

        if (alert.isNotifyByEmail() && alert.getUserEmail() != null) {
            try {
                sendEmail(alert.getUserEmail(),
                        alert.getMedicationName() + " is back in stock — MedFinder",
                        body);
                log.info("📧 Email sent to {} for '{}'", alert.getUserEmail(), alert.getMedicationName());
            } catch (Exception e) {
                log.error("❌ Email failed to {}: {}", alert.getUserEmail(), e.getMessage());
            }
        }

        if (alert.isNotifyBySMS() && alert.getUserPhone() != null) {
            // SMS via Twilio would go here
            log.info("📱 SMS (not yet implemented) to {}", alert.getUserPhone());
        }
    }

    private void sendEmail(String to, String subject, String text) {
        SimpleMailMessage msg = new SimpleMailMessage();
        msg.setFrom(fromEmail);
        msg.setTo(to);
        msg.setSubject(subject);
        msg.setText(text);
        mailSender.send(msg);
    }
}