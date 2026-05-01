package com.intellimeds.api.alerts;

import com.intellimeds.api.alerts.dto.*;
import com.intellimeds.api.common.NotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

@Service
public class AlertService {

    private final AlertRepository repo;

    public AlertService(AlertRepository repo) {
        this.repo = repo;
    }

    @Transactional(readOnly = true)
    public List<AlertDto> list(UUID userId) {
        return repo.findAllByUserIdOrderByCreatedDateDesc(userId).stream().map(AlertService::toDto).toList();
    }

    @Transactional
    public AlertDto create(UUID userId, CreateAlertRequest req) {
        AlertEntity e = AlertEntity.builder()
                .userId(userId)
                .medicationName(req.medicationName())
                .maxPrice(req.maxPrice())
                .emailEnabled(req.emailEnabled())
                .status("active")
                .active(true)
                .build();
        return toDto(repo.save(e));
    }

    @Transactional
    public AlertDto patch(UUID userId, UUID id, PatchAlertRequest req) {
        AlertEntity e = repo.findByIdAndUserId(id, userId)
                .orElseThrow(() -> new NotFoundException("Alert not found"));
        if (req.active() != null) e.setActive(req.active());
        if (req.emailEnabled() != null) e.setEmailEnabled(req.emailEnabled());
        if (req.maxPrice() != null) e.setMaxPrice(req.maxPrice());
        if (req.status() != null) e.setStatus(req.status());
        // do not change lastNotified here; that would be done by notifier job
        return toDto(repo.save(e));
    }

    @Transactional
    public void delete(UUID userId, UUID id) {
        repo.findByIdAndUserId(id, userId).orElseThrow(() -> new NotFoundException("Alert not found"));
        repo.deleteByIdAndUserId(id, userId);
    }

    private static AlertDto toDto(AlertEntity e) {
        return AlertDto.builder()
                .id(e.getId().toString())
                .medicationName(e.getMedicationName())
                .maxPrice(e.getMaxPrice())
                .emailEnabled(e.isEmailEnabled())
                .createdDate(e.getCreatedDate() == null ? null : e.getCreatedDate().toString())
                .lastNotified(e.getLastNotified() == null ? null : e.getLastNotified().toString())
                .status(e.getStatus())
                .active(e.isActive())
                .build();
    }
}
