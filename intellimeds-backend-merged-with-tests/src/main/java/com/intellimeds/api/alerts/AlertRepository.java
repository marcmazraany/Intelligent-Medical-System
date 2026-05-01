package com.intellimeds.api.alerts;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface AlertRepository extends JpaRepository<AlertEntity, UUID> {

    List<AlertEntity> findAllByUserIdOrderByCreatedDateDesc(UUID userId);

    Optional<AlertEntity> findByIdAndUserId(UUID id, UUID userId);

    void deleteByIdAndUserId(UUID id, UUID userId);

    // Used by PharmacyAlertNotifier scheduled job
    @Query("SELECT a FROM AlertEntity a WHERE a.active = true AND a.emailEnabled = true")
    List<AlertEntity> findAllActiveAlerts();
}
