package com.medmanager.repository;

import com.medmanager.entity.PharmacyInventoryCache;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface PharmacyInventoryCacheRepository extends JpaRepository<PharmacyInventoryCache, Long> {

    // Exact name match (used for search)
    @Query("SELECT c FROM PharmacyInventoryCache c WHERE LOWER(c.medicationName) = LOWER(:name) AND c.available = true")
    List<PharmacyInventoryCache> findByMedicationNameIgnoreCaseAndAvailableTrue(@Param("name") String name);

    // Partial name match (fallback search)
    @Query("SELECT c FROM PharmacyInventoryCache c WHERE LOWER(c.medicationName) LIKE LOWER(CONCAT('%',:name,'%')) AND c.available = true")
    List<PharmacyInventoryCache> findByMedicationNameContainingIgnoreCaseAndAvailableTrue(@Param("name") String name);

    // Find one entry for a specific pharmacy + medication (for upsert)
    @Query("SELECT c FROM PharmacyInventoryCache c WHERE c.pharmacyNodeId = :pharmacyNodeId AND LOWER(c.medicationName) = LOWER(:name)")
    List<PharmacyInventoryCache> findByPharmacyNodeIdAndMedicationNameIgnoreCase(
            @Param("pharmacyNodeId") Long pharmacyNodeId,
            @Param("name") String name);

    // Items due for a priority re-check (used by SmartPharmacyScanner)
    List<PharmacyInventoryCache> findByNextCheckTimeBefore(LocalDateTime time);

    // All entries for a pharmacy (used by full scan)
    List<PharmacyInventoryCache> findByPharmacyNodeId(Long pharmacyNodeId);

    // Legacy — kept for backward compatibility
    List<PharmacyInventoryCache> findByPharmacyNodeIdAndMedicationId(Long pharmacyNodeId, Long medicationId);
    List<PharmacyInventoryCache> findByMedicationIdAndAvailableTrue(Long medicationId);

    void deleteByPharmacyNodeId(Long pharmacyNodeId);
}