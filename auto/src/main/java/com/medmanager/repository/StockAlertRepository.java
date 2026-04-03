package com.medmanager.repository;

import com.medmanager.entity.StockAlert;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface StockAlertRepository extends JpaRepository<StockAlert, Long> {

    List<StockAlert> findByUserEmailAndActiveTrue(String email);

    @Query("SELECT a FROM StockAlert a WHERE LOWER(a.medicationName) = LOWER(:name) AND a.active = true")
    List<StockAlert> findByMedicationNameIgnoreCaseAndActiveTrue(@Param("name") String name);

    // Legacy
    List<StockAlert> findByMedicationIdAndActiveTrue(Long medicationId);
}