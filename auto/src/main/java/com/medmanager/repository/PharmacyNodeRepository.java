package com.medmanager.repository;

import com.medmanager.entity.PharmacyNode;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface PharmacyNodeRepository extends JpaRepository<PharmacyNode, Long> {

    // All active pharmacies (used by scanner and ping)
    List<PharmacyNode> findByActiveTrue();

    // Only pharmacies that support live on-demand pinging
    List<PharmacyNode> findByActiveTrueAndSupportsLivePingTrue();
}