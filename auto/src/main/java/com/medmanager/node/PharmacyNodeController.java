package com.medmanager.node;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping("/api/pharmacy")
@CrossOrigin(origins = "*")
@Slf4j
public class PharmacyNodeController {

    // In-memory inventory — no database needed on the node side
    private final Map<Long, MedicationInventory> inventory = new ConcurrentHashMap<>();

    @GetMapping("/inventory")
    public ResponseEntity<List<MedicationInventory>> getInventory() {
        log.info("📦 Inventory requested — {} items", inventory.size());
        return ResponseEntity.ok(new ArrayList<>(inventory.values()));
    }

    @PostMapping("/inventory")
    public ResponseEntity<MedicationInventory> addItem(@RequestBody MedicationInventory item) {
        inventory.put(item.getMedicationId(), item);
        log.info("📝 Added: {} — {} units", item.getMedicationName(), item.getStockQuantity());
        return ResponseEntity.ok(item);
    }

    @PostMapping("/inventory/bulk")
    public ResponseEntity<String> bulkAdd(@RequestBody List<MedicationInventory> items) {
        items.forEach(i -> inventory.put(i.getMedicationId(), i));
        log.info("📝 Bulk added {} items", items.size());
        return ResponseEntity.ok("Added " + items.size() + " items");
    }

    @DeleteMapping("/inventory/{id}")
    public ResponseEntity<Void> removeItem(@PathVariable Long id) {
        inventory.remove(id);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("OK — " + inventory.size() + " items in stock");
    }

    // Simple inventory DTO (no Lombok — keeps the node self-contained)
    public static class MedicationInventory {
        private Long medicationId;
        private String medicationName;
        private String dosage;
        private Integer stockQuantity;
        private Double price;
        private String currency;
        private boolean available;

        public Long getMedicationId()              { return medicationId; }
        public void setMedicationId(Long v)        { this.medicationId = v; }
        public String getMedicationName()          { return medicationName; }
        public void setMedicationName(String v)    { this.medicationName = v; }
        public String getDosage()                  { return dosage; }
        public void setDosage(String v)            { this.dosage = v; }
        public Integer getStockQuantity()          { return stockQuantity; }
        public void setStockQuantity(Integer v)    { this.stockQuantity = v; }
        public Double getPrice()                   { return price; }
        public void setPrice(Double v)             { this.price = v; }
        public String getCurrency()                { return currency; }
        public void setCurrency(String v)          { this.currency = v; }
        public boolean isAvailable()               { return available; }
        public void setAvailable(boolean v)        { this.available = v; }
    }
}