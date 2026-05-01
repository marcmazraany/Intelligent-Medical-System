package com.medmanager.entity;

import com.medmanager.enums.PharmacyDataSourceType;
import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "pharmacy_nodes")
@Data
@NoArgsConstructor
public class PharmacyNode {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private Double latitude;

    @Column(nullable = false)
    private Double longitude;

    private String address;
    private String phoneNumber;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private PharmacyDataSourceType dataSourceType;

    // ── REST_API ──────────────────────────────────────────────────────
    @Column(length = 512)
    private String apiEndpoint;
    private String host;
    private Integer port;

    // ── Database (POSTGRESQL / MYSQL / Supabase) ──────────────────────
    @Column(length = 512)
    private String databaseUrl;
    private String databaseUsername;
    private String databasePassword;

    // The inventory table name — used when customQuery is null
    // e.g. "pharmacy_inventory", "products", "stock", "medicines"
    private String tableName;

    // Custom SQL query — use this when the pharmacy has a complex schema
    // (multiple tables, joins, aggregations).
    // If set, this is used INSTEAD of "SELECT * FROM tableName".
    // The query result columns must match your col_* mappings below.
    //
    // Example for a pharmacy with separate medicines + inventory tables:
    //   SELECT m.name, m.strength, SUM(i.quantity) as total_quantity, 0.0 as price
    //   FROM inventory i
    //   JOIN medicines m ON i.gtin = m.gtin
    //   GROUP BY m.gtin, m.name, m.strength
    //   HAVING SUM(i.quantity) > 0
    @Column(length = 2048)
    private String customQuery;

    // ── Per-pharmacy column mappings ──────────────────────────────────
    // Every pharmacy's DB has different column names.
    // Fill these in once during onboarding.
    // If null, the service tries a list of common name variants automatically.
    //
    // Onboarding questions:
    //   1. "What is your inventory table name?"       → tableName (or provide customQuery)
    //   2. "Which column is the medication name?"     → colMedicationName
    //   3. "Which column is the stock quantity?"      → colStockQuantity
    //   4. "Which column is the price?"               → colPrice
    //   5. "Which column is the dosage?" (optional)   → colDosage
    //   6. "Which column is the currency?" (optional) → colCurrency
    private String colMedicationName;
    private String colStockQuantity;
    private String colPrice;
    private String colDosage;
    private String colCurrency;

    // ── File (EXCEL_FILE / CSV_FILE) ──────────────────────────────────
    @Column(length = 512)
    private String filePath;

    // ── Behaviour ─────────────────────────────────────────────────────
    // true  = internet-reachable, pinged live on cache miss
    //         (Supabase, public REST, Google Sheets CSV)
    // false = local only, populated by agent (future)
    @Column(nullable = false)
    private boolean supportsLivePing = true;

    @Column(nullable = false)
    private boolean active = true;
}