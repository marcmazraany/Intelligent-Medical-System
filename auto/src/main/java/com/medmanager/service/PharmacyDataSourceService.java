package com.medmanager.service;

import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.drive.Drive;
import com.google.api.services.drive.DriveScopes;
import com.google.auth.http.HttpCredentialsAdapter;
import com.google.auth.oauth2.GoogleCredentials;
import com.medmanager.dto.PharmacyInventoryDTO;
import com.medmanager.entity.PharmacyNode;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.poi.ss.usermodel.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.*;
import java.net.URL;
import java.sql.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Service
@RequiredArgsConstructor
@Slf4j
public class PharmacyDataSourceService {

    private final WebClient webClient;

    @Value("${google.drive.service.account.key:}")
    private String serviceAccountKeyPath;

    private final Map<Long, HikariDataSource> pools = new ConcurrentHashMap<>();
    private Drive driveClient;

    @PostConstruct
    public void init() {
        if (serviceAccountKeyPath == null || serviceAccountKeyPath.isBlank()) {
            log.warn("⚠️ google.drive.service.account.key not set — Google Drive sources disabled");
            return;
        }
        try {
            File keyFile = new File(serviceAccountKeyPath);
            if (!keyFile.exists()) {
                log.warn("⚠️ Service account key file not found: {}", serviceAccountKeyPath);
                return;
            }
            GoogleCredentials credentials = GoogleCredentials
                    .fromStream(new FileInputStream(keyFile))
                    .createScoped(Collections.singleton(DriveScopes.DRIVE_READONLY));

            driveClient = new Drive.Builder(
                    GoogleNetHttpTransport.newTrustedTransport(),
                    GsonFactory.getDefaultInstance(),
                    new HttpCredentialsAdapter(credentials))
                    .setApplicationName("MedFinder")
                    .build();

            log.info("✅ Google Drive client initialised");
        } catch (Exception e) {
            log.error("❌ Failed to initialise Google Drive client: {}", e.getMessage());
        }
    }

    // ── Router ────────────────────────────────────────────────────────
    public List<PharmacyInventoryDTO> readInventory(PharmacyNode pharmacy) {
        return switch (pharmacy.getDataSourceType()) {
            case REST_API           -> readFromRest(pharmacy);
            case POSTGRESQL         -> readFromDb(pharmacy, "org.postgresql.Driver");
            case MYSQL              -> readFromDb(pharmacy, "com.mysql.cj.jdbc.Driver");
            case EXCEL_FILE         -> readFromExcel(pharmacy);
            case CSV_FILE           -> readFromCsv(pharmacy);
            case GOOGLE_DRIVE_EXCEL -> readFromGoogleDrive(pharmacy, false);
            case GOOGLE_DRIVE_CSV   -> readFromGoogleDrive(pharmacy, true);
        };
    }

    // ── REST ──────────────────────────────────────────────────────────
    private List<PharmacyInventoryDTO> readFromRest(PharmacyNode pharmacy) {
        try {
            String url = pharmacy.getApiEndpoint() != null
                    ? pharmacy.getApiEndpoint()
                    : "http://" + pharmacy.getHost() + ":" + pharmacy.getPort()
                    + "/api/pharmacy/inventory";

            List<PharmacyInventoryDTO> result = webClient.get()
                    .uri(url)
                    .retrieve()
                    .bodyToFlux(PharmacyInventoryDTO.class)
                    .timeout(Duration.ofSeconds(5))
                    .collectList()
                    .block();

            int count = result != null ? result.size() : 0;
            log.info("✅ REST: {} items from {}", count, pharmacy.getName());
            return result != null ? result : List.of();

        } catch (Exception e) {
            log.error("❌ REST failed for {}: {}", pharmacy.getName(), e.getMessage());
            return List.of();
        }
    }

    // ── Database (PostgreSQL / Supabase / MySQL) ──────────────────────
    private List<PharmacyInventoryDTO> readFromDb(PharmacyNode pharmacy, String driver) {
        List<PharmacyInventoryDTO> inventory = new ArrayList<>();

        if (pharmacy.getDatabaseUrl() == null || pharmacy.getDatabaseUrl().isBlank()) {
            log.error("❌ No databaseUrl for {}", pharmacy.getName());
            return inventory;
        }

        boolean hasTable = pharmacy.getTableName()  != null && !pharmacy.getTableName().isBlank();
        boolean hasQuery = pharmacy.getCustomQuery() != null && !pharmacy.getCustomQuery().isBlank();
        if (!hasTable && !hasQuery) {
            log.error("❌ {} has no tableName or customQuery", pharmacy.getName());
            return inventory;
        }

        String sql = hasQuery
                ? pharmacy.getCustomQuery()
                : "SELECT * FROM " + pharmacy.getTableName();

        try {
            javax.sql.DataSource ds = getOrCreatePool(pharmacy, driver);

            try (Connection conn = ds.getConnection();
                 Statement stmt = conn.createStatement();
                 ResultSet rs = stmt.executeQuery(sql)) {

                ResultSetMetaData meta = rs.getMetaData();
                Map<String, Integer> cols = new HashMap<>();
                for (int i = 1; i <= meta.getColumnCount(); i++) {
                    cols.put(meta.getColumnLabel(i).toLowerCase(), i);
                }

                while (rs.next()) {
                    try {
                        String name = resolveString(rs, cols, pharmacy.getColMedicationName(),
                                "medication_name","name","drug_name","medicine_name",
                                "product_name","item_name","nom","medicament","drug","product","title");
                        if (name == null || name.isBlank()) continue;

                        int qty = resolveInt(rs, cols, pharmacy.getColStockQuantity(),
                                "stock_quantity","quantity","qty","stock","units","count",
                                "units_in_stock","available_qty","total_quantity","qte","quantite");

                        double price = resolveDouble(rs, cols, pharmacy.getColPrice(),
                                "price","unit_price","cost","selling_price",
                                "retail_price","prix","tarif","sale_price");

                        String dosage   = resolveString(rs, cols, pharmacy.getColDosage(),
                                "dosage","dose","strength","concentration","form");
                        String currency = resolveString(rs, cols, pharmacy.getColCurrency(),
                                "currency","currency_code","devise");

                        PharmacyInventoryDTO dto = new PharmacyInventoryDTO();
                        dto.setMedicationName(name);
                        dto.setStockQuantity(qty);
                        dto.setPrice(price);
                        dto.setDosage(dosage != null ? dosage : "");
                        dto.setCurrency(currency != null && !currency.isBlank() ? currency : "LBP");
                        dto.setAvailable(qty > 0);
                        inventory.add(dto);

                    } catch (Exception e) {
                        log.warn("Skipping row in {}: {}", pharmacy.getName(), e.getMessage());
                    }
                }
            }
            log.info("✅ DB: {} items from {}", inventory.size(), pharmacy.getName());

        } catch (Exception e) {
            log.error("❌ DB read failed for {}: {}", pharmacy.getName(), e.getMessage());
            invalidatePool(pharmacy.getId());
        }
        return inventory;
    }

    // ── Google Drive ──────────────────────────────────────────────────
    // Handles both:
    //   - Regular uploaded files (.xlsx, .csv) → direct download
    //   - Google Sheets files → export as CSV or Excel first
    //
    // filePath stores just the file ID, e.g. 1miAFZJiXMAb9X3hrRlRgMjmHnUipDVnFp8itueK0axk
    // If they paste the full URL by mistake, extractDriveFileId() handles it.
    private List<PharmacyInventoryDTO> readFromGoogleDrive(PharmacyNode pharmacy, boolean isCsv) {
        if (driveClient == null) {
            log.error("❌ Google Drive client not initialised — check service account key path in application.properties");
            return List.of();
        }

        String fileId = pharmacy.getFilePath();
        if (fileId == null || fileId.isBlank()) {
            log.error("❌ No file ID in filePath for {}", pharmacy.getName());
            return List.of();
        }

        // Handle full URL pasted by mistake
        if (fileId.contains("drive.google.com") || fileId.contains("docs.google.com")) {
            fileId = extractDriveFileId(fileId);
        }

        try {
            log.info("📥 Google Drive: downloading {} ({})", pharmacy.getName(), fileId);

            // Check the file's MIME type to determine download strategy
            com.google.api.services.drive.model.File fileMeta = driveClient.files()
                    .get(fileId)
                    .setFields("mimeType, name")
                    .execute();

            String mimeType = fileMeta.getMimeType();
            log.debug("📋 Drive file: {} | mimeType: {}", fileMeta.getName(), mimeType);

            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

            if (mimeType != null && mimeType.contains("spreadsheet")) {
                // Google Sheets file — must be exported, cannot be downloaded directly
                // Export as CSV (simpler) or Excel depending on what was requested
                String exportMime = isCsv
                        ? "text/csv"
                        : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

                log.debug("📤 Exporting Google Sheet as {}", exportMime);
                driveClient.files().export(fileId, exportMime)
                        .executeMediaAndDownloadTo(outputStream);

            } else {
                // Regular uploaded file (.xlsx, .csv) — download directly
                driveClient.files().get(fileId)
                        .executeMediaAndDownloadTo(outputStream);
            }

            InputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());

            // Parse using existing parsers
            List<PharmacyInventoryDTO> result = isCsv
                    ? parseCsvStream(inputStream, pharmacy)
                    : parseExcelStream(inputStream, pharmacy);

            log.info("✅ Google Drive: {} items from {}", result.size(), pharmacy.getName());
            return result;

        } catch (Exception e) {
            log.error("❌ Google Drive failed for {}: {}", pharmacy.getName(), e.getMessage());
            return List.of();
        }
    }

    // Extracts file ID from any Google Drive/Sheets URL format:
    //   https://drive.google.com/file/d/FILE_ID/view
    //   https://docs.google.com/spreadsheets/d/FILE_ID/edit
    //   https://drive.google.com/open?id=FILE_ID
    private String extractDriveFileId(String url) {
        try {
            if (url.contains("/d/")) {
                return url.split("/d/")[1].split("/")[0].split("\\?")[0];
            }
            if (url.contains("id=")) {
                return url.split("id=")[1].split("&")[0];
            }
        } catch (Exception e) {
            log.warn("Could not extract file ID from URL: {}", url);
        }
        return url;
    }

    // ── Excel ─────────────────────────────────────────────────────────
    private List<PharmacyInventoryDTO> readFromExcel(PharmacyNode pharmacy) {
        try {
            InputStream in = openStream(pharmacy.getFilePath());
            if (in == null) return List.of();
            return parseExcelStream(in, pharmacy);
        } catch (Exception e) {
            log.error("❌ Excel failed for {}: {}", pharmacy.getName(), e.getMessage());
            return List.of();
        }
    }

    private List<PharmacyInventoryDTO> parseExcelStream(InputStream in, PharmacyNode pharmacy) {
        List<PharmacyInventoryDTO> inventory = new ArrayList<>();
        try {
            Workbook wb = WorkbookFactory.create(in);
            Sheet sheet = wb.getSheetAt(0);
            Row header = sheet.getRow(0);
            if (header == null) {
                log.error("❌ Excel has no header row: {}", pharmacy.getName());
                return inventory;
            }

            Map<String, Integer> colIndex = new HashMap<>();
            for (int c = 0; c < header.getLastCellNum(); c++) {
                String h = cellStr(header.getCell(c)).toLowerCase().trim();
                if (!h.isBlank()) colIndex.put(h, c);
            }

            int nameCol   = findExcelCol(colIndex, pharmacy.getColMedicationName(),
                    "medication_name","name","drug_name","medicine","product_name","item_name","nom");
            int qtyCol    = findExcelCol(colIndex, pharmacy.getColStockQuantity(),
                    "stock_quantity","quantity","qty","stock","units","qte","total_quantity");
            int priceCol  = findExcelCol(colIndex, pharmacy.getColPrice(),
                    "price","unit_price","cost","selling_price","retail_price","prix");
            int dosageCol = findExcelCol(colIndex, pharmacy.getColDosage(),
                    "dosage","dose","strength","concentration","form");
            int curCol    = findExcelCol(colIndex, pharmacy.getColCurrency(),
                    "currency","currency_code","devise");

            if (nameCol < 0 || qtyCol < 0) {
                log.error("❌ Cannot find name/qty columns in Excel for {}. Headers found: {}",
                        pharmacy.getName(), colIndex.keySet());
                return inventory;
            }

            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);
                if (row == null) continue;
                String name = cellStr(row.getCell(nameCol));
                if (name.isBlank()) continue;
                int qty = (int) cellNum(row.getCell(qtyCol));

                PharmacyInventoryDTO dto = new PharmacyInventoryDTO();
                dto.setMedicationName(name);
                dto.setStockQuantity(qty);
                dto.setPrice(priceCol  >= 0 ? cellNum(row.getCell(priceCol))  : 0.0);
                dto.setDosage(dosageCol >= 0 ? cellStr(row.getCell(dosageCol)) : "");
                String cur = curCol >= 0 ? cellStr(row.getCell(curCol)) : "";
                dto.setCurrency(cur.isBlank() ? "LBP" : cur);
                dto.setAvailable(qty > 0);
                inventory.add(dto);
            }
            wb.close();
            log.info("✅ Excel parsed: {} items from {}", inventory.size(), pharmacy.getName());

        } catch (Exception e) {
            log.error("❌ Excel parse failed for {}: {}", pharmacy.getName(), e.getMessage());
        }
        return inventory;
    }

    // ── CSV ───────────────────────────────────────────────────────────
    private List<PharmacyInventoryDTO> readFromCsv(PharmacyNode pharmacy) {
        try {
            InputStream in = openStream(pharmacy.getFilePath());
            if (in == null) return List.of();
            return parseCsvStream(in, pharmacy);
        } catch (Exception e) {
            log.error("❌ CSV failed for {}: {}", pharmacy.getName(), e.getMessage());
            return List.of();
        }
    }

    private List<PharmacyInventoryDTO> parseCsvStream(InputStream in, PharmacyNode pharmacy) {
        List<PharmacyInventoryDTO> inventory = new ArrayList<>();
        try {
            Iterable<CSVRecord> records = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .build()
                    .parse(new InputStreamReader(in));

            for (CSVRecord rec : records) {
                String name = csvCol(rec, pharmacy.getColMedicationName(),
                        "medication_name","name","drug_name","medicine",
                        "product_name","item_name","nom");
                if (name.isBlank()) continue;

                int qty = parseInt(csvCol(rec, pharmacy.getColStockQuantity(),
                        "stock_quantity","quantity","qty","stock","units","qte","total_quantity"));
                double price = parseDouble(csvCol(rec, pharmacy.getColPrice(),
                        "price","unit_price","cost","selling_price","retail_price","prix"));
                String dosage   = csvCol(rec, pharmacy.getColDosage(),
                        "dosage","dose","strength","concentration","form");
                String currency = csvCol(rec, pharmacy.getColCurrency(),
                        "currency","currency_code","devise");

                PharmacyInventoryDTO dto = new PharmacyInventoryDTO();
                dto.setMedicationName(name);
                dto.setStockQuantity(qty);
                dto.setPrice(price);
                dto.setDosage(dosage);
                dto.setCurrency(currency.isBlank() ? "LBP" : currency);
                dto.setAvailable(qty > 0);
                inventory.add(dto);
            }
            log.info("✅ CSV parsed: {} items from {}", inventory.size(), pharmacy.getName());

        } catch (Exception e) {
            log.error("❌ CSV parse failed for {}: {}", pharmacy.getName(), e.getMessage());
        }
        return inventory;
    }

    // ── HikariCP pool ─────────────────────────────────────────────────
    private javax.sql.DataSource getOrCreatePool(PharmacyNode pharmacy, String driver) {
        return pools.computeIfAbsent(pharmacy.getId(), id -> {
            log.info("🔌 Creating pool for {} → {}", pharmacy.getName(), pharmacy.getDatabaseUrl());
            HikariConfig cfg = new HikariConfig();
            cfg.setJdbcUrl(pharmacy.getDatabaseUrl());
            cfg.setUsername(pharmacy.getDatabaseUsername());
            cfg.setPassword(pharmacy.getDatabasePassword());
            cfg.setDriverClassName(driver);
            cfg.setMaximumPoolSize(3);
            cfg.setConnectionTimeout(8_000);
            cfg.setIdleTimeout(300_000);
            cfg.setMaxLifetime(600_000);
            cfg.setPoolName("pharm-" + pharmacy.getId());
            if (pharmacy.getDatabaseUrl().contains("supabase.co")) {
                cfg.addDataSourceProperty("ssl", "true");
                cfg.addDataSourceProperty("sslmode", "require");
            }
            return new HikariDataSource(cfg);
        });
    }

    private void invalidatePool(Long id) {
        HikariDataSource pool = pools.remove(id);
        if (pool != null && !pool.isClosed()) pool.close();
    }

    // ── Column resolution ─────────────────────────────────────────────
    private String resolveString(ResultSet rs, Map<String, Integer> cols,
                                 String explicit, String... aliases) throws SQLException {
        if (explicit != null && !explicit.isBlank()) {
            String key = explicit.toLowerCase();
            if (cols.containsKey(key)) {
                String v = rs.getString(cols.get(key));
                return v != null ? v.trim() : null;
            }
            log.warn("Mapped column '{}' not found in result set", explicit);
        }
        for (String alias : aliases) {
            if (cols.containsKey(alias)) {
                String v = rs.getString(cols.get(alias));
                if (v != null && !v.isBlank()) return v.trim();
            }
        }
        return null;
    }

    private int resolveInt(ResultSet rs, Map<String, Integer> cols,
                           String explicit, String... aliases) throws SQLException {
        if (explicit != null && !explicit.isBlank()) {
            String key = explicit.toLowerCase();
            if (cols.containsKey(key)) return rs.getInt(cols.get(key));
            log.warn("Mapped column '{}' not found in result set", explicit);
        }
        for (String alias : aliases) {
            if (cols.containsKey(alias)) return rs.getInt(cols.get(alias));
        }
        return 0;
    }

    private double resolveDouble(ResultSet rs, Map<String, Integer> cols,
                                 String explicit, String... aliases) throws SQLException {
        if (explicit != null && !explicit.isBlank()) {
            String key = explicit.toLowerCase();
            if (cols.containsKey(key)) return rs.getDouble(cols.get(key));
            log.warn("Mapped column '{}' not found in result set", explicit);
        }
        for (String alias : aliases) {
            if (cols.containsKey(alias)) return rs.getDouble(cols.get(alias));
        }
        return 0.0;
    }

    private int findExcelCol(Map<String, Integer> headers, String explicit, String... aliases) {
        if (explicit != null && !explicit.isBlank()) {
            Integer idx = headers.get(explicit.toLowerCase().trim());
            if (idx != null) return idx;
        }
        for (String alias : aliases) {
            Integer idx = headers.get(alias);
            if (idx != null) return idx;
        }
        return -1;
    }

    private String csvCol(CSVRecord rec, String explicit, String... aliases) {
        if (explicit != null && !explicit.isBlank()) {
            try {
                String v = rec.get(explicit);
                if (v != null && !v.isBlank()) return v.trim();
            } catch (IllegalArgumentException ignored) {}
        }
        for (String alias : aliases) {
            try {
                String v = rec.get(alias);
                if (v != null && !v.isBlank()) return v.trim();
            } catch (IllegalArgumentException ignored) {}
        }
        return "";
    }

    // ── File / stream helpers ─────────────────────────────────────────
    private InputStream openStream(String path) throws IOException {
        if (path == null || path.isBlank()) return null;
        if (path.startsWith("http://") || path.startsWith("https://"))
            return new URL(path).openStream();
        File f = new File(path);
        if (!f.exists()) { log.warn("File not found: {}", path); return null; }
        return new FileInputStream(f);
    }

    private String cellStr(Cell c) {
        if (c == null) return "";
        return switch (c.getCellType()) {
            case STRING  -> c.getStringCellValue().trim();
            case NUMERIC -> String.valueOf((long) c.getNumericCellValue());
            default      -> "";
        };
    }

    private double cellNum(Cell c) {
        if (c == null) return 0;
        return switch (c.getCellType()) {
            case NUMERIC -> c.getNumericCellValue();
            case STRING  -> parseDouble(c.getStringCellValue());
            default      -> 0;
        };
    }

    private int parseInt(String s) {
        try { return Integer.parseInt(s.replaceAll("[^0-9]", "")); }
        catch (Exception e) { return 0; }
    }

    private double parseDouble(String s) {
        try { return Double.parseDouble(s.replaceAll("[^0-9.]", "")); }
        catch (Exception e) { return 0.0; }
    }
}