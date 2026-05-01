package com.medmanager.enums;

public enum PharmacyDataSourceType {
    REST_API,           // HTTP endpoint — your PharmacyNodeController or any REST API
    POSTGRESQL,         // Direct JDBC — local Postgres OR Supabase
    MYSQL,              // Direct JDBC — local MySQL / MariaDB
    EXCEL_FILE,         // Local .xlsx path on your server OR public URL
    CSV_FILE,           // Local .csv path on your server OR public URL
    GOOGLE_DRIVE_EXCEL, // Private Excel file on pharmacy's Google Drive
    GOOGLE_DRIVE_CSV    // Private CSV file on pharmacy's Google Drive
}