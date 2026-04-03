package com.medmanager.controller;

import com.medmanager.service.SmartPharmacyScanner;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/test")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class TestController {

    private final SmartPharmacyScanner scanner;

    @PostMapping("/trigger-scan")
    public ResponseEntity<String> triggerScan() {
        scanner.triggerSmartScan();
        return ResponseEntity.ok("Scan triggered");
    }
}