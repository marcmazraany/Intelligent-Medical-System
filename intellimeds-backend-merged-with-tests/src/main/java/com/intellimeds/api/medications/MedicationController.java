package com.intellimeds.api.medications;

import com.intellimeds.api.medications.dto.MedicationDto;
import com.intellimeds.api.medications.dto.MedicationScanBatchResponse;
import com.intellimeds.api.medications.dto.UpsertMedicationRequest;
import com.intellimeds.api.security.AuthUser;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/medications")
public class MedicationController {

    private final MedicationService service;
    private final MedicationScanService scanService;

    public MedicationController(
            MedicationService service,
            MedicationScanService scanService
    ) {
        this.service = service;
        this.scanService = scanService;
    }

    @GetMapping
    public List<MedicationDto> list(@AuthenticationPrincipal AuthUser user) {
        return service.list(user.userId());
    }

    @PostMapping
    public MedicationDto create(
            @AuthenticationPrincipal AuthUser user,
            @Valid @RequestBody UpsertMedicationRequest request
    ) {
        return service.create(user.userId(), request);
    }

    @PutMapping("/{id}")
    public MedicationDto update(
            @AuthenticationPrincipal AuthUser user,
            @PathVariable UUID id,
            @Valid @RequestBody UpsertMedicationRequest request
    ) {
        return service.update(user.userId(), id, request);
    }

    @DeleteMapping("/{id}")
    public void delete(
            @AuthenticationPrincipal AuthUser user,
            @PathVariable UUID id
    ) {
        service.delete(user.userId(), id);
    }

    /**
     * Scan barcode - if barcode not found, returns error telling frontend
     * to prompt user for text photo
     */
    @PostMapping("/scan")
    public ResponseEntity<?> scan(
            @AuthenticationPrincipal AuthUser user,
            @RequestParam("file") MultipartFile file
    ) {
        try {
            MedicationScanBatchResponse response = scanService.scanBarcode(user.userId(), file);
            return ResponseEntity.ok(response);

        } catch (MedicationScannerClient.BarcodeNotFoundException ex) {
            // Barcode not found - return 404 with special message for frontend
            Map<String, Object> errorResponse = ex.getResponse();
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(errorResponse);
        }
    }

    /**
     * NEW: Scan medication text/name with OCR
     * Frontend calls this after user takes photo of medication name
     */
    @PostMapping("/scan-text")
    public MedicationScanBatchResponse scanText(
            @AuthenticationPrincipal AuthUser user,
            @RequestParam("file") MultipartFile file
    ) {
        return scanService.scanText(user.userId(), file);
    }
}