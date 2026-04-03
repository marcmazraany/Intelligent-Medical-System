package com.intellimeds.api.medications;

import com.intellimeds.api.medications.dto.MedicationDto;
import com.intellimeds.api.medications.dto.MedicationScanBatchResponse;
import com.intellimeds.api.medications.dto.UpsertMedicationRequest;
import com.intellimeds.api.security.AuthUser;
import jakarta.validation.Valid;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
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

    @PostMapping("/scan")
    public MedicationScanBatchResponse scan(
            @AuthenticationPrincipal AuthUser user,
            @RequestParam("file") MultipartFile file
    ) {
        return scanService.scanAndSave(user.userId(), file);
    }
}