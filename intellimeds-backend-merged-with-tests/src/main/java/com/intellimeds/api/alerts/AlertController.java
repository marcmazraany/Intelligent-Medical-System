package com.intellimeds.api.alerts;

import com.intellimeds.api.alerts.dto.*;
import com.intellimeds.api.security.AuthUser;
import jakarta.validation.Valid;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/alerts")
public class AlertController {

    private final AlertService service;

    public AlertController(AlertService service) {
        this.service = service;
    }

    @GetMapping
    public List<AlertDto> list(@AuthenticationPrincipal AuthUser user) {
        return service.list(user.userId());
    }

    @PostMapping
    public AlertDto create(@AuthenticationPrincipal AuthUser user, @Valid @RequestBody CreateAlertRequest req) {
        return service.create(user.userId(), req);
    }

    @PatchMapping("/{id}")
    public AlertDto patch(@AuthenticationPrincipal AuthUser user, @PathVariable UUID id, @RequestBody PatchAlertRequest req) {
        return service.patch(user.userId(), id, req);
    }

    @DeleteMapping("/{id}")
    public java.util.Map<String, String> delete(@AuthenticationPrincipal AuthUser user, @PathVariable UUID id) {
        service.delete(user.userId(), id);
        return java.util.Map.of("status", "deleted");
    }
}
