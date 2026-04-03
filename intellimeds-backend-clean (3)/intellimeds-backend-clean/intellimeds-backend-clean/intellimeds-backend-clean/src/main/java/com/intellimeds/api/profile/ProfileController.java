package com.intellimeds.api.profile;

import com.intellimeds.api.profile.dto.UserProfileDto;
import com.intellimeds.api.security.AuthUser;
import jakarta.validation.Valid;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/profile")
public class ProfileController {

    private final ProfileService service;

    public ProfileController(ProfileService service) {
        this.service = service;
    }

    @GetMapping
    public UserProfileDto get(@AuthenticationPrincipal AuthUser user) {
        return service.get(user.userId());
    }

    @PutMapping
    public UserProfileDto put(@AuthenticationPrincipal AuthUser user, @Valid @RequestBody UserProfileDto dto) {
        return service.upsert(user.userId(), dto);
    }
}
