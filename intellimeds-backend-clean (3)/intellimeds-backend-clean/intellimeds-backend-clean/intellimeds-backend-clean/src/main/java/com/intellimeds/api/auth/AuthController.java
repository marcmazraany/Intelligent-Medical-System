package com.intellimeds.api.auth;

import com.intellimeds.api.auth.dto.*;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService service;

    public AuthController(AuthService service) {
        this.service = service;
    }

    @PostMapping("/signup")
    public AuthResponse signup(@Valid @RequestBody SignUpRequest req) {
        var out = service.signUp(req.firstName(),req.lastName(), req.email(), req.phone(), req.password());
        return AuthResponse.builder()
                .accessToken(out.accessToken())
                .refreshToken(out.refreshToken())
                .profile(out.profile())
                .build();
    }

    @PostMapping("/signin")
    public AuthResponse signin(@Valid @RequestBody SignInRequest req) {
        var out = service.signIn(req.email(), req.password());
        return AuthResponse.builder()
                .accessToken(out.accessToken())
                .refreshToken(out.refreshToken())
                .profile(out.profile())
                .build();
    }

    @PostMapping("/refresh")
    public java.util.Map<String, String> refresh(@Valid @RequestBody RefreshRequest req) {
        String newAccess = service.refresh(req.refreshToken());
        return java.util.Map.of("accessToken", newAccess);
    }
}
