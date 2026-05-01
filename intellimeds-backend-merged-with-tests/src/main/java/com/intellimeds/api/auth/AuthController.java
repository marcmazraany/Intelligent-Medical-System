package com.intellimeds.api.auth;

import com.intellimeds.api.auth.dto.*;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

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
    public Map<String, String> refresh(@Valid @RequestBody RefreshRequest req) {
        String newAccess = service.refresh(req.refreshToken());
        return Map.of("accessToken", newAccess);
    }

    // ============ PASSWORD RESET WITH CODE ============

    /**
     * Step 1: Send verification code to email
     * User enters email → Gets 6-digit code in email
     */
    @PostMapping("/forgot-password")
    public ResponseEntity<Map<String, Object>> sendVerificationCode(@Valid @RequestBody ForgotPasswordRequest req) {
        service.sendVerificationCode(req.email());

        return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "If this email is registered, you will receive a verification code shortly"
        ));
    }

    /**
     * Step 2: Verify code and reset password
     * User enters: email + code + new password
     */
    @PostMapping("/reset-password")
    public ResponseEntity<Map<String, Object>> resetPassword(@Valid @RequestBody ResetPasswordWithCodeRequest req) {
        service.resetPasswordWithCode(req.email(), req.code(), req.newPassword());

        return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "Your password has been reset successfully"
        ));
    }
}