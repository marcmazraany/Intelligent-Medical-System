package com.intellimeds.api.auth;

import com.intellimeds.api.common.BadRequestException;
import com.intellimeds.api.common.NotFoundException;
import com.intellimeds.api.profile.ProfileEntity;
import com.intellimeds.api.profile.ProfileRepository;
import com.intellimeds.api.profile.dto.UserProfileDto;
import com.intellimeds.api.security.JwtService;
import com.intellimeds.api.users.UserEntity;
import com.intellimeds.api.users.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class AuthService {

    private static final Logger log = LoggerFactory.getLogger(AuthService.class);

    private final UserRepository users;
    private final ProfileRepository profiles;
    private final PasswordEncoder encoder;
    private final JwtService jwt;
    private final EmailService emailService;

    // In-memory code storage (simple - no database needed)
    private final Map<String, VerificationCode> verificationCodes = new ConcurrentHashMap<>();

    public AuthService(
            UserRepository users,
            ProfileRepository profiles,
            PasswordEncoder encoder,
            JwtService jwt,
            EmailService emailService
    ) {
        this.users = users;
        this.profiles = profiles;
        this.encoder = encoder;
        this.jwt = jwt;
        this.emailService = emailService;
    }

    @Transactional
    public AuthTokensAndProfile signUp(String firstName, String lastName, String email, String phone, String password) {
        if (users.existsByEmailIgnoreCase(email)) {
            throw new BadRequestException("Email already in use");
        }

        UserEntity u = UserEntity.builder()
                .email(email.toLowerCase())
                .phone(phone)
                .passwordHash(encoder.encode(password))
                .build();
        u = users.save(u);

        ProfileEntity p = ProfileEntity.builder()
                .user(u)
                .firstName(firstName)
                .lastName(lastName)
                .build();
        p = profiles.save(p);

        String access = jwt.createAccessToken(u.getId(), u.getEmail());
        String refresh = jwt.createRefreshToken(u.getId());
        return new AuthTokensAndProfile(access, refresh, toProfileDto(p));
    }

    @Transactional(readOnly = true)
    public AuthTokensAndProfile signIn(String email, String password) {
        UserEntity u = users.findByEmailIgnoreCase(email)
                .orElseThrow(() -> new BadRequestException("Invalid email or password"));

        if (!encoder.matches(password, u.getPasswordHash())) {
            throw new BadRequestException("Invalid email or password");
        }

        ProfileEntity p = profiles.findById(u.getId())
                .orElseGet(() -> profiles.save(
                        ProfileEntity.builder()
                                .user(u)
                                .firstName("User")
                                .lastName("User")
                                .build()
                ));

        String access = jwt.createAccessToken(u.getId(), u.getEmail());
        String refresh = jwt.createRefreshToken(u.getId());
        return new AuthTokensAndProfile(access, refresh, toProfileDto(p));
    }

    public String refresh(String refreshToken) {
        var claims = jwt.parse(refreshToken);
        if (!"refresh".equals(claims.get("type", String.class))) {
            throw new BadRequestException("Invalid refresh token");
        }
        UUID userId = UUID.fromString(claims.getSubject());
        UserEntity u = users.findById(userId)
                .orElseThrow(() -> new BadRequestException("Invalid refresh token"));
        return jwt.createAccessToken(u.getId(), u.getEmail());
    }

    // ============ PASSWORD RESET WITH VERIFICATION CODE ============

    /**
     * Step 1: Send verification code to email
     */
    @Transactional
    public void sendVerificationCode(String email) {
        log.info("Verification code requested for: {}", email);

        // Find user (don't reveal if email exists)
        UserEntity user = users.findByEmailIgnoreCase(email).orElse(null);

        if (user == null) {
            log.warn("Verification code requested for non-existent email: {}", email);
            // Don't reveal email doesn't exist - just return silently
            return;
        }

        // Generate 6-digit code
        String code = generateSixDigitCode();

        // Store code with expiry (10 minutes)
        VerificationCode verificationCode = new VerificationCode(
                user.getId(),
                user.getEmail(),
                code,
                Instant.now().plus(10, ChronoUnit.MINUTES)
        );
        verificationCodes.put(email.toLowerCase(), verificationCode);

        log.info("Generated verification code for {}: {}", email, code);

        // Send email with code
        try {
            emailService.sendVerificationCode(user.getEmail(), code);
            log.info("Verification code email sent successfully to: {}", email);
        } catch (Exception e) {
            log.error("Failed to send verification email", e);
            throw new BadRequestException("Failed to send verification code. Please try again later.");
        }
    }

    /**
     * Step 2: Verify code and reset password
     */
    @Transactional
    public void resetPasswordWithCode(String email, String code, String newPassword) {
        log.info("Password reset attempt with code for: {}", email);

        // Get and validate code
        VerificationCode storedCode = verificationCodes.get(email.toLowerCase());

        if (storedCode == null) {
            log.warn("No verification code found for: {}", email);
            throw new BadRequestException("Invalid or expired verification code");
        }

        if (storedCode.isExpired()) {
            verificationCodes.remove(email.toLowerCase());
            log.warn("Expired verification code used for: {}", email);
            throw new BadRequestException("Verification code has expired. Please request a new one.");
        }

        if (!storedCode.code.equals(code)) {
            log.warn("Invalid verification code entered for: {}", email);
            throw new BadRequestException("Invalid verification code");
        }

        // Find user
        UserEntity user = users.findById(storedCode.userId)
                .orElseThrow(() -> new NotFoundException("User not found"));

        // Update password
        user.setPasswordHash(encoder.encode(newPassword));
        users.save(user);

        // Remove used code
        verificationCodes.remove(email.toLowerCase());

        log.info("Password successfully reset for: {}", user.getEmail());

        // Send confirmation email
        try {
            emailService.sendPasswordResetConfirmation(user.getEmail());
        } catch (Exception e) {
            log.warn("Failed to send confirmation email", e);
        }
    }

    // ============ HELPER METHODS ============

    /**
     * Generate random 6-digit code
     */
    private String generateSixDigitCode() {
        Random random = new Random();
        int code = 100000 + random.nextInt(900000); // 100000 to 999999
        return String.valueOf(code);
    }

    private static UserProfileDto toProfileDto(ProfileEntity p) {
        return UserProfileDto.builder()
                .firstName(p.getFirstName())
                .lastName(p.getLastName())
                .dob(p.getDob())
                .gender(p.getGender())
                .height(p.getHeight())
                .weight(p.getWeight())
                .allergies(p.getAllergies())
                .bloodType(p.getBloodType())
                .chronicConditions(p.getChronicConditions())
                .notes(p.getNotes())
                .build();
    }

    // Inner class for code storage
    private static class VerificationCode {
        final UUID userId;
        final String email;
        final String code;
        final Instant expiresAt;

        VerificationCode(UUID userId, String email, String code, Instant expiresAt) {
            this.userId = userId;
            this.email = email;
            this.code = code;
            this.expiresAt = expiresAt;
        }

        boolean isExpired() {
            return Instant.now().isAfter(expiresAt);
        }
    }

    public record AuthTokensAndProfile(String accessToken, String refreshToken, UserProfileDto profile) {}
}