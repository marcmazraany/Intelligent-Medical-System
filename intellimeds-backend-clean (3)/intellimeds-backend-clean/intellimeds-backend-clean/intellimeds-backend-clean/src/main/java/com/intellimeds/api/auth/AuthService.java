package com.intellimeds.api.auth;

import com.intellimeds.api.common.BadRequestException;
import com.intellimeds.api.profile.ProfileEntity;
import com.intellimeds.api.profile.ProfileRepository;
import com.intellimeds.api.profile.dto.UserProfileDto;
import com.intellimeds.api.security.JwtService;
import com.intellimeds.api.users.UserEntity;
import com.intellimeds.api.users.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.UUID;

@Service
public class AuthService {

    private final UserRepository users;
    private final ProfileRepository profiles;
    private final PasswordEncoder encoder;
    private final JwtService jwt;

    public AuthService(UserRepository users, ProfileRepository profiles, PasswordEncoder encoder, JwtService jwt) {
        this.users = users;
        this.profiles = profiles;
        this.encoder = encoder;
        this.jwt = jwt;
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

        // Always create & save profile row
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
    public record AuthTokensAndProfile(String accessToken, String refreshToken, UserProfileDto profile) {}
}
