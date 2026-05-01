package com.intellimeds.api.auth;

import com.intellimeds.api.common.BadRequestException;
import com.intellimeds.api.profile.ProfileEntity;
import com.intellimeds.api.profile.ProfileRepository;
import com.intellimeds.api.security.JwtService;
import com.intellimeds.api.users.UserEntity;
import com.intellimeds.api.users.UserRepository;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNoException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("Authentication – AuthService Unit Tests")
class AuthServiceTest {

    @Mock private UserRepository userRepository;
    @Mock private ProfileRepository profileRepository;
    @Mock private PasswordEncoder passwordEncoder;
    @Mock private JwtService jwtService;
    @Mock private EmailService emailService;

    @InjectMocks
    private AuthService authService;

    private static final UUID USER_ID = UUID.fromString("11111111-1111-1111-1111-111111111111");
    private static final String EMAIL = "jana@intellimeds.com";
    private static final String PASSWORD = "securePassword123";
    private static final String HASH = "$2a$10$hashedpassword";

    private UserEntity user() {
        UserEntity user = new UserEntity();
        user.setId(USER_ID);
        user.setEmail(EMAIL);
        user.setPhone("+96170000000");
        user.setPasswordHash(HASH);
        return user;
    }

    private ProfileEntity profile(UserEntity user) {
        ProfileEntity profile = new ProfileEntity();
        profile.setUserId(USER_ID);
        profile.setUser(user);
        profile.setFirstName("Jana");
        profile.setLastName("Barhouche");
        return profile;
    }

    @Nested
    @DisplayName("signUp")
    class SignUp {
        @Test
        @DisplayName("throws BadRequestException when email already exists")
        void signUp_throwsWhenEmailAlreadyExists() {
            when(userRepository.existsByEmailIgnoreCase(EMAIL)).thenReturn(true);

            assertThatThrownBy(() -> authService.signUp("Jana", "Barhouche", EMAIL, "+96170000000", PASSWORD))
                    .isInstanceOf(BadRequestException.class)
                    .hasMessageContaining("Email already in use");
        }

        @Test
        @DisplayName("encodes password before saving user")
        void signUp_encodesPasswordBeforeSavingUser() {
            when(userRepository.existsByEmailIgnoreCase(EMAIL)).thenReturn(false);
            when(passwordEncoder.encode(PASSWORD)).thenReturn(HASH);
            when(userRepository.save(any(UserEntity.class))).thenAnswer(invocation -> {
                UserEntity saved = invocation.getArgument(0);
                saved.setId(USER_ID);
                return saved;
            });
            when(profileRepository.save(any(ProfileEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));
            when(jwtService.createAccessToken(USER_ID, EMAIL)).thenReturn("access.token");
            when(jwtService.createRefreshToken(USER_ID)).thenReturn("refresh.token");

            authService.signUp("Jana", "Barhouche", EMAIL, "+96170000000", PASSWORD);

            verify(passwordEncoder).encode(PASSWORD);
            ArgumentCaptor<UserEntity> userCaptor = ArgumentCaptor.forClass(UserEntity.class);
            verify(userRepository).save(userCaptor.capture());
            assertThat(userCaptor.getValue().getPasswordHash()).isEqualTo(HASH);
        }

        @Test
        @DisplayName("creates a profile row with the supplied first and last name")
        void signUp_createsProfile() {
            when(userRepository.existsByEmailIgnoreCase(EMAIL)).thenReturn(false);
            when(passwordEncoder.encode(anyString())).thenReturn(HASH);
            when(userRepository.save(any(UserEntity.class))).thenAnswer(invocation -> {
                UserEntity saved = invocation.getArgument(0);
                saved.setId(USER_ID);
                return saved;
            });
            when(profileRepository.save(any(ProfileEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));
            when(jwtService.createAccessToken(USER_ID, EMAIL)).thenReturn("access.token");
            when(jwtService.createRefreshToken(USER_ID)).thenReturn("refresh.token");

            authService.signUp("Jana", "Barhouche", EMAIL, "+96170000000", PASSWORD);

            verify(profileRepository).save(argThat(profile ->
                    "Jana".equals(profile.getFirstName()) && "Barhouche".equals(profile.getLastName())));
        }

        @Test
        @DisplayName("returns access token, refresh token, and profile")
        void signUp_returnsTokensAndProfile() {
            when(userRepository.existsByEmailIgnoreCase(EMAIL)).thenReturn(false);
            when(passwordEncoder.encode(anyString())).thenReturn(HASH);
            when(userRepository.save(any(UserEntity.class))).thenAnswer(invocation -> {
                UserEntity saved = invocation.getArgument(0);
                saved.setId(USER_ID);
                return saved;
            });
            when(profileRepository.save(any(ProfileEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));
            when(jwtService.createAccessToken(USER_ID, EMAIL)).thenReturn("access-jwt");
            when(jwtService.createRefreshToken(USER_ID)).thenReturn("refresh-jwt");

            AuthService.AuthTokensAndProfile response = authService.signUp(
                    "Jana", "Barhouche", EMAIL, "+96170000000", PASSWORD);

            assertThat(response.accessToken()).isEqualTo("access-jwt");
            assertThat(response.refreshToken()).isEqualTo("refresh-jwt");
            assertThat(response.profile().firstName()).isEqualTo("Jana");
        }
    }

    @Nested
    @DisplayName("signIn")
    class SignIn {
        @Test
        @DisplayName("throws BadRequestException for an unknown email")
        void signIn_throwsForUnknownEmail() {
            when(userRepository.findByEmailIgnoreCase(EMAIL)).thenReturn(Optional.empty());

            assertThatThrownBy(() -> authService.signIn(EMAIL, PASSWORD))
                    .isInstanceOf(BadRequestException.class)
                    .hasMessageContaining("Invalid email or password");
        }

        @Test
        @DisplayName("throws BadRequestException when password does not match")
        void signIn_throwsForBadPassword() {
            UserEntity user = user();
            when(userRepository.findByEmailIgnoreCase(EMAIL)).thenReturn(Optional.of(user));
            when(passwordEncoder.matches(PASSWORD, HASH)).thenReturn(false);

            assertThatThrownBy(() -> authService.signIn(EMAIL, PASSWORD))
                    .isInstanceOf(BadRequestException.class)
                    .hasMessageContaining("Invalid email or password");
        }

        @Test
        @DisplayName("returns tokens for valid credentials")
        void signIn_returnsTokensForValidCredentials() {
            UserEntity user = user();
            when(userRepository.findByEmailIgnoreCase(EMAIL)).thenReturn(Optional.of(user));
            when(passwordEncoder.matches(PASSWORD, HASH)).thenReturn(true);
            when(profileRepository.findById(USER_ID)).thenReturn(Optional.of(profile(user)));
            when(jwtService.createAccessToken(USER_ID, EMAIL)).thenReturn("access");
            when(jwtService.createRefreshToken(USER_ID)).thenReturn("refresh");

            AuthService.AuthTokensAndProfile response = authService.signIn(EMAIL, PASSWORD);

            assertThat(response.accessToken()).isEqualTo("access");
            assertThat(response.refreshToken()).isEqualTo("refresh");
            assertThat(response.profile().lastName()).isEqualTo("Barhouche");
        }
    }

    @Nested
    @DisplayName("password reset with verification code")
    class PasswordReset {
        @Test
        @DisplayName("sendVerificationCode does not reveal unknown emails")
        void sendVerificationCode_doesNotThrowForUnknownEmail() {
            when(userRepository.findByEmailIgnoreCase("unknown@test.com")).thenReturn(Optional.empty());

            assertThatNoException().isThrownBy(() -> authService.sendVerificationCode("unknown@test.com"));
            verify(emailService, never()).sendVerificationCode(anyString(), anyString());
        }

        @Test
        @DisplayName("resetPasswordWithCode throws when code is wrong")
        void resetPasswordWithCode_throwsForWrongCode() {
            UserEntity user = user();
            when(userRepository.findByEmailIgnoreCase(EMAIL)).thenReturn(Optional.of(user));

            authService.sendVerificationCode(EMAIL);

            assertThatThrownBy(() -> authService.resetPasswordWithCode(EMAIL, "000000", "newPassword"))
                    .isInstanceOf(BadRequestException.class)
                    .hasMessageContaining("Invalid verification code");
        }

        @Test
        @DisplayName("resetPasswordWithCode updates the password when code is correct")
        void resetPasswordWithCode_updatesPasswordForCorrectCode() {
            UserEntity user = user();
            when(userRepository.findByEmailIgnoreCase(EMAIL)).thenReturn(Optional.of(user));
            ArgumentCaptor<String> codeCaptor = ArgumentCaptor.forClass(String.class);

            authService.sendVerificationCode(EMAIL);
            verify(emailService).sendVerificationCode(eq(EMAIL), codeCaptor.capture());

            when(userRepository.findById(USER_ID)).thenReturn(Optional.of(user));
            when(passwordEncoder.encode("newPassword")).thenReturn("newHash");
            when(userRepository.save(any(UserEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));

            authService.resetPasswordWithCode(EMAIL, codeCaptor.getValue(), "newPassword");

            verify(passwordEncoder).encode("newPassword");
            verify(userRepository).save(argThat(saved -> "newHash".equals(saved.getPasswordHash())));
            verify(emailService).sendPasswordResetConfirmation(EMAIL);
        }
    }
}
