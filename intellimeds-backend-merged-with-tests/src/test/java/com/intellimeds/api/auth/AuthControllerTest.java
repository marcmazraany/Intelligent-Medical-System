package com.intellimeds.api.auth;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.intellimeds.api.common.BadRequestException;
import com.intellimeds.api.common.GlobalExceptionHandler;
import com.intellimeds.api.profile.dto.UserProfileDto;
import com.intellimeds.api.security.JwtService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Map;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(AuthController.class)
@AutoConfigureMockMvc(addFilters = false)
@Import(GlobalExceptionHandler.class)
@DisplayName("Authentication – AuthController Endpoint Tests")
class AuthControllerTest {

    @Autowired private MockMvc mvc;
    @Autowired private ObjectMapper mapper;

    @MockBean private AuthService authService;
    @MockBean private JwtService jwtService;

    private static final String SIGNUP = "/api/auth/signup";
    private static final String SIGNIN = "/api/auth/signin";
    private static final String FORGOT = "/api/auth/forgot-password";
    private static final String RESET = "/api/auth/reset-password";

    private AuthService.AuthTokensAndProfile tokens() {
        return new AuthService.AuthTokensAndProfile(
                "access.jwt",
                "refresh.jwt",
                UserProfileDto.builder().firstName("Jana").lastName("Barhouche").build()
        );
    }

    @Nested
    @DisplayName("POST /api/auth/signup")
    class SignupEndpoint {
        @Test
        @DisplayName("returns tokens and profile for a valid request")
        void signup_returnsTokensForValidRequest() throws Exception {
            when(authService.signUp(anyString(), anyString(), anyString(), anyString(), anyString())).thenReturn(tokens());

            mvc.perform(post(SIGNUP)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(mapper.writeValueAsString(Map.of(
                                    "firstName", "Jana",
                                    "lastName", "Barhouche",
                                    "email", "jana@test.com",
                                    "phone", "+96170000001",
                                    "password", "password123"
                            ))))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.accessToken").value("access.jwt"))
                    .andExpect(jsonPath("$.refreshToken").value("refresh.jwt"))
                    .andExpect(jsonPath("$.profile.firstName").value("Jana"));
        }

        @Test
        @DisplayName("returns 400 when email is missing")
        void signup_returns400WhenEmailMissing() throws Exception {
            mvc.perform(post(SIGNUP)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(mapper.writeValueAsString(Map.of(
                                    "firstName", "Jana",
                                    "lastName", "Barhouche",
                                    "password", "password123"
                            ))))
                    .andExpect(status().isBadRequest());
        }

        @Test
        @DisplayName("returns 400 when the email is already in use")
        void signup_returns400WhenEmailAlreadyRegistered() throws Exception {
            when(authService.signUp(anyString(), anyString(), anyString(), anyString(), anyString()))
                    .thenThrow(new BadRequestException("Email already in use"));

            mvc.perform(post(SIGNUP)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(mapper.writeValueAsString(Map.of(
                                    "firstName", "Jana",
                                    "lastName", "Barhouche",
                                    "email", "existing@test.com",
                                    "phone", "+96170000001",
                                    "password", "password123"
                            ))))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.message").value("Email already in use"));
        }
    }

    @Nested
    @DisplayName("POST /api/auth/signin")
    class SigninEndpoint {
        @Test
        @DisplayName("returns tokens for correct credentials")
        void signin_returnsTokensForCorrectCredentials() throws Exception {
            when(authService.signIn(anyString(), anyString())).thenReturn(tokens());

            mvc.perform(post(SIGNIN)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(mapper.writeValueAsString(Map.of(
                                    "email", "jana@test.com",
                                    "password", "password123"
                            ))))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.accessToken").value("access.jwt"));
        }

        @Test
        @DisplayName("returns 400 for invalid credentials")
        void signin_returns400ForInvalidCredentials() throws Exception {
            when(authService.signIn(anyString(), anyString()))
                    .thenThrow(new BadRequestException("Invalid email or password"));

            mvc.perform(post(SIGNIN)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(mapper.writeValueAsString(Map.of(
                                    "email", "jana@test.com",
                                    "password", "wrong"
                            ))))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.message").value("Invalid email or password"));
        }

        @Test
        @DisplayName("returns 400 on empty body")
        void signin_returns400OnEmptyBody() throws Exception {
            mvc.perform(post(SIGNIN)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content("{}"))
                    .andExpect(status().isBadRequest());
        }
    }

    @Nested
    @DisplayName("POST /api/auth/forgot-password")
    class ForgotPasswordEndpoint {
        @Test
        @DisplayName("returns the safe success response")
        void forgotPassword_returnsSafeSuccessResponse() throws Exception {
            mvc.perform(post(FORGOT)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(mapper.writeValueAsString(Map.of("email", "any@email.com"))))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.success").value(true));
        }

        @Test
        @DisplayName("returns 400 when email is missing")
        void forgotPassword_returns400WhenEmailMissing() throws Exception {
            mvc.perform(post(FORGOT)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content("{}"))
                    .andExpect(status().isBadRequest());
        }
    }

    @Nested
    @DisplayName("POST /api/auth/reset-password")
    class ResetPasswordEndpoint {
        @Test
        @DisplayName("returns success when code and new password are valid")
        void resetPassword_returnsSuccess() throws Exception {
            mvc.perform(post(RESET)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(mapper.writeValueAsString(Map.of(
                                    "email", "jana@test.com",
                                    "code", "123456",
                                    "newPassword", "newPassword123"
                            ))))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.success").value(true));
        }

        @Test
        @DisplayName("returns 400 when the code is invalid")
        void resetPassword_returns400WhenCodeInvalid() throws Exception {
            doThrow(new BadRequestException("Invalid verification code"))
                    .when(authService).resetPasswordWithCode(anyString(), anyString(), anyString());

            mvc.perform(post(RESET)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(mapper.writeValueAsString(Map.of(
                                    "email", "jana@test.com",
                                    "code", "000000",
                                    "newPassword", "newPassword123"
                            ))))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.message").value("Invalid verification code"));
        }
    }
}
