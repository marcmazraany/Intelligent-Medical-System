package com.intellimeds.api.security;

import io.jsonwebtoken.Claims;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@DisplayName("Security – JwtService Unit Tests")
class JwtServiceTest {

    private JwtService jwtService;

    private static final UUID USER_ID = UUID.fromString("11111111-1111-1111-1111-111111111111");
    private static final String EMAIL = "test@intellimeds.com";

    @BeforeEach
    void setUp() {
        jwtService = new JwtService(props("intellimeds-test-secret-that-is-long-enough-123"));
    }

    private JwtProperties props(String secret) {
        JwtProperties props = new JwtProperties();
        props.setIssuer("intellimeds-test");
        props.setSecret(secret);
        props.setAccessTtlMinutes(30);
        props.setRefreshTtlDays(14);
        return props;
    }

    @Nested
    @DisplayName("access tokens")
    class AccessTokens {
        @Test
        @DisplayName("createAccessToken returns a three-part JWT")
        void createAccessToken_returnsThreePartJwt() {
            String token = jwtService.createAccessToken(USER_ID, EMAIL);

            assertThat(token).isNotBlank();
            assertThat(token.split("\\.")).hasSize(3);
        }

        @Test
        @DisplayName("parse returns user id, email, issuer, and type claims")
        void parse_returnsAccessTokenClaims() {
            String token = jwtService.createAccessToken(USER_ID, EMAIL);

            Claims claims = jwtService.parse(token);

            assertThat(claims.getSubject()).isEqualTo(USER_ID.toString());
            assertThat(claims.get("email", String.class)).isEqualTo(EMAIL);
            assertThat(claims.get("type", String.class)).isEqualTo("access");
            assertThat(claims.getIssuer()).isEqualTo("intellimeds-test");
            assertThat(claims.getExpiration()).isNotNull();
        }
    }

    @Nested
    @DisplayName("refresh tokens")
    class RefreshTokens {
        @Test
        @DisplayName("createRefreshToken stores subject and refresh type")
        void createRefreshToken_hasRefreshType() {
            String token = jwtService.createRefreshToken(USER_ID);

            Claims claims = jwtService.parse(token);

            assertThat(claims.getSubject()).isEqualTo(USER_ID.toString());
            assertThat(claims.get("type", String.class)).isEqualTo("refresh");
        }

        @Test
        @DisplayName("access and refresh tokens for the same user are different")
        void accessAndRefreshTokens_areDifferent() {
            String access = jwtService.createAccessToken(USER_ID, EMAIL);
            String refresh = jwtService.createRefreshToken(USER_ID);

            assertThat(access).isNotEqualTo(refresh);
        }
    }

    @Nested
    @DisplayName("invalid tokens")
    class InvalidTokens {
        @Test
        @DisplayName("parse throws for a malformed token")
        void parse_throwsForMalformedToken() {
            assertThatThrownBy(() -> jwtService.parse("not.a.jwt"))
                    .isInstanceOf(Exception.class);
        }

        @Test
        @DisplayName("parse throws for token signed with a different secret")
        void parse_throwsForWrongSecret() {
            JwtService otherService = new JwtService(props("different-intellimeds-secret-long-enough-456"));
            String tokenFromOtherSecret = otherService.createAccessToken(USER_ID, EMAIL);

            assertThatThrownBy(() -> jwtService.parse(tokenFromOtherSecret))
                    .isInstanceOf(Exception.class);
        }
    }
}
