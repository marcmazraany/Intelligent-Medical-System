package com.intellimeds.api.security;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Data
@ConfigurationProperties(prefix = "app.jwt")
public class JwtProperties {
    private String issuer;
    private String secret;
    private int accessTtlMinutes;
    private int refreshTtlDays;
}
