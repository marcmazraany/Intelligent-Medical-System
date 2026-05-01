package com.intellimeds.api.auth.dto;

import com.intellimeds.api.profile.dto.UserProfileDto;
import lombok.Builder;

@Builder
public record AuthResponse(
        String accessToken,
        String refreshToken,
        UserProfileDto profile
) {}
