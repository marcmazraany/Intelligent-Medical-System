package com.intellimeds.api.profile.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Builder;

import java.time.LocalDate;

@Builder
public record UserProfileDto(
        String firstName,
        String lastName,
        LocalDate dob,
        String gender,
        String height,
        String weight,
        String allergies,
        String bloodType,
        String chronicConditions,
        String notes
) {}
