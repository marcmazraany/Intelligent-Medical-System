package com.intellimeds.api.profile;

import com.intellimeds.api.common.NotFoundException;
import com.intellimeds.api.profile.dto.UserProfileDto;
import com.intellimeds.api.users.UserEntity;
import com.intellimeds.api.users.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.UUID;

@Service
public class ProfileService {

    private final ProfileRepository profiles;
    private final UserRepository users;

    public ProfileService(ProfileRepository profiles, UserRepository users) {
        this.profiles = profiles;
        this.users = users;
    }

    @Transactional(readOnly = true)
    public UserProfileDto get(UUID userId) {
        ProfileEntity p = profiles.findById(userId)
                .orElseThrow(() -> new NotFoundException("Profile not found"));
        return toDto(p);
    }

    @Transactional
    public UserProfileDto upsert(UUID userId, UserProfileDto dto) {
        UserEntity user = users.findById(userId)
                .orElseThrow(() -> new NotFoundException("User not found"));

        ProfileEntity p = profiles.findById(userId)
                .orElse(ProfileEntity.builder().user(user).build());

        p.setFirstName(dto.firstName());
        p.setLastName(dto.lastName());
        p.setDob(dto.dob());

        p.setGender(dto.gender());
        p.setHeight(dto.height());
        p.setWeight(dto.weight());
        p.setAllergies(dto.allergies());
        p.setBloodType(dto.bloodType());
        p.setChronicConditions(dto.chronicConditions());
        p.setNotes(dto.notes());

        ProfileEntity saved = profiles.save(p);
        return toDto(saved);
    }

    private static UserProfileDto toDto(ProfileEntity p) {
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
}