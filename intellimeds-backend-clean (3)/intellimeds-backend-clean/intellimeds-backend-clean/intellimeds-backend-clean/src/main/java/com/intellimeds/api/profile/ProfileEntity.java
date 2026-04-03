package com.intellimeds.api.profile;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDate;
import java.util.UUID;

@Entity
@Table(name = "profiles")
@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ProfileEntity {

    @Id
    @Column(name = "user_id", nullable = false, updatable = false)
    private UUID userId;

    @OneToOne(fetch = FetchType.LAZY)
    @MapsId
    @JoinColumn(name = "user_id")
    private com.intellimeds.api.users.UserEntity user;

    @Column(name = "first_name")
    private String firstName;

    @Column(name = "last_name")
    private String lastName;

    private LocalDate dob;
    private String gender;
    private String height;
    private String weight;
    private String allergies;

    @Column(name = "blood_type")
    private String bloodType;

    @Column(name = "chronic_conditions")
    private String chronicConditions;

    @Column(columnDefinition = "text")
    private String notes;
}
