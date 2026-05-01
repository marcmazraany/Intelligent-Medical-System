package com.medmanager;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("Medication Management Application Tests")
class MedicationManagementApplicationTests {

    @Test
    @DisplayName("Application entry point class is available")
    void applicationClassIsAvailable() {
        assertThat(MedicationManagementApplication.class).isNotNull();
    }
}
