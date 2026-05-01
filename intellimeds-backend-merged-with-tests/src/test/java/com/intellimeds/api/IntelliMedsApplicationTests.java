package com.intellimeds.api;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("Medication Management Application Tests")
class IntelliMedsApplicationTests {

    @Test
    @DisplayName("Application entry point class is available")
    void applicationEntryPointClassIsAvailable() {
        assertThat(IntelliMedsApplication.class).isNotNull();
    }
}
