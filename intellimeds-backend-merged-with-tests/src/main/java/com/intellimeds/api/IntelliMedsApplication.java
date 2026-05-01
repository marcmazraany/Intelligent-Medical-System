package com.intellimeds.api;

import com.intellimeds.api.ai.AiServiceProperties;
import com.intellimeds.api.medications.MedicationScannerProperties;
import com.intellimeds.api.pharmacy.config.PharmacyFinderProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties({
        AiServiceProperties.class,
        MedicationScannerProperties.class,
        PharmacyFinderProperties.class
})
public class IntelliMedsApplication {
    public static void main(String[] args) {
        SpringApplication.run(IntelliMedsApplication.class, args);
    }
}
