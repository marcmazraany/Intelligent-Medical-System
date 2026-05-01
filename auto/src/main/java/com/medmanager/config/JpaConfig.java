package com.medmanager.config;

import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@Configuration
@EnableJpaRepositories(basePackages = "com.medmanager.repository")
@EntityScan(basePackages = "com.medmanager.entity")
public class JpaConfig {
}