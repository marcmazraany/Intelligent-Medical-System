package com.intellimeds.api.ai;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

@Configuration
public class AiHttpConfig {

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}