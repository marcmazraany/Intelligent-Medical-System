package com.medmanager.config;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

@Component
@Slf4j
public class SimulatedDelayInterceptor implements HandlerInterceptor {

    @Value("${test.simulate.network.delay:false}")
    private boolean simulateDelay;

    @Value("${test.simulate.delay.ms:150}")
    private int delayMs;

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        if (simulateDelay && request.getRequestURI().contains("/api/pharmacy/inventory")) {
            try {
                Thread.sleep(delayMs);
                log.debug("Simulated {}ms delay", delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        return true;
    }
}