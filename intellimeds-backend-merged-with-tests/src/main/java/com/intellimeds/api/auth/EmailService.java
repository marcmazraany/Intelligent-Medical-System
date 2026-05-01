package com.intellimeds.api.auth;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailService {

    private static final Logger log = LoggerFactory.getLogger(EmailService.class);

    private final JavaMailSender mailSender;

    @Value("${spring.mail.username}")
    private String fromEmail;

    public EmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }

    /**
     * Send password reset verification code
     */
    public void sendVerificationCode(String toEmail, String code) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(toEmail);
            message.setSubject("IntelliMeds Password Reset Code");
            message.setText(buildVerificationEmailBody(code));

            mailSender.send(message);

            log.info("Verification code sent to: {}", toEmail);
        } catch (Exception e) {
            log.error("Failed to send verification email to: {}", toEmail, e);
            throw new RuntimeException("Failed to send verification email", e);
        }
    }

    /**
     * Send password reset confirmation
     */
    public void sendPasswordResetConfirmation(String toEmail) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(toEmail);
            message.setSubject("Your IntelliMeds Password Was Reset");
            message.setText(buildConfirmationEmailBody());

            mailSender.send(message);

            log.info("Password reset confirmation sent to: {}", toEmail);
        } catch (Exception e) {
            log.error("Failed to send confirmation email to: {}", toEmail, e);
            // Don't throw - confirmation email is not critical
        }
    }

    private String buildVerificationEmailBody(String code) {
        return String.format("""
            Hello,

            Your IntelliMeds password reset verification code is:

            %s

            This code will expire in 10 minutes.

            If you didn't request this password reset, please ignore this email.

            Best regards,
            The IntelliMeds Team
            """, code);
    }

    private String buildConfirmationEmailBody() {
        return """
            Hello,

            Your IntelliMeds password has been successfully reset.

            If you did not make this change, please contact our support team immediately.

            Best regards,
            The IntelliMeds Team
            """;
    }
}