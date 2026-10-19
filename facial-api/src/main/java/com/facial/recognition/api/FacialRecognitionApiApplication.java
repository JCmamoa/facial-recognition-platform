package com.facial.recognition.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

/**
 * Main Spring Boot application entry point for the Facial Recognition & Biometric Detection Platform.
 */
@SpringBootApplication(scanBasePackages = {
        "com.facial.recognition.api",
        "com.facial.recognition.core",
        "com.facial.recognition.dl4j",
        "com.facial.recognition.persistence",
        "com.facial.recognition.messaging"
})
@EntityScan(basePackages = "com.facial.recognition.persistence.entity")
@EnableJpaRepositories(basePackages = "com.facial.recognition.persistence.repository")
public class FacialRecognitionApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(FacialRecognitionApiApplication.class, args);
    }
}
