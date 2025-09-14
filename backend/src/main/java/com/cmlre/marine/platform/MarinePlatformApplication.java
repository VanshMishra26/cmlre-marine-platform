package com.cmlre.marine.platform;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.data.mongodb.config.EnableMongoAuditing;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Main Spring Boot application class for CMLRE Marine Platform
 * 
 * This application provides a comprehensive platform for marine data integration,
 * analysis, and visualization, supporting oceanographic, taxonomic, morphological,
 * and molecular biology data.
 */
@SpringBootApplication
@EnableMongoAuditing
@EnableCaching
@EnableAsync
@EnableScheduling
public class MarinePlatformApplication {

    public static void main(String[] args) {
        SpringApplication.run(MarinePlatformApplication.class, args);
    }
}
