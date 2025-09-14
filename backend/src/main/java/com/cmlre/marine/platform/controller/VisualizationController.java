package com.cmlre.marine.platform.controller;

import com.cmlre.marine.platform.service.VisualizationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * REST Controller for Data Visualization
 */
@RestController
@RequestMapping("/visualization")
@Tag(name = "Visualization", description = "APIs for data visualization and analytics")
@CrossOrigin(origins = "*")
public class VisualizationController {

    @Autowired
    private VisualizationService visualizationService;

    @Operation(summary = "Get oceanographic trends", description = "Get time series trends for oceanographic parameters")
    @GetMapping("/oceanographic-trends")
    public ResponseEntity<Map<String, Object>> getOceanographicTrends(
            @Parameter(description = "Region filter") @RequestParam(required = false) String region,
            @Parameter(description = "Parameter name") @RequestParam String parameter,
            @Parameter(description = "Start date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @Parameter(description = "Aggregation period") @RequestParam(defaultValue = "monthly") String period) {
        
        Map<String, Object> trends = visualizationService.getOceanographicTrends(
            region, parameter, startDate, endDate, period);
        return ResponseEntity.ok(trends);
    }

    @Operation(summary = "Get biodiversity trends", description = "Get biodiversity trends and species richness over time")
    @GetMapping("/biodiversity-trends")
    public ResponseEntity<Map<String, Object>> getBiodiversityTrends(
            @Parameter(description = "Region filter") @RequestParam(required = false) String region,
            @Parameter(description = "Start date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        Map<String, Object> trends = visualizationService.getBiodiversityTrends(region, startDate, endDate);
        return ResponseEntity.ok(trends);
    }

    @Operation(summary = "Get spatial distribution", description = "Get spatial distribution of marine data")
    @GetMapping("/spatial-distribution")
    public ResponseEntity<Map<String, Object>> getSpatialDistribution(
            @Parameter(description = "Data type") @RequestParam String dataType,
            @Parameter(description = "Parameter name") @RequestParam String parameter,
            @Parameter(description = "Start date") @RequestParam(required = false) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam(required = false) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        Map<String, Object> distribution = visualizationService.getSpatialDistribution(
            dataType, parameter, startDate, endDate);
        return ResponseEntity.ok(distribution);
    }

    @Operation(summary = "Get correlation matrix", description = "Get correlation matrix between different parameters")
    @GetMapping("/correlation-matrix")
    public ResponseEntity<Map<String, Object>> getCorrelationMatrix(
            @Parameter(description = "Region filter") @RequestParam(required = false) String region,
            @Parameter(description = "Start date") @RequestParam(required = false) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam(required = false) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @Parameter(description = "Parameters to include") @RequestParam(required = false) String[] parameters) {
        
        Map<String, Object> correlation = visualizationService.getCorrelationMatrix(
            region, startDate, endDate, parameters);
        return ResponseEntity.ok(correlation);
    }

    @Operation(summary = "Get depth profile", description = "Get vertical depth profile for oceanographic parameters")
    @GetMapping("/depth-profile")
    public ResponseEntity<Map<String, Object>> getDepthProfile(
            @Parameter(description = "Station ID") @RequestParam String stationId,
            @Parameter(description = "Parameter name") @RequestParam String parameter,
            @Parameter(description = "Collection date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime collectionDate) {
        
        Map<String, Object> profile = visualizationService.getDepthProfile(stationId, parameter, collectionDate);
        return ResponseEntity.ok(profile);
    }

    @Operation(summary = "Get species distribution", description = "Get spatial distribution of species")
    @GetMapping("/species-distribution")
    public ResponseEntity<Map<String, Object>> getSpeciesDistribution(
            @Parameter(description = "Species name") @RequestParam(required = false) String speciesName,
            @Parameter(description = "Taxonomic level") @RequestParam(required = false) String taxonomicLevel,
            @Parameter(description = "Region filter") @RequestParam(required = false) String region) {
        
        Map<String, Object> distribution = visualizationService.getSpeciesDistribution(
            speciesName, taxonomicLevel, region);
        return ResponseEntity.ok(distribution);
    }

    @Operation(summary = "Get environmental factors", description = "Get environmental factors affecting marine ecosystems")
    @GetMapping("/environmental-factors")
    public ResponseEntity<Map<String, Object>> getEnvironmentalFactors(
            @Parameter(description = "Region filter") @RequestParam(required = false) String region,
            @Parameter(description = "Start date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        Map<String, Object> factors = visualizationService.getEnvironmentalFactors(region, startDate, endDate);
        return ResponseEntity.ok(factors);
    }

    @Operation(summary = "Get data quality dashboard", description = "Get data quality metrics and status dashboard")
    @GetMapping("/data-quality-dashboard")
    public ResponseEntity<Map<String, Object>> getDataQualityDashboard() {
        Map<String, Object> dashboard = visualizationService.getDataQualityDashboard();
        return ResponseEntity.ok(dashboard);
    }

    @Operation(summary = "Get real-time monitoring", description = "Get real-time monitoring data and alerts")
    @GetMapping("/real-time-monitoring")
    public ResponseEntity<Map<String, Object>> getRealTimeMonitoring(
            @Parameter(description = "Region filter") @RequestParam(required = false) String region,
            @Parameter(description = "Alert threshold") @RequestParam(defaultValue = "0.8") double threshold) {
        
        Map<String, Object> monitoring = visualizationService.getRealTimeMonitoring(region, threshold);
        return ResponseEntity.ok(monitoring);
    }

    @Operation(summary = "Generate custom visualization", description = "Generate custom visualization based on user parameters")
    @PostMapping("/custom")
    public ResponseEntity<Map<String, Object>> generateCustomVisualization(
            @RequestBody Map<String, Object> visualizationConfig) {
        
        Map<String, Object> visualization = visualizationService.generateCustomVisualization(visualizationConfig);
        return ResponseEntity.ok(visualization);
    }
}
