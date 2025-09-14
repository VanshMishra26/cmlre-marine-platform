package com.cmlre.marine.platform.controller;

import com.cmlre.marine.platform.model.OceanographicData;
import com.cmlre.marine.platform.service.OceanographicDataService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * REST Controller for Oceanographic Data management
 */
@RestController
@RequestMapping("/oceanography")
@Tag(name = "Oceanographic Data", description = "APIs for managing oceanographic data")
@CrossOrigin(origins = "*")
public class OceanographicDataController {

    @Autowired
    private OceanographicDataService oceanographicDataService;

    @Operation(summary = "Get all oceanographic data", description = "Retrieve paginated oceanographic data with optional filtering")
    @GetMapping
    public ResponseEntity<Page<OceanographicData>> getAllOceanographicData(
            @Parameter(description = "Station ID filter") @RequestParam(optional = true) String stationId,
            @Parameter(description = "Cruise ID filter") @RequestParam(optional = true) String cruiseId,
            @Parameter(description = "Region filter") @RequestParam(optional = true) String region,
            @Parameter(description = "Start date filter") @RequestParam(optional = true) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date filter") @RequestParam(optional = true) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @Parameter(description = "Data source filter") @RequestParam(optional = true) String dataSource,
            Pageable pageable) {
        
        Page<OceanographicData> data = oceanographicDataService.findWithFilters(
            stationId, cruiseId, region, startDate, endDate, dataSource, pageable);
        return ResponseEntity.ok(data);
    }

    @Operation(summary = "Get oceanographic data by ID", description = "Retrieve specific oceanographic data record")
    @GetMapping("/{id}")
    public ResponseEntity<OceanographicData> getOceanographicDataById(@PathVariable String id) {
        Optional<OceanographicData> data = oceanographicDataService.findById(id);
        return data.map(ResponseEntity::ok)
                  .orElse(ResponseEntity.notFound().build());
    }

    @Operation(summary = "Create new oceanographic data", description = "Add new oceanographic data record")
    @PostMapping
    public ResponseEntity<OceanographicData> createOceanographicData(
            @Valid @RequestBody OceanographicData oceanographicData) {
        OceanographicData savedData = oceanographicDataService.save(oceanographicData);
        return ResponseEntity.status(HttpStatus.CREATED).body(savedData);
    }

    @Operation(summary = "Update oceanographic data", description = "Update existing oceanographic data record")
    @PutMapping("/{id}")
    public ResponseEntity<OceanographicData> updateOceanographicData(
            @PathVariable String id, @Valid @RequestBody OceanographicData oceanographicData) {
        Optional<OceanographicData> existingData = oceanographicDataService.findById(id);
        if (existingData.isPresent()) {
            oceanographicData.setId(id);
            OceanographicData updatedData = oceanographicDataService.save(oceanographicData);
            return ResponseEntity.ok(updatedData);
        }
        return ResponseEntity.notFound().build();
    }

    @Operation(summary = "Delete oceanographic data", description = "Delete oceanographic data record")
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteOceanographicData(@PathVariable String id) {
        if (oceanographicDataService.existsById(id)) {
            oceanographicDataService.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }

    @Operation(summary = "Bulk upload oceanographic data", description = "Upload multiple oceanographic data records from file")
    @PostMapping("/bulk-upload")
    public ResponseEntity<Map<String, Object>> bulkUploadOceanographicData(
            @RequestParam("file") MultipartFile file,
            @RequestParam(required = false) String dataSource) {
        
        Map<String, Object> result = oceanographicDataService.bulkUpload(file, dataSource);
        return ResponseEntity.ok(result);
    }

    @Operation(summary = "Get oceanographic data statistics", description = "Get statistical summary of oceanographic data")
    @GetMapping("/statistics")
    public ResponseEntity<Map<String, Object>> getOceanographicDataStatistics(
            @Parameter(description = "Region filter") @RequestParam(optional = true) String region,
            @Parameter(description = "Start date filter") @RequestParam(optional = true) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date filter") @RequestParam(optional = true) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        Map<String, Object> statistics = oceanographicDataService.getStatistics(region, startDate, endDate);
        return ResponseEntity.ok(statistics);
    }

    @Operation(summary = "Get data by location", description = "Get oceanographic data within geographic bounds")
    @GetMapping("/location")
    public ResponseEntity<List<OceanographicData>> getDataByLocation(
            @Parameter(description = "Minimum latitude") @RequestParam double minLat,
            @Parameter(description = "Maximum latitude") @RequestParam double maxLat,
            @Parameter(description = "Minimum longitude") @RequestParam double minLon,
            @Parameter(description = "Maximum longitude") @RequestParam double maxLon,
            @Parameter(description = "Minimum depth") @RequestParam(required = false) Double minDepth,
            @Parameter(description = "Maximum depth") @RequestParam(required = false) Double maxDepth) {
        
        List<OceanographicData> data = oceanographicDataService.findByLocationBounds(
            minLat, maxLat, minLon, maxLon, minDepth, maxDepth);
        return ResponseEntity.ok(data);
    }

    @Operation(summary = "Get data quality metrics", description = "Get data quality assessment metrics")
    @GetMapping("/quality-metrics")
    public ResponseEntity<Map<String, Object>> getDataQualityMetrics() {
        Map<String, Object> metrics = oceanographicDataService.getDataQualityMetrics();
        return ResponseEntity.ok(metrics);
    }

    @Operation(summary = "Export oceanographic data", description = "Export oceanographic data in various formats")
    @GetMapping("/export")
    public ResponseEntity<byte[]> exportOceanographicData(
            @Parameter(description = "Export format") @RequestParam String format,
            @Parameter(description = "Station ID filter") @RequestParam(optional = true) String stationId,
            @Parameter(description = "Cruise ID filter") @RequestParam(optional = true) String cruiseId,
            @Parameter(description = "Region filter") @RequestParam(optional = true) String region,
            @Parameter(description = "Start date filter") @RequestParam(optional = true) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date filter") @RequestParam(optional = true) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        byte[] exportData = oceanographicDataService.exportData(format, stationId, cruiseId, region, startDate, endDate);
        return ResponseEntity.ok()
                .header("Content-Disposition", "attachment; filename=oceanographic_data." + format)
                .body(exportData);
    }

    @Operation(summary = "Get time series data", description = "Get time series data for specific parameters")
    @GetMapping("/time-series")
    public ResponseEntity<Map<String, Object>> getTimeSeriesData(
            @Parameter(description = "Station ID") @RequestParam String stationId,
            @Parameter(description = "Parameter name") @RequestParam String parameter,
            @Parameter(description = "Start date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        Map<String, Object> timeSeries = oceanographicDataService.getTimeSeriesData(
            stationId, parameter, startDate, endDate);
        return ResponseEntity.ok(timeSeries);
    }
}
