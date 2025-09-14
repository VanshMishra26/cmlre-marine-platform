package com.cmlre.marine.platform.service;

import com.cmlre.marine.platform.model.OceanographicData;
import com.cmlre.marine.platform.repository.OceanographicDataRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.aggregation.Aggregation;
import org.springframework.data.mongodb.core.aggregation.AggregationResults;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;
import java.util.*;

import static org.springframework.data.mongodb.core.aggregation.Aggregation.*;

/**
 * Service class for Oceanographic Data operations
 */
@Service
public class OceanographicDataService {

    @Autowired
    private OceanographicDataRepository oceanographicDataRepository;

    @Autowired
    private MongoTemplate mongoTemplate;

    @Autowired
    private FileProcessingService fileProcessingService;

    public Page<OceanographicData> findWithFilters(String stationId, String cruiseId, String region, 
                                                  LocalDateTime startDate, LocalDateTime endDate, 
                                                  String dataSource, Pageable pageable) {
        return oceanographicDataRepository.findWithFilters(stationId, cruiseId, region, 
                                                          startDate, endDate, dataSource, pageable);
    }

    public Optional<OceanographicData> findById(String id) {
        return oceanographicDataRepository.findById(id);
    }

    public OceanographicData save(OceanographicData oceanographicData) {
        return oceanographicDataRepository.save(oceanographicData);
    }

    public boolean existsById(String id) {
        return oceanographicDataRepository.existsById(id);
    }

    public void deleteById(String id) {
        oceanographicDataRepository.deleteById(id);
    }

    public Map<String, Object> bulkUpload(MultipartFile file, String dataSource) {
        try {
            List<OceanographicData> dataList = fileProcessingService.processOceanographicDataFile(file, dataSource);
            List<OceanographicData> savedData = oceanographicDataRepository.saveAll(dataList);
            
            Map<String, Object> result = new HashMap<>();
            result.put("success", true);
            result.put("totalRecords", savedData.size());
            result.put("message", "Successfully uploaded " + savedData.size() + " records");
            result.put("data", savedData);
            
            return result;
        } catch (Exception e) {
            Map<String, Object> result = new HashMap<>();
            result.put("success", false);
            result.put("error", e.getMessage());
            return result;
        }
    }

    public Map<String, Object> getStatistics(String region, LocalDateTime startDate, LocalDateTime endDate) {
        Criteria criteria = new Criteria();
        
        if (region != null) {
            criteria.and("location.region").is(region);
        }
        if (startDate != null) {
            criteria.and("collectionDate").gte(startDate);
        }
        if (endDate != null) {
            criteria.and("collectionDate").lte(endDate);
        }

        Aggregation aggregation = newAggregation(
            match(criteria),
            group()
                .count().as("totalRecords")
                .avg("physical.temperature.surface").as("avgSurfaceTemp")
                .avg("physical.temperature.bottom").as("avgBottomTemp")
                .avg("physical.salinity.surface").as("avgSurfaceSalinity")
                .avg("physical.salinity.bottom").as("avgBottomSalinity")
                .avg("chemical.dissolvedOxygen.surface").as("avgSurfaceOxygen")
                .avg("chemical.dissolvedOxygen.bottom").as("avgBottomOxygen")
                .avg("biological.primaryProduction").as("avgPrimaryProduction")
                .avg("biological.speciesRichness").as("avgSpeciesRichness")
        );

        AggregationResults<Map> results = mongoTemplate.aggregate(aggregation, "oceanographic_data", Map.class);
        Map<String, Object> statistics = results.getUniqueMappedResult();
        
        if (statistics == null) {
            statistics = new HashMap<>();
            statistics.put("totalRecords", 0);
        }

        return statistics;
    }

    public List<OceanographicData> findByLocationBounds(double minLat, double maxLat, double minLon, 
                                                       double maxLon, Double minDepth, Double maxDepth) {
        Criteria criteria = new Criteria();
        criteria.and("location.latitude").gte(minLat).lte(maxLat);
        criteria.and("location.longitude").gte(minLon).lte(maxLon);
        
        if (minDepth != null) {
            criteria.and("location.depth").gte(minDepth);
        }
        if (maxDepth != null) {
            criteria.and("location.depth").lte(maxDepth);
        }

        Query query = new Query(criteria);
        return mongoTemplate.find(query, OceanographicData.class);
    }

    public Map<String, Object> getDataQualityMetrics() {
        Aggregation aggregation = newAggregation(
            group()
                .count().as("totalRecords")
                .sum(Criteria.where("qualityControl.status").is("approved")).as("approvedRecords")
                .sum(Criteria.where("qualityControl.status").is("pending")).as("pendingRecords")
                .sum(Criteria.where("qualityControl.status").is("rejected")).as("rejectedRecords")
                .sum(Criteria.where("qualityControl.status").is("needs_review")).as("needsReviewRecords")
        );

        AggregationResults<Map> results = mongoTemplate.aggregate(aggregation, "oceanographic_data", Map.class);
        Map<String, Object> metrics = results.getUniqueMappedResult();
        
        if (metrics == null) {
            metrics = new HashMap<>();
            metrics.put("totalRecords", 0);
            metrics.put("approvedRecords", 0);
            metrics.put("pendingRecords", 0);
            metrics.put("rejectedRecords", 0);
            metrics.put("needsReviewRecords", 0);
        }

        // Calculate quality percentage
        int totalRecords = (Integer) metrics.getOrDefault("totalRecords", 0);
        int approvedRecords = (Integer) metrics.getOrDefault("approvedRecords", 0);
        double qualityPercentage = totalRecords > 0 ? (double) approvedRecords / totalRecords * 100 : 0;
        metrics.put("qualityPercentage", qualityPercentage);

        return metrics;
    }

    public byte[] exportData(String format, String stationId, String cruiseId, String region, 
                           LocalDateTime startDate, LocalDateTime endDate) {
        List<OceanographicData> data = findDataForExport(stationId, cruiseId, region, startDate, endDate);
        return fileProcessingService.exportOceanographicData(data, format);
    }

    private List<OceanographicData> findDataForExport(String stationId, String cruiseId, String region, 
                                                     LocalDateTime startDate, LocalDateTime endDate) {
        Criteria criteria = new Criteria();
        
        if (stationId != null) {
            criteria.and("stationId").is(stationId);
        }
        if (cruiseId != null) {
            criteria.and("cruiseId").is(cruiseId);
        }
        if (region != null) {
            criteria.and("location.region").is(region);
        }
        if (startDate != null) {
            criteria.and("collectionDate").gte(startDate);
        }
        if (endDate != null) {
            criteria.and("collectionDate").lte(endDate);
        }

        Query query = new Query(criteria);
        return mongoTemplate.find(query, OceanographicData.class);
    }

    public Map<String, Object> getTimeSeriesData(String stationId, String parameter, 
                                                LocalDateTime startDate, LocalDateTime endDate) {
        Criteria criteria = new Criteria();
        criteria.and("stationId").is(stationId);
        criteria.and("collectionDate").gte(startDate).lte(endDate);

        Query query = new Query(criteria);
        List<OceanographicData> data = mongoTemplate.find(query, OceanographicData.class);

        List<Map<String, Object>> timeSeries = new ArrayList<>();
        for (OceanographicData record : data) {
            Map<String, Object> point = new HashMap<>();
            point.put("timestamp", record.getCollectionDate());
            point.put("value", extractParameterValue(record, parameter));
            timeSeries.add(point);
        }

        Map<String, Object> result = new HashMap<>();
        result.put("stationId", stationId);
        result.put("parameter", parameter);
        result.put("timeSeries", timeSeries);
        result.put("dataPoints", timeSeries.size());

        return result;
    }

    private Object extractParameterValue(OceanographicData record, String parameter) {
        // Extract parameter value based on parameter name
        switch (parameter.toLowerCase()) {
            case "surface_temperature":
                return record.getPhysical() != null && record.getPhysical().getTemperature() != null 
                    ? record.getPhysical().getTemperature().getSurface() : null;
            case "bottom_temperature":
                return record.getPhysical() != null && record.getPhysical().getTemperature() != null 
                    ? record.getPhysical().getTemperature().getBottom() : null;
            case "surface_salinity":
                return record.getPhysical() != null && record.getPhysical().getSalinity() != null 
                    ? record.getPhysical().getSalinity().getSurface() : null;
            case "bottom_salinity":
                return record.getPhysical() != null && record.getPhysical().getSalinity() != null 
                    ? record.getPhysical().getSalinity().getBottom() : null;
            case "surface_oxygen":
                return record.getChemical() != null && record.getChemical().getDissolvedOxygen() != null 
                    ? record.getChemical().getDissolvedOxygen().getSurface() : null;
            case "bottom_oxygen":
                return record.getChemical() != null && record.getChemical().getDissolvedOxygen() != null 
                    ? record.getChemical().getDissolvedOxygen().getBottom() : null;
            case "primary_production":
                return record.getBiological() != null ? record.getBiological().getPrimaryProduction() : null;
            case "species_richness":
                return record.getBiological() != null ? record.getBiological().getSpeciesRichness() : null;
            default:
                return null;
        }
    }
}
