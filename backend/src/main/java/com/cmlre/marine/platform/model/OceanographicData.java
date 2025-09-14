package com.cmlre.marine.platform.model;

import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import jakarta.validation.constraints.*;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Oceanographic data entity representing physical, chemical, and biological
 * oceanographic measurements collected during marine surveys.
 */
@Document(collection = "oceanographic_data")
public class OceanographicData {
    
    @Id
    private String id;
    
    @NotBlank(message = "Station ID is required")
    @Indexed
    private String stationId;
    
    @NotBlank(message = "Cruise ID is required")
    @Indexed
    private String cruiseId;
    
    @NotBlank(message = "Expedition is required")
    private String expedition;
    
    @NotNull(message = "Location is required")
    private Location location;
    
    private PhysicalParameters physical;
    private ChemicalParameters chemical;
    private BiologicalParameters biological;
    private EnvironmentalConditions environmental;
    
    @NotNull(message = "Collection date is required")
    @Indexed
    private LocalDateTime collectionDate;
    
    @NotBlank(message = "Collection method is required")
    private String collectionMethod;
    
    private List<String> instruments;
    private QualityControl qualityControl;
    
    @NotBlank(message = "Data source is required")
    private String dataSource;
    
    private String processingLevel = "raw";
    private List<String> rawDataFiles;
    private List<String> processedDataFiles;
    private String notes;
    private List<String> tags;
    
    @CreatedDate
    private LocalDateTime createdAt;
    
    @LastModifiedDate
    private LocalDateTime updatedAt;
    
    // Nested classes for structured data
    public static class Location {
        @NotNull(message = "Latitude is required")
        @DecimalMin(value = "-90.0", message = "Latitude must be between -90 and 90")
        @DecimalMax(value = "90.0", message = "Latitude must be between -90 and 90")
        private Double latitude;
        
        @NotNull(message = "Longitude is required")
        @DecimalMin(value = "-180.0", message = "Longitude must be between -180 and 180")
        @DecimalMax(value = "180.0", message = "Longitude must be between -180 and 180")
        private Double longitude;
        
        @NotNull(message = "Depth is required")
        @Min(value = 0, message = "Depth must be non-negative")
        private Double depth;
        
        @NotBlank(message = "Region is required")
        private String region;
        
        // Getters and setters
        public Double getLatitude() { return latitude; }
        public void setLatitude(Double latitude) { this.latitude = latitude; }
        public Double getLongitude() { return longitude; }
        public void setLongitude(Double longitude) { this.longitude = longitude; }
        public Double getDepth() { return depth; }
        public void setDepth(Double depth) { this.depth = depth; }
        public String getRegion() { return region; }
        public void setRegion(String region) { this.region = region; }
    }
    
    public static class PhysicalParameters {
        private TemperatureData temperature;
        private SalinityData salinity;
        private Double density;
        private CurrentData currents;
        private WaveData waves;
        
        // Getters and setters
        public TemperatureData getTemperature() { return temperature; }
        public void setTemperature(TemperatureData temperature) { this.temperature = temperature; }
        public SalinityData getSalinity() { return salinity; }
        public void setSalinity(SalinityData salinity) { this.salinity = salinity; }
        public Double getDensity() { return density; }
        public void setDensity(Double density) { this.density = density; }
        public CurrentData getCurrents() { return currents; }
        public void setCurrents(CurrentData currents) { this.currents = currents; }
        public WaveData getWaves() { return waves; }
        public void setWaves(WaveData waves) { this.waves = waves; }
    }
    
    public static class TemperatureData {
        private Double surface;
        private Double bottom;
        private List<DepthProfile> profile;
        
        // Getters and setters
        public Double getSurface() { return surface; }
        public void setSurface(Double surface) { this.surface = surface; }
        public Double getBottom() { return bottom; }
        public void setBottom(Double bottom) { this.bottom = bottom; }
        public List<DepthProfile> getProfile() { return profile; }
        public void setProfile(List<DepthProfile> profile) { this.profile = profile; }
    }
    
    public static class SalinityData {
        private Double surface;
        private Double bottom;
        private List<DepthProfile> profile;
        
        // Getters and setters
        public Double getSurface() { return surface; }
        public void setSurface(Double surface) { this.surface = surface; }
        public Double getBottom() { return bottom; }
        public void setBottom(Double bottom) { this.bottom = bottom; }
        public List<DepthProfile> getProfile() { return profile; }
        public void setProfile(List<DepthProfile> profile) { this.profile = profile; }
    }
    
    public static class DepthProfile {
        private Double depth;
        private Double value;
        
        // Getters and setters
        public Double getDepth() { return depth; }
        public void setDepth(Double depth) { this.depth = depth; }
        public Double getValue() { return value; }
        public void setValue(Double value) { this.value = value; }
    }
    
    public static class CurrentData {
        private VelocityData surface;
        private VelocityData bottom;
        
        // Getters and setters
        public VelocityData getSurface() { return surface; }
        public void setSurface(VelocityData surface) { this.surface = surface; }
        public VelocityData getBottom() { return bottom; }
        public void setBottom(VelocityData bottom) { this.bottom = bottom; }
    }
    
    public static class VelocityData {
        private Double speed;
        private Double direction;
        
        // Getters and setters
        public Double getSpeed() { return speed; }
        public void setSpeed(Double speed) { this.speed = speed; }
        public Double getDirection() { return direction; }
        public void setDirection(Double direction) { this.direction = direction; }
    }
    
    public static class WaveData {
        private Double height;
        private Double period;
        private Double direction;
        
        // Getters and setters
        public Double getHeight() { return height; }
        public void setHeight(Double height) { this.height = height; }
        public Double getPeriod() { return period; }
        public void setPeriod(Double period) { this.period = period; }
        public Double getDirection() { return direction; }
        public void setDirection(Double direction) { this.direction = direction; }
    }
    
    public static class ChemicalParameters {
        private OxygenData dissolvedOxygen;
        private NutrientData nutrients;
        private Double pH;
        private Double alkalinity;
        private Double carbonDioxide;
        private ChlorophyllData chlorophyll;
        
        // Getters and setters
        public OxygenData getDissolvedOxygen() { return dissolvedOxygen; }
        public void setDissolvedOxygen(OxygenData dissolvedOxygen) { this.dissolvedOxygen = dissolvedOxygen; }
        public NutrientData getNutrients() { return nutrients; }
        public void setNutrients(NutrientData nutrients) { this.nutrients = nutrients; }
        public Double getpH() { return pH; }
        public void setpH(Double pH) { this.pH = pH; }
        public Double getAlkalinity() { return alkalinity; }
        public void setAlkalinity(Double alkalinity) { this.alkalinity = alkalinity; }
        public Double getCarbonDioxide() { return carbonDioxide; }
        public void setCarbonDioxide(Double carbonDioxide) { this.carbonDioxide = carbonDioxide; }
        public ChlorophyllData getChlorophyll() { return chlorophyll; }
        public void setChlorophyll(ChlorophyllData chlorophyll) { this.chlorophyll = chlorophyll; }
    }
    
    public static class OxygenData {
        private Double surface;
        private Double bottom;
        private List<DepthProfile> profile;
        
        // Getters and setters
        public Double getSurface() { return surface; }
        public void setSurface(Double surface) { this.surface = surface; }
        public Double getBottom() { return bottom; }
        public void setBottom(Double bottom) { this.bottom = bottom; }
        public List<DepthProfile> getProfile() { return profile; }
        public void setProfile(List<DepthProfile> profile) { this.profile = profile; }
    }
    
    public static class NutrientData {
        private Double nitrate;
        private Double phosphate;
        private Double silicate;
        private Double ammonia;
        
        // Getters and setters
        public Double getNitrate() { return nitrate; }
        public void setNitrate(Double nitrate) { this.nitrate = nitrate; }
        public Double getPhosphate() { return phosphate; }
        public void setPhosphate(Double phosphate) { this.phosphate = phosphate; }
        public Double getSilicate() { return silicate; }
        public void setSilicate(Double silicate) { this.silicate = silicate; }
        public Double getAmmonia() { return ammonia; }
        public void setAmmonia(Double ammonia) { this.ammonia = ammonia; }
    }
    
    public static class ChlorophyllData {
        private Double surface;
        private List<DepthProfile> profile;
        
        // Getters and setters
        public Double getSurface() { return surface; }
        public void setSurface(Double surface) { this.surface = surface; }
        public List<DepthProfile> getProfile() { return profile; }
        public void setProfile(List<DepthProfile> profile) { this.profile = profile; }
    }
    
    public static class BiologicalParameters {
        private Double primaryProduction;
        private Double zooplanktonBiomass;
        private Double phytoplanktonBiomass;
        private Double bacterialAbundance;
        private Integer speciesRichness;
        
        // Getters and setters
        public Double getPrimaryProduction() { return primaryProduction; }
        public void setPrimaryProduction(Double primaryProduction) { this.primaryProduction = primaryProduction; }
        public Double getZooplanktonBiomass() { return zooplanktonBiomass; }
        public void setZooplanktonBiomass(Double zooplanktonBiomass) { this.zooplanktonBiomass = zooplanktonBiomass; }
        public Double getPhytoplanktonBiomass() { return phytoplanktonBiomass; }
        public void setPhytoplanktonBiomass(Double phytoplanktonBiomass) { this.phytoplanktonBiomass = phytoplanktonBiomass; }
        public Double getBacterialAbundance() { return bacterialAbundance; }
        public void setBacterialAbundance(Double bacterialAbundance) { this.bacterialAbundance = bacterialAbundance; }
        public Integer getSpeciesRichness() { return speciesRichness; }
        public void setSpeciesRichness(Integer speciesRichness) { this.speciesRichness = speciesRichness; }
    }
    
    public static class EnvironmentalConditions {
        private WeatherData weather;
        private SeaStateData seaState;
        private Double visibility;
        private Double cloudCover;
        
        // Getters and setters
        public WeatherData getWeather() { return weather; }
        public void setWeather(WeatherData weather) { this.weather = weather; }
        public SeaStateData getSeaState() { return seaState; }
        public void setSeaState(SeaStateData seaState) { this.seaState = seaState; }
        public Double getVisibility() { return visibility; }
        public void setVisibility(Double visibility) { this.visibility = visibility; }
        public Double getCloudCover() { return cloudCover; }
        public void setCloudCover(Double cloudCover) { this.cloudCover = cloudCover; }
    }
    
    public static class WeatherData {
        private Double windSpeed;
        private Double windDirection;
        private Double airTemperature;
        private Double humidity;
        private Double pressure;
        
        // Getters and setters
        public Double getWindSpeed() { return windSpeed; }
        public void setWindSpeed(Double windSpeed) { this.windSpeed = windSpeed; }
        public Double getWindDirection() { return windDirection; }
        public void setWindDirection(Double windDirection) { this.windDirection = windDirection; }
        public Double getAirTemperature() { return airTemperature; }
        public void setAirTemperature(Double airTemperature) { this.airTemperature = airTemperature; }
        public Double getHumidity() { return humidity; }
        public void setHumidity(Double humidity) { this.humidity = humidity; }
        public Double getPressure() { return pressure; }
        public void setPressure(Double pressure) { this.pressure = pressure; }
    }
    
    public static class SeaStateData {
        private String type;
        private String description;
        
        // Getters and setters
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }
    
    public static class QualityControl {
        private String status = "pending";
        private String notes;
        private String reviewedBy;
        private LocalDateTime reviewedAt;
        
        // Getters and setters
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getNotes() { return notes; }
        public void setNotes(String notes) { this.notes = notes; }
        public String getReviewedBy() { return reviewedBy; }
        public void setReviewedBy(String reviewedBy) { this.reviewedBy = reviewedBy; }
        public LocalDateTime getReviewedAt() { return reviewedAt; }
        public void setReviewedAt(LocalDateTime reviewedAt) { this.reviewedAt = reviewedAt; }
    }
    
    // Main entity getters and setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getStationId() { return stationId; }
    public void setStationId(String stationId) { this.stationId = stationId; }
    public String getCruiseId() { return cruiseId; }
    public void setCruiseId(String cruiseId) { this.cruiseId = cruiseId; }
    public String getExpedition() { return expedition; }
    public void setExpedition(String expedition) { this.expedition = expedition; }
    public Location getLocation() { return location; }
    public void setLocation(Location location) { this.location = location; }
    public PhysicalParameters getPhysical() { return physical; }
    public void setPhysical(PhysicalParameters physical) { this.physical = physical; }
    public ChemicalParameters getChemical() { return chemical; }
    public void setChemical(ChemicalParameters chemical) { this.chemical = chemical; }
    public BiologicalParameters getBiological() { return biological; }
    public void setBiological(BiologicalParameters biological) { this.biological = biological; }
    public EnvironmentalConditions getEnvironmental() { return environmental; }
    public void setEnvironmental(EnvironmentalConditions environmental) { this.environmental = environmental; }
    public LocalDateTime getCollectionDate() { return collectionDate; }
    public void setCollectionDate(LocalDateTime collectionDate) { this.collectionDate = collectionDate; }
    public String getCollectionMethod() { return collectionMethod; }
    public void setCollectionMethod(String collectionMethod) { this.collectionMethod = collectionMethod; }
    public List<String> getInstruments() { return instruments; }
    public void setInstruments(List<String> instruments) { this.instruments = instruments; }
    public QualityControl getQualityControl() { return qualityControl; }
    public void setQualityControl(QualityControl qualityControl) { this.qualityControl = qualityControl; }
    public String getDataSource() { return dataSource; }
    public void setDataSource(String dataSource) { this.dataSource = dataSource; }
    public String getProcessingLevel() { return processingLevel; }
    public void setProcessingLevel(String processingLevel) { this.processingLevel = processingLevel; }
    public List<String> getRawDataFiles() { return rawDataFiles; }
    public void setRawDataFiles(List<String> rawDataFiles) { this.rawDataFiles = rawDataFiles; }
    public List<String> getProcessedDataFiles() { return processedDataFiles; }
    public void setProcessedDataFiles(List<String> processedDataFiles) { this.processedDataFiles = processedDataFiles; }
    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }
    public List<String> getTags() { return tags; }
    public void setTags(List<String> tags) { this.tags = tags; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
