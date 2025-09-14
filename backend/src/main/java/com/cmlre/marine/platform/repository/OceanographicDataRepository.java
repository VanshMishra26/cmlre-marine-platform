package com.cmlre.marine.platform.repository;

import com.cmlre.marine.platform.model.OceanographicData;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;

/**
 * Repository interface for Oceanographic Data
 */
@Repository
public interface OceanographicDataRepository extends MongoRepository<OceanographicData, String> {

    @Query("{'stationId': ?0, 'cruiseId': ?1, 'location.region': ?2, 'collectionDate': {$gte: ?3, $lte: ?4}, 'dataSource': ?5}")
    Page<OceanographicData> findWithFilters(String stationId, String cruiseId, String region, 
                                           LocalDateTime startDate, LocalDateTime endDate, 
                                           String dataSource, Pageable pageable);

    @Query("{'stationId': ?0}")
    Page<OceanographicData> findByStationId(String stationId, Pageable pageable);

    @Query("{'cruiseId': ?0}")
    Page<OceanographicData> findByCruiseId(String cruiseId, Pageable pageable);

    @Query("{'location.region': ?0}")
    Page<OceanographicData> findByRegion(String region, Pageable pageable);

    @Query("{'collectionDate': {$gte: ?0, $lte: ?1}}")
    Page<OceanographicData> findByCollectionDateBetween(LocalDateTime startDate, LocalDateTime endDate, Pageable pageable);

    @Query("{'dataSource': ?0}")
    Page<OceanographicData> findByDataSource(String dataSource, Pageable pageable);

    @Query("{'qualityControl.status': ?0}")
    Page<OceanographicData> findByQualityControlStatus(String status, Pageable pageable);

    @Query("{'location.latitude': {$gte: ?0, $lte: ?1}, 'location.longitude': {$gte: ?2, $lte: ?3}}")
    Page<OceanographicData> findByLocationBounds(double minLat, double maxLat, double minLon, double maxLon, Pageable pageable);

    @Query("{'location.region': ?0, 'collectionDate': {$gte: ?1, $lte: ?2}}")
    Page<OceanographicData> findByRegionAndCollectionDateBetween(String region, LocalDateTime startDate, LocalDateTime endDate, Pageable pageable);

    @Query(value = "{}", count = true)
    long countAll();

    @Query(value = "{'qualityControl.status': 'approved'}", count = true)
    long countApproved();

    @Query(value = "{'qualityControl.status': 'pending'}", count = true)
    long countPending();

    @Query(value = "{'qualityControl.status': 'rejected'}", count = true)
    long countRejected();

    @Query(value = "{'location.region': ?0}", count = true)
    long countByRegion(String region);

    @Query(value = "{'dataSource': ?0}", count = true)
    long countByDataSource(String dataSource);
}
