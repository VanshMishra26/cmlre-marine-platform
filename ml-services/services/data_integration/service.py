"""
Data Integration Service
Handles cross-disciplinary correlation analysis and integrated visualization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE, UMAP
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataIntegrationService:
    """Service for cross-disciplinary data integration and correlation analysis"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        self.model_path = Path("models/data_integration")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Analysis parameters
        self.correlation_methods = ['pearson', 'spearman', 'kendall']
        self.clustering_methods = ['kmeans', 'dbscan', 'hierarchical']
        self.dimensionality_reduction_methods = ['pca', 'tsne', 'umap', 'factor_analysis']
        
    async def load_models(self):
        """Load pre-trained models for data integration"""
        try:
            # Load correlation analysis models
            correlation_model_path = self.model_path / "correlation_analyzer.pkl"
            if correlation_model_path.exists():
                self.models['correlation_analyzer'] = joblib.load(correlation_model_path)
                logger.info("Loaded correlation analysis model")
            
            # Load clustering models
            clustering_model_path = self.model_path / "clustering_analyzer.pkl"
            if clustering_model_path.exists():
                self.models['clustering_analyzer'] = joblib.load(clustering_model_path)
                logger.info("Loaded clustering analysis model")
            
            # Load predictive models
            predictive_model_path = self.model_path / "predictive_analyzer.pkl"
            if predictive_model_path.exists():
                self.models['predictive_analyzer'] = joblib.load(predictive_model_path)
                logger.info("Loaded predictive analysis model")
            
            # Load feature scalers
            scaler_path = self.model_path / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scalers['feature_scaler'] = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            
            logger.info("Data integration models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data integration models: {e}")
            # Initialize with default models if loading fails
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models if pre-trained models are not available"""
        self.models['correlation_analyzer'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['clustering_analyzer'] = KMeans(n_clusters=5, random_state=42)
        self.models['predictive_analyzer'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scalers['feature_scaler'] = StandardScaler()
        logger.info("Initialized default data integration models")
    
    async def correlate_data(self, request) -> Dict[str, Any]:
        """Perform cross-disciplinary correlation analysis"""
        try:
            # Extract data from request
            oceanographic_data = request.oceanographic_data or []
            taxonomic_data = request.taxonomic_data or []
            morphological_data = request.morphological_data or []
            molecular_data = request.molecular_data or []
            
            # Prepare integrated dataset
            integrated_data = self._prepare_integrated_dataset(
                oceanographic_data, taxonomic_data, morphological_data, molecular_data)
            
            if integrated_data.empty:
                return {
                    "success": False,
                    "error": "No data available for correlation analysis"
                }
            
            # Perform correlation analysis
            correlation_results = self._perform_correlation_analysis(
                integrated_data, request.correlation_methods)
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(integrated_data)
            
            # Perform clustering analysis
            clustering_results = self._perform_clustering_analysis(integrated_data)
            
            # Perform dimensionality reduction
            dimensionality_results = self._perform_dimensionality_reduction(integrated_data)
            
            # Generate insights
            insights = self._generate_insights(correlation_results, statistical_tests, clustering_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(correlation_results, insights)
            
            return {
                "success": True,
                "correlation_analysis": correlation_results,
                "statistical_tests": statistical_tests,
                "clustering_analysis": clustering_results,
                "dimensionality_reduction": dimensionality_results,
                "insights": insights,
                "recommendations": recommendations,
                "data_summary": self._generate_data_summary(integrated_data)
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _prepare_integrated_dataset(self, oceanographic_data: List[Dict], taxonomic_data: List[Dict],
                                  morphological_data: List[Dict], molecular_data: List[Dict]) -> pd.DataFrame:
        """Prepare integrated dataset from multiple data sources"""
        try:
            # Convert data to DataFrames
            oceanographic_df = self._convert_oceanographic_data(oceanographic_data)
            taxonomic_df = self._convert_taxonomic_data(taxonomic_data)
            morphological_df = self._convert_morphological_data(morphological_data)
            molecular_df = self._convert_molecular_data(molecular_data)
            
            # Merge datasets based on common identifiers
            integrated_df = self._merge_datasets(oceanographic_df, taxonomic_df, morphological_df, molecular_df)
            
            # Clean and preprocess data
            integrated_df = self._clean_and_preprocess_data(integrated_df)
            
            return integrated_df
            
        except Exception as e:
            logger.error(f"Error preparing integrated dataset: {e}")
            return pd.DataFrame()
    
    def _convert_oceanographic_data(self, data: List[Dict]) -> pd.DataFrame:
        """Convert oceanographic data to DataFrame"""
        if not data:
            return pd.DataFrame()
        
        records = []
        for record in data:
            # Extract key oceanographic parameters
            oceanographic_record = {
                'station_id': record.get('stationId', ''),
                'cruise_id': record.get('cruiseId', ''),
                'latitude': record.get('location', {}).get('latitude', np.nan),
                'longitude': record.get('location', {}).get('longitude', np.nan),
                'depth': record.get('location', {}).get('depth', np.nan),
                'region': record.get('location', {}).get('region', ''),
                'collection_date': record.get('collectionDate', ''),
                'surface_temperature': record.get('physical', {}).get('temperature', {}).get('surface', np.nan),
                'bottom_temperature': record.get('physical', {}).get('temperature', {}).get('bottom', np.nan),
                'surface_salinity': record.get('physical', {}).get('salinity', {}).get('surface', np.nan),
                'bottom_salinity': record.get('physical', {}).get('salinity', {}).get('bottom', np.nan),
                'surface_oxygen': record.get('chemical', {}).get('dissolvedOxygen', {}).get('surface', np.nan),
                'bottom_oxygen': record.get('chemical', {}).get('dissolvedOxygen', {}).get('bottom', np.nan),
                'primary_production': record.get('biological', {}).get('primaryProduction', np.nan),
                'species_richness': record.get('biological', {}).get('speciesRichness', np.nan),
                'data_source': 'oceanographic'
            }
            records.append(oceanographic_record)
        
        return pd.DataFrame(records)
    
    def _convert_taxonomic_data(self, data: List[Dict]) -> pd.DataFrame:
        """Convert taxonomic data to DataFrame"""
        if not data:
            return pd.DataFrame()
        
        records = []
        for record in data:
            taxonomic_record = {
                'specimen_id': record.get('specimenId', ''),
                'species_name': record.get('species', {}).get('scientificName', ''),
                'common_name': record.get('species', {}).get('commonName', ''),
                'kingdom': record.get('taxonomy', {}).get('kingdom', ''),
                'phylum': record.get('taxonomy', {}).get('phylum', ''),
                'class': record.get('taxonomy', {}).get('class', ''),
                'order': record.get('taxonomy', {}).get('order', ''),
                'family': record.get('taxonomy', {}).get('family', ''),
                'genus': record.get('taxonomy', {}).get('genus', ''),
                'species': record.get('taxonomy', {}).get('species', ''),
                'latitude': record.get('collection', {}).get('location', {}).get('latitude', np.nan),
                'longitude': record.get('collection', {}).get('location', {}).get('longitude', np.nan),
                'depth': record.get('collection', {}).get('location', {}).get('depth', np.nan),
                'region': record.get('collection', {}).get('location', {}).get('region', ''),
                'collection_date': record.get('collection', {}).get('collectionDate', ''),
                'abundance': record.get('ecology', {}).get('abundance', np.nan),
                'biomass': record.get('ecology', {}).get('biomass', np.nan),
                'conservation_status': record.get('ecology', {}).get('conservationStatus', ''),
                'data_source': 'taxonomic'
            }
            records.append(taxonomic_record)
        
        return pd.DataFrame(records)
    
    def _convert_morphological_data(self, data: List[Dict]) -> pd.DataFrame:
        """Convert morphological data to DataFrame"""
        if not data:
            return pd.DataFrame()
        
        records = []
        for record in data:
            morphological_record = {
                'specimen_id': record.get('specimenId', ''),
                'species_id': record.get('speciesId', ''),
                'total_length': record.get('bodyMeasurements', {}).get('totalLength', np.nan),
                'standard_length': record.get('bodyMeasurements', {}).get('standardLength', np.nan),
                'head_length': record.get('bodyMeasurements', {}).get('headLength', np.nan),
                'body_depth': record.get('bodyMeasurements', {}).get('bodyDepth', np.nan),
                'eye_diameter': record.get('bodyMeasurements', {}).get('eyeDiameter', np.nan),
                'otolith_length': record.get('otolith', {}).get('left', {}).get('length', np.nan),
                'otolith_width': record.get('otolith', {}).get('left', {}).get('width', np.nan),
                'otolith_area': record.get('otolith', {}).get('left', {}).get('measurements', {}).get('area', np.nan),
                'otolith_circularity': record.get('otolith', {}).get('left', {}).get('measurements', {}).get('circularity', np.nan),
                'collection_date': record.get('collection', {}).get('date', ''),
                'data_source': 'morphological'
            }
            records.append(morphological_record)
        
        return pd.DataFrame(records)
    
    def _convert_molecular_data(self, data: List[Dict]) -> pd.DataFrame:
        """Convert molecular data to DataFrame"""
        if not data:
            return pd.DataFrame()
        
        records = []
        for record in data:
            molecular_record = {
                'sample_id': record.get('sampleId', ''),
                'species_id': record.get('speciesId', ''),
                'sequence_length': record.get('sequence', {}).get('length', np.nan),
                'gc_content': record.get('sequence', {}).get('gcContent', np.nan),
                'sequence_quality': record.get('sequence', {}).get('quality', ''),
                'target_gene': record.get('pcr', {}).get('targetGene', ''),
                'is_edna': record.get('edna', {}).get('isEdna', False),
                'abundance': record.get('edna', {}).get('abundance', np.nan),
                'detection_confidence': record.get('edna', {}).get('detectionConfidence', ''),
                'collection_date': record.get('sample', {}).get('collectionDate', ''),
                'data_source': 'molecular'
            }
            records.append(molecular_record)
        
        return pd.DataFrame(records)
    
    def _merge_datasets(self, oceanographic_df: pd.DataFrame, taxonomic_df: pd.DataFrame,
                       morphological_df: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Merge datasets based on common identifiers"""
        try:
            # Start with oceanographic data as base
            merged_df = oceanographic_df.copy()
            
            # Merge taxonomic data based on location and date
            if not taxonomic_df.empty:
                merged_df = self._merge_by_location_and_date(merged_df, taxonomic_df, 'taxonomic')
            
            # Merge morphological data based on species
            if not morphological_df.empty:
                merged_df = self._merge_by_species(merged_df, morphological_df, 'morphological')
            
            # Merge molecular data based on species
            if not molecular_df.empty:
                merged_df = self._merge_by_species(merged_df, molecular_df, 'molecular')
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            return pd.DataFrame()
    
    def _merge_by_location_and_date(self, base_df: pd.DataFrame, merge_df: pd.DataFrame, 
                                   data_type: str) -> pd.DataFrame:
        """Merge datasets based on location and date"""
        try:
            # Convert collection_date to datetime
            base_df['collection_date'] = pd.to_datetime(base_df['collection_date'], errors='coerce')
            merge_df['collection_date'] = pd.to_datetime(merge_df['collection_date'], errors='coerce')
            
            # Merge on location and date (with tolerance)
            merged_df = base_df.merge(
                merge_df, 
                on=['latitude', 'longitude', 'region'], 
                how='left',
                suffixes=('', f'_{data_type}')
            )
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging by location and date: {e}")
            return base_df
    
    def _merge_by_species(self, base_df: pd.DataFrame, merge_df: pd.DataFrame, 
                         data_type: str) -> pd.DataFrame:
        """Merge datasets based on species"""
        try:
            # This is a simplified merge - in practice, you'd need more sophisticated matching
            merged_df = base_df.merge(
                merge_df,
                left_on='species_name',
                right_on='species_name',
                how='left',
                suffixes=('', f'_{data_type}')
            )
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging by species: {e}")
            return base_df
    
    def _clean_and_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess integrated dataset"""
        try:
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            
            # Remove outliers using IQR method
            for column in numeric_columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning and preprocessing data: {e}")
            return df
    
    def _perform_correlation_analysis(self, df: pd.DataFrame, methods: List[str]) -> Dict[str, Any]:
        """Perform correlation analysis"""
        try:
            # Select numeric columns for correlation
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            correlation_data = df[numeric_columns]
            
            correlation_results = {}
            
            for method in methods:
                if method == 'pearson':
                    corr_matrix = correlation_data.corr(method='pearson')
                elif method == 'spearman':
                    corr_matrix = correlation_data.corr(method='spearman')
                elif method == 'kendall':
                    corr_matrix = correlation_data.corr(method='kendall')
                else:
                    continue
                
                # Find strong correlations
                strong_correlations = self._find_strong_correlations(corr_matrix, threshold=0.7)
                
                correlation_results[method] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'strong_correlations': strong_correlations,
                    'summary_statistics': self._calculate_correlation_summary(corr_matrix)
                }
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations above threshold"""
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) >= 0.8 else 'moderate'
                    })
        
        return strong_correlations
    
    def _calculate_correlation_summary(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation summary statistics"""
        # Remove diagonal values
        corr_values = corr_matrix.values
        np.fill_diagonal(corr_values, np.nan)
        corr_values = corr_values[~np.isnan(corr_values)]
        
        return {
            'mean_correlation': float(np.mean(corr_values)),
            'std_correlation': float(np.std(corr_values)),
            'max_correlation': float(np.max(corr_values)),
            'min_correlation': float(np.min(corr_values)),
            'strong_correlations_count': int(np.sum(np.abs(corr_values) >= 0.7)),
            'moderate_correlations_count': int(np.sum((np.abs(corr_values) >= 0.5) & (np.abs(corr_values) < 0.7)))
        }
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            test_results = {}
            
            # Perform t-tests between different data sources
            data_sources = df['data_source'].unique()
            if len(data_sources) > 1:
                for i, source1 in enumerate(data_sources):
                    for source2 in data_sources[i+1:]:
                        source1_data = df[df['data_source'] == source1][numeric_columns]
                        source2_data = df[df['data_source'] == source2][numeric_columns]
                        
                        for column in numeric_columns:
                            if column in source1_data.columns and column in source2_data.columns:
                                try:
                                    t_stat, p_value = stats.ttest_ind(
                                        source1_data[column].dropna(),
                                        source2_data[column].dropna()
                                    )
                                    test_results[f"{source1}_vs_{source2}_{column}"] = {
                                        't_statistic': float(t_stat),
                                        'p_value': float(p_value),
                                        'significant': p_value < 0.05
                                    }
                                except:
                                    continue
            
            # Perform ANOVA tests
            anova_results = {}
            for column in numeric_columns:
                if column in df.columns:
                    try:
                        groups = [group[column].dropna() for name, group in df.groupby('data_source')]
                        f_stat, p_value = stats.f_oneway(*groups)
                        anova_results[column] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                    except:
                        continue
            
            return {
                't_tests': test_results,
                'anova_tests': anova_results
            }
            
        except Exception as e:
            logger.error(f"Error in statistical tests: {e}")
            return {}
    
    def _perform_clustering_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            clustering_data = df[numeric_columns].dropna()
            
            if clustering_data.empty:
                return {}
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clustering_data)
            
            clustering_results = {}
            
            # K-means clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans_labels = kmeans.fit_predict(scaled_data)
            
            clustering_results['kmeans'] = {
                'labels': kmeans_labels.tolist(),
                'n_clusters': 5,
                'inertia': float(kmeans.inertia_),
                'silhouette_score': float(self._calculate_silhouette_score(scaled_data, kmeans_labels))
            }
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(scaled_data)
            
            clustering_results['dbscan'] = {
                'labels': dbscan_labels.tolist(),
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'n_noise': int(list(dbscan_labels).count(-1))
            }
            
            # Hierarchical clustering
            hierarchical = AgglomerativeClustering(n_clusters=5)
            hierarchical_labels = hierarchical.fit_predict(scaled_data)
            
            clustering_results['hierarchical'] = {
                'labels': hierarchical_labels.tolist(),
                'n_clusters': 5
            }
            
            return clustering_results
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            return {}
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(data, labels)
        except:
            return 0.0
    
    def _perform_dimensionality_reduction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform dimensionality reduction"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            reduction_data = df[numeric_columns].dropna()
            
            if reduction_data.empty:
                return {}
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(reduction_data)
            
            reduction_results = {}
            
            # PCA
            pca = PCA(n_components=min(10, scaled_data.shape[1]))
            pca_result = pca.fit_transform(scaled_data)
            
            reduction_results['pca'] = {
                'components': pca_result.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist()
            }
            
            # t-SNE
            if scaled_data.shape[0] > 50:  # t-SNE requires sufficient data
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(scaled_data)
                
                reduction_results['tsne'] = {
                    'components': tsne_result.tolist()
                }
            
            # UMAP
            try:
                import umap
                umap_reducer = umap.UMAP(n_components=2, random_state=42)
                umap_result = umap_reducer.fit_transform(scaled_data)
                
                reduction_results['umap'] = {
                    'components': umap_result.tolist()
                }
            except ImportError:
                pass
            
            return reduction_results
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            return {}
    
    def _generate_insights(self, correlation_results: Dict, statistical_tests: Dict, 
                          clustering_results: Dict) -> List[str]:
        """Generate insights from analysis results"""
        insights = []
        
        # Correlation insights
        for method, results in correlation_results.items():
            strong_correlations = results.get('strong_correlations', [])
            if strong_correlations:
                insights.append(f"Found {len(strong_correlations)} strong correlations using {method} method")
                
                # Find strongest correlation
                strongest = max(strong_correlations, key=lambda x: abs(x['correlation']))
                insights.append(f"Strongest correlation: {strongest['variable1']} vs {strongest['variable2']} ({strongest['correlation']:.3f})")
        
        # Statistical test insights
        if 'anova_tests' in statistical_tests:
            significant_tests = [test for test in statistical_tests['anova_tests'].values() if test['significant']]
            if significant_tests:
                insights.append(f"Found {len(significant_tests)} significant differences between data sources")
        
        # Clustering insights
        if 'kmeans' in clustering_results:
            n_clusters = clustering_results['kmeans']['n_clusters']
            silhouette_score = clustering_results['kmeans']['silhouette_score']
            insights.append(f"K-means clustering identified {n_clusters} distinct groups (silhouette score: {silhouette_score:.3f})")
        
        return insights
    
    def _generate_recommendations(self, correlation_results: Dict, insights: List[str]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Data quality recommendations
        if len(insights) < 3:
            recommendations.append("Consider collecting more data to improve analysis reliability")
        
        # Correlation recommendations
        for method, results in correlation_results.items():
            strong_correlations = results.get('strong_correlations', [])
            if len(strong_correlations) > 10:
                recommendations.append("High number of strong correlations detected - consider focusing on the most significant relationships")
        
        # Clustering recommendations
        if 'kmeans' in correlation_results:
            silhouette_score = correlation_results['kmeans'].get('silhouette_score', 0)
            if silhouette_score < 0.3:
                recommendations.append("Low clustering quality - consider different clustering parameters or additional features")
        
        return recommendations
    
    def _generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of integrated dataset"""
        return {
            'total_records': len(df),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'missing_data_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'data_sources': df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {}
        }
    
    async def create_visualization(self, request) -> Dict[str, Any]:
        """Create integrated visualizations"""
        try:
            # Extract data from request
            oceanographic_data = request.oceanographic_data or []
            taxonomic_data = request.taxonomic_data or []
            morphological_data = request.morphological_data or []
            molecular_data = request.molecular_data or []
            
            # Prepare integrated dataset
            integrated_data = self._prepare_integrated_dataset(
                oceanographic_data, taxonomic_data, morphological_data, molecular_data)
            
            if integrated_data.empty:
                return {
                    "success": False,
                    "error": "No data available for visualization"
                }
            
            # Create visualizations based on type
            visualization_type = request.visualization_type
            visualizations = {}
            
            if visualization_type == 'scatter':
                visualizations = self._create_scatter_plots(integrated_data)
            elif visualization_type == 'heatmap':
                visualizations = self._create_heatmaps(integrated_data)
            elif visualization_type == 'timeseries':
                visualizations = self._create_timeseries_plots(integrated_data)
            elif visualization_type == 'spatial':
                visualizations = self._create_spatial_plots(integrated_data)
            else:
                visualizations = self._create_comprehensive_visualizations(integrated_data)
            
            return {
                "success": True,
                "visualizations": visualizations,
                "data_summary": self._generate_data_summary(integrated_data)
            }
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_scatter_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create scatter plots"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            plots = {}
            
            # Create scatter plots for key variable pairs
            key_variables = ['surface_temperature', 'surface_salinity', 'species_richness', 'abundance']
            available_variables = [col for col in key_variables if col in numeric_columns]
            
            for i, var1 in enumerate(available_variables):
                for var2 in available_variables[i+1:]:
                    plot_data = df[[var1, var2]].dropna()
                    if not plot_data.empty:
                        plots[f"{var1}_vs_{var2}"] = {
                            'type': 'scatter',
                            'data': plot_data.to_dict('records'),
                            'x_axis': var1,
                            'y_axis': var2
                        }
            
            return plots
            
        except Exception as e:
            logger.error(f"Error creating scatter plots: {e}")
            return {}
    
    def _create_heatmaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create heatmaps"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            correlation_data = df[numeric_columns].corr()
            
            return {
                'correlation_heatmap': {
                    'type': 'heatmap',
                    'data': correlation_data.to_dict(),
                    'title': 'Correlation Matrix'
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating heatmaps: {e}")
            return {}
    
    def _create_timeseries_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create time series plots"""
        try:
            if 'collection_date' not in df.columns:
                return {}
            
            # Convert collection_date to datetime
            df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')
            df = df.dropna(subset=['collection_date'])
            
            # Group by date and calculate means
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            time_series_data = df.groupby('collection_date')[numeric_columns].mean()
            
            plots = {}
            for column in numeric_columns:
                if column in time_series_data.columns:
                    plots[f"timeseries_{column}"] = {
                        'type': 'timeseries',
                        'data': time_series_data[column].to_dict(),
                        'x_axis': 'date',
                        'y_axis': column
                    }
            
            return plots
            
        except Exception as e:
            logger.error(f"Error creating time series plots: {e}")
            return {}
    
    def _create_spatial_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create spatial plots"""
        try:
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                return {}
            
            spatial_data = df[['latitude', 'longitude']].dropna()
            
            return {
                'spatial_distribution': {
                    'type': 'scatter_map',
                    'data': spatial_data.to_dict('records'),
                    'x_axis': 'longitude',
                    'y_axis': 'latitude'
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating spatial plots: {e}")
            return {}
    
    def _create_comprehensive_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive visualizations"""
        try:
            visualizations = {}
            
            # Combine all visualization types
            visualizations.update(self._create_scatter_plots(df))
            visualizations.update(self._create_heatmaps(df))
            visualizations.update(self._create_timeseries_plots(df))
            visualizations.update(self._create_spatial_plots(df))
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating comprehensive visualizations: {e}")
            return {}
