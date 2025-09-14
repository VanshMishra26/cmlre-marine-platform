"""
Taxonomy Classification Service
Handles species identification and classification using morphological and molecular data
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import requests
import json

logger = logging.getLogger(__name__)

class TaxonomyClassificationService:
    """Service for taxonomic classification and species identification"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_extractors = {}
        self.model_path = Path("models/taxonomy")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # External API configurations
        self.gbif_api_url = "https://api.gbif.org/v1"
        self.fishbase_api_url = "https://fishbase.ropensci.org"
        
    async def load_models(self):
        """Load pre-trained models for taxonomy classification"""
        try:
            # Load morphological classification model
            morph_model_path = self.model_path / "morphological_classifier.pkl"
            if morph_model_path.exists():
                self.models['morphological_classifier'] = joblib.load(morph_model_path)
                logger.info("Loaded morphological classification model")
            
            # Load molecular classification model
            mol_model_path = self.model_path / "molecular_classifier.pkl"
            if mol_model_path.exists():
                self.models['molecular_classifier'] = joblib.load(mol_model_path)
                logger.info("Loaded molecular classification model")
            
            # Load integrated classification model
            int_model_path = self.model_path / "integrated_classifier.pkl"
            if int_model_path.exists():
                self.models['integrated_classifier'] = joblib.load(int_model_path)
                logger.info("Loaded integrated classification model")
            
            # Load feature scalers
            scaler_path = self.model_path / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scalers['feature_scaler'] = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            
            # Load label encoders
            label_encoder_path = self.model_path / "label_encoder.pkl"
            if label_encoder_path.exists():
                self.label_encoders['species_encoder'] = joblib.load(label_encoder_path)
                logger.info("Loaded label encoder")
            
            logger.info("Taxonomy classification models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading taxonomy models: {e}")
            # Initialize with default models if loading fails
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models if pre-trained models are not available"""
        self.models['morphological_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['molecular_classifier'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.models['integrated_classifier'] = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        self.scalers['feature_scaler'] = StandardScaler()
        self.label_encoders['species_encoder'] = LabelEncoder()
        logger.info("Initialized default taxonomy models")
    
    async def classify_species(self, request) -> Dict[str, Any]:
        """Classify species using morphological and molecular data"""
        try:
            results = {}
            
            # Morphological classification
            if request.image_path or request.morphological_data:
                morph_result = await self._classify_by_morphology(request)
                results['morphological_classification'] = morph_result
            
            # Molecular classification
            if request.molecular_data:
                mol_result = await self._classify_by_molecular_data(request)
                results['molecular_classification'] = mol_result
            
            # Integrated classification
            if len(results) > 1:
                integrated_result = await self._integrated_classification(results)
                results['integrated_classification'] = integrated_result
            
            # Get final prediction
            final_prediction = self._get_final_prediction(results)
            
            # Get taxonomic hierarchy
            taxonomic_hierarchy = await self._get_taxonomic_hierarchy(final_prediction['species'])
            
            # Get similar species
            similar_species = await self._get_similar_species(final_prediction['species'])
            
            return {
                "success": True,
                "final_prediction": final_prediction,
                "taxonomic_hierarchy": taxonomic_hierarchy,
                "similar_species": similar_species,
                "classification_details": results,
                "confidence": final_prediction.get('confidence', 0.0),
                "processing_time": 0.0  # Will be calculated
            }
            
        except Exception as e:
            logger.error(f"Error in species classification: {e}")
            return {
                "success": False,
                "error": str(e),
                "final_prediction": {"species": "unknown", "confidence": 0.0}
            }
    
    async def _classify_by_morphology(self, request) -> Dict[str, Any]:
        """Classify species using morphological data"""
        try:
            features = {}
            
            # Extract features from image if provided
            if request.image_path:
                image_features = await self._extract_image_features(request.image_path)
                features.update(image_features)
            
            # Extract features from morphological data if provided
            if request.morphological_data:
                morph_features = self._extract_morphological_features(request.morphological_data)
                features.update(morph_features)
            
            # Prepare feature vector
            feature_vector = self._prepare_morphological_feature_vector(features)
            
            # Classify using morphological model
            if 'morphological_classifier' in self.models:
                prediction = self.models['morphological_classifier'].predict([feature_vector])[0]
                probabilities = self.models['morphological_classifier'].predict_proba([feature_vector])[0]
                
                # Get class names
                class_names = self.models['morphological_classifier'].classes_
                
                return {
                    "predicted_species": prediction,
                    "confidence": float(max(probabilities)),
                    "all_probabilities": dict(zip(class_names, probabilities.tolist())),
                    "features_used": list(features.keys()),
                    "method": "morphological"
                }
            else:
                return {
                    "predicted_species": "unknown",
                    "confidence": 0.0,
                    "message": "Morphological classification model not available"
                }
                
        except Exception as e:
            logger.error(f"Error in morphological classification: {e}")
            return {"error": str(e), "predicted_species": "unknown", "confidence": 0.0}
    
    async def _extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """Extract morphological features from image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract shape features
            shape_features = self._extract_shape_features(gray)
            
            # Extract color features
            color_features = self._extract_color_features(image)
            
            # Extract texture features
            texture_features = self._extract_texture_features(gray)
            
            # Extract fin features
            fin_features = self._extract_fin_features(gray)
            
            return {
                **shape_features,
                **color_features,
                **texture_features,
                **fin_features
            }
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return {}
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate basic shape features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate shape descriptors
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            "body_area": float(area),
            "body_perimeter": float(perimeter),
            "body_circularity": float(circularity),
            "body_aspect_ratio": float(aspect_ratio),
            "body_solidity": float(solidity),
            "body_width": float(w),
            "body_height": float(h)
        }
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color-based features"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        bgr_mean = np.mean(image, axis=(0, 1))
        hsv_mean = np.mean(hsv, axis=(0, 1))
        lab_mean = np.mean(lab, axis=(0, 1))
        
        return {
            "bgr_mean_b": float(bgr_mean[0]),
            "bgr_mean_g": float(bgr_mean[1]),
            "bgr_mean_r": float(bgr_mean[2]),
            "hsv_mean_h": float(hsv_mean[0]),
            "hsv_mean_s": float(hsv_mean[1]),
            "hsv_mean_v": float(hsv_mean[2]),
            "lab_mean_l": float(lab_mean[0]),
            "lab_mean_a": float(lab_mean[1]),
            "lab_mean_b": float(lab_mean[2])
        }
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features"""
        # Calculate Local Binary Pattern (simplified)
        lbp = self._calculate_lbp(image)
        
        # Calculate Gabor filter responses
        gabor_responses = self._calculate_gabor_responses(image)
        
        # Calculate Haralick texture features (simplified)
        haralick_features = self._calculate_haralick_features(image)
        
        return {
            **lbp,
            **gabor_responses,
            **haralick_features
        }
    
    def _calculate_lbp(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate Local Binary Pattern features"""
        # Simplified LBP calculation
        rows, cols = image.shape
        lbp_image = np.zeros_like(image)
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = image[i, j]
                binary_string = ""
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if image[i + di, j + dj] >= center:
                            binary_string += "1"
                        else:
                            binary_string += "0"
                
                lbp_image[i, j] = int(binary_string, 2)
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        return {
            "lbp_uniformity": float(np.sum(hist ** 2)),
            "lbp_entropy": float(-np.sum(hist * np.log2(hist + 1e-7))),
            "lbp_mean": float(np.mean(lbp_image)),
            "lbp_std": float(np.std(lbp_image))
        }
    
    def _calculate_gabor_responses(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate Gabor filter responses"""
        # Simplified Gabor filter implementation
        responses = {}
        
        # Different orientations and frequencies
        orientations = [0, 45, 90, 135]
        frequencies = [0.1, 0.2, 0.3]
        
        for i, orientation in enumerate(orientations):
            for j, frequency in enumerate(frequencies):
                # Simplified Gabor kernel
                kernel = self._create_gabor_kernel(frequency, orientation)
                response = cv2.filter2D(image, cv2.CV_64F, kernel)
                
                responses[f"gabor_{orientation}_{frequency}_mean"] = float(np.mean(response))
                responses[f"gabor_{orientation}_{frequency}_std"] = float(np.std(response))
        
        return responses
    
    def _create_gabor_kernel(self, frequency: float, orientation: float) -> np.ndarray:
        """Create Gabor kernel"""
        # Simplified Gabor kernel
        size = 15
        kernel = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                x = i - size // 2
                y = j - size // 2
                
                # Rotate coordinates
                angle_rad = np.radians(orientation)
                x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
                y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
                
                # Gabor function
                kernel[i, j] = np.exp(-(x_rot**2 + y_rot**2) / (2 * 2**2)) * \
                              np.cos(2 * np.pi * frequency * x_rot)
        
        return kernel
    
    def _calculate_haralick_features(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate Haralick texture features (simplified)"""
        # Simplified Haralick features
        return {
            "haralick_contrast": float(np.var(image)),
            "haralick_energy": float(np.sum(image**2) / (image.size**2)),
            "haralick_homogeneity": float(np.sum(1 / (1 + np.abs(image - np.mean(image)))) / image.size),
            "haralick_correlation": float(np.corrcoef(image.flatten(), image.flatten())[0, 1])
        }
    
    def _extract_fin_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract fin-related features"""
        # Simplified fin feature extraction
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Find potential fin contours
        fin_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        fin_features = {
            "fin_count": len(fin_contours),
            "fin_total_area": sum(cv2.contourArea(c) for c in fin_contours),
            "fin_avg_area": np.mean([cv2.contourArea(c) for c in fin_contours]) if fin_contours else 0
        }
        
        return fin_features
    
    def _extract_morphological_features(self, morphological_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from morphological measurements"""
        features = {}
        
        # Extract body measurements
        body_measurements = morphological_data.get('bodyMeasurements', {})
        for key, value in body_measurements.items():
            if isinstance(value, (int, float)):
                features[f"body_{key}"] = float(value)
        
        # Extract fin measurements
        fin_measurements = morphological_data.get('fins', {})
        for fin_type, fin_data in fin_measurements.items():
            if isinstance(fin_data, dict):
                for key, value in fin_data.items():
                    if isinstance(value, (int, float)):
                        features[f"fin_{fin_type}_{key}"] = float(value)
        
        # Extract otolith measurements
        otolith_data = morphological_data.get('otolith', {})
        if otolith_data:
            left_otolith = otolith_data.get('left', {})
            right_otolith = otolith_data.get('right', {})
            
            for side, otolith in [('left', left_otolith), ('right', right_otolith)]:
                if otolith and otolith.get('present'):
                    measurements = otolith.get('measurements', {})
                    for key, value in measurements.items():
                        if isinstance(value, (int, float)):
                            features[f"otolith_{side}_{key}"] = float(value)
        
        return features
    
    def _prepare_morphological_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for morphological classification"""
        # Define expected features
        expected_features = [
            'body_area', 'body_perimeter', 'body_circularity', 'body_aspect_ratio',
            'body_solidity', 'body_width', 'body_height', 'body_length',
            'head_length', 'eye_diameter', 'fin_count', 'fin_total_area'
        ]
        
        feature_vector = []
        for feature in expected_features:
            feature_vector.append(features.get(feature, 0.0))
        
        return np.array(feature_vector)
    
    async def _classify_by_molecular_data(self, request) -> Dict[str, Any]:
        """Classify species using molecular data"""
        try:
            # Extract molecular features
            molecular_features = self._extract_molecular_features(request.molecular_data)
            
            # Prepare feature vector
            feature_vector = self._prepare_molecular_feature_vector(molecular_features)
            
            # Classify using molecular model
            if 'molecular_classifier' in self.models:
                prediction = self.models['molecular_classifier'].predict([feature_vector])[0]
                probabilities = self.models['molecular_classifier'].predict_proba([feature_vector])[0]
                
                # Get class names
                class_names = self.models['molecular_classifier'].classes_
                
                return {
                    "predicted_species": prediction,
                    "confidence": float(max(probabilities)),
                    "all_probabilities": dict(zip(class_names, probabilities.tolist())),
                    "features_used": list(molecular_features.keys()),
                    "method": "molecular"
                }
            else:
                return {
                    "predicted_species": "unknown",
                    "confidence": 0.0,
                    "message": "Molecular classification model not available"
                }
                
        except Exception as e:
            logger.error(f"Error in molecular classification: {e}")
            return {"error": str(e), "predicted_species": "unknown", "confidence": 0.0}
    
    def _extract_molecular_features(self, molecular_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from molecular data"""
        features = {}
        
        # Extract sequence features
        sequence_data = molecular_data.get('sequence', {})
        if sequence_data:
            sequence = sequence_data.get('consensus', '')
            if sequence:
                features.update(self._calculate_sequence_features(sequence))
        
        # Extract barcode features
        barcode_data = molecular_data.get('barcode', {})
        if barcode_data:
            barcode_sequence = barcode_data.get('barcodeSequence', '')
            if barcode_sequence:
                features.update(self._calculate_barcode_features(barcode_sequence))
        
        # Extract genetic diversity features
        genetic_diversity = molecular_data.get('geneticDiversity', {})
        for key, value in genetic_diversity.items():
            if isinstance(value, (int, float)):
                features[f"genetic_{key}"] = float(value)
        
        return features
    
    def _calculate_sequence_features(self, sequence: str) -> Dict[str, float]:
        """Calculate sequence-based features"""
        if not sequence:
            return {}
        
        sequence = sequence.upper()
        
        # Basic composition
        total_length = len(sequence)
        a_count = sequence.count('A')
        t_count = sequence.count('T')
        g_count = sequence.count('G')
        c_count = sequence.count('C')
        n_count = sequence.count('N')
        
        # Calculate frequencies
        a_freq = a_count / total_length if total_length > 0 else 0
        t_freq = t_count / total_length if total_length > 0 else 0
        g_freq = g_count / total_length if total_length > 0 else 0
        c_freq = c_count / total_length if total_length > 0 else 0
        n_freq = n_count / total_length if total_length > 0 else 0
        
        # Calculate GC content
        gc_content = (g_count + c_count) / total_length if total_length > 0 else 0
        
        # Calculate AT content
        at_content = (a_count + t_count) / total_length if total_length > 0 else 0
        
        return {
            "sequence_length": float(total_length),
            "a_frequency": float(a_freq),
            "t_frequency": float(t_freq),
            "g_frequency": float(g_freq),
            "c_frequency": float(c_freq),
            "n_frequency": float(n_freq),
            "gc_content": float(gc_content),
            "at_content": float(at_content)
        }
    
    def _calculate_barcode_features(self, barcode_sequence: str) -> Dict[str, float]:
        """Calculate barcode-specific features"""
        features = self._calculate_sequence_features(barcode_sequence)
        
        # Add barcode-specific features
        features["barcode_length"] = features.get("sequence_length", 0)
        features["barcode_gc_content"] = features.get("gc_content", 0)
        
        return features
    
    def _prepare_molecular_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for molecular classification"""
        # Define expected features
        expected_features = [
            'sequence_length', 'a_frequency', 't_frequency', 'g_frequency',
            'c_frequency', 'gc_content', 'at_content', 'barcode_length',
            'barcode_gc_content'
        ]
        
        feature_vector = []
        for feature in expected_features:
            feature_vector.append(features.get(feature, 0.0))
        
        return np.array(feature_vector)
    
    async def _integrated_classification(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integrated classification using multiple data sources"""
        try:
            # Combine features from different sources
            combined_features = []
            
            # Add morphological features
            if 'morphological_classification' in results:
                morph_result = results['morphological_classification']
                if 'confidence' in morph_result:
                    combined_features.append(morph_result['confidence'])
                else:
                    combined_features.append(0.0)
            
            # Add molecular features
            if 'molecular_classification' in results:
                mol_result = results['molecular_classification']
                if 'confidence' in mol_result:
                    combined_features.append(mol_result['confidence'])
                else:
                    combined_features.append(0.0)
            
            # Use integrated model if available
            if 'integrated_classifier' in self.models and len(combined_features) > 0:
                prediction = self.models['integrated_classifier'].predict([combined_features])[0]
                probabilities = self.models['integrated_classifier'].predict_proba([combined_features])[0]
                
                return {
                    "predicted_species": prediction,
                    "confidence": float(max(probabilities)),
                    "method": "integrated"
                }
            else:
                # Fallback to weighted average
                return self._weighted_average_classification(results)
                
        except Exception as e:
            logger.error(f"Error in integrated classification: {e}")
            return {"error": str(e), "predicted_species": "unknown", "confidence": 0.0}
    
    def _weighted_average_classification(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform weighted average classification"""
        predictions = []
        confidences = []
        
        for result in results.values():
            if isinstance(result, dict) and 'predicted_species' in result:
                predictions.append(result['predicted_species'])
                confidences.append(result.get('confidence', 0.0))
        
        if not predictions:
            return {"predicted_species": "unknown", "confidence": 0.0}
        
        # Use the prediction with highest confidence
        max_confidence_idx = confidences.index(max(confidences))
        best_prediction = predictions[max_confidence_idx]
        best_confidence = confidences[max_confidence_idx]
        
        return {
            "predicted_species": best_prediction,
            "confidence": best_confidence,
            "method": "weighted_average"
        }
    
    def _get_final_prediction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get the final prediction from all classification results"""
        if 'integrated_classification' in results:
            return results['integrated_classification']
        elif 'morphological_classification' in results:
            return results['morphological_classification']
        elif 'molecular_classification' in results:
            return results['molecular_classification']
        else:
            return {"species": "unknown", "confidence": 0.0}
    
    async def _get_taxonomic_hierarchy(self, species_name: str) -> Dict[str, str]:
        """Get taxonomic hierarchy for a species"""
        try:
            # Query GBIF API for taxonomic information
            response = requests.get(f"{self.gbif_api_url}/species/search", 
                                 params={"q": species_name, "limit": 1})
            
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    species_data = data['results'][0]
                    return {
                        "kingdom": species_data.get('kingdom', ''),
                        "phylum": species_data.get('phylum', ''),
                        "class": species_data.get('class', ''),
                        "order": species_data.get('order', ''),
                        "family": species_data.get('family', ''),
                        "genus": species_data.get('genus', ''),
                        "species": species_data.get('scientificName', '')
                    }
            
            return {"species": species_name}
            
        except Exception as e:
            logger.error(f"Error getting taxonomic hierarchy: {e}")
            return {"species": species_name}
    
    async def _get_similar_species(self, species_name: str) -> List[Dict[str, Any]]:
        """Get similar species based on taxonomic classification"""
        try:
            # Query GBIF API for similar species
            response = requests.get(f"{self.gbif_api_url}/species/search", 
                                 params={"q": species_name, "limit": 10})
            
            if response.status_code == 200:
                data = response.json()
                similar_species = []
                
                for species_data in data['results'][:5]:  # Top 5 similar species
                    similar_species.append({
                        "scientific_name": species_data.get('scientificName', ''),
                        "common_name": species_data.get('vernacularName', ''),
                        "family": species_data.get('family', ''),
                        "genus": species_data.get('genus', ''),
                        "confidence": 0.8  # Placeholder confidence
                    })
                
                return similar_species
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting similar species: {e}")
            return []
    
    async def identify_species(self, request) -> Dict[str, Any]:
        """Identify species from images or morphological data"""
        # This is essentially the same as classify_species but with different naming
        return await self.classify_species(request)
