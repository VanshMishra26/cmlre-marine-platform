"""
Otolith Analysis Service
Handles otolith shape analysis, morphometrics, and species classification
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class OtolithAnalysisService:
    """Service for otolith shape analysis and species classification"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        self.model_path = Path("models/otolith")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    async def load_models(self):
        """Load pre-trained models for otolith analysis"""
        try:
            # Load shape classification model
            shape_model_path = self.model_path / "shape_classifier.pkl"
            if shape_model_path.exists():
                self.models['shape_classifier'] = joblib.load(shape_model_path)
                logger.info("Loaded shape classification model")
            
            # Load species classification model
            species_model_path = self.model_path / "species_classifier.pkl"
            if species_model_path.exists():
                self.models['species_classifier'] = joblib.load(species_model_path)
                logger.info("Loaded species classification model")
            
            # Load feature scaler
            scaler_path = self.model_path / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scalers['feature_scaler'] = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            
            logger.info("Otolith analysis models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading otolith models: {e}")
            # Initialize with default models if loading fails
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models if pre-trained models are not available"""
        self.models['shape_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['species_classifier'] = SVC(kernel='rbf', probability=True, random_state=42)
        self.scalers['feature_scaler'] = StandardScaler()
        logger.info("Initialized default models")
    
    async def analyze_otolith(self, request) -> Dict[str, Any]:
        """Analyze otolith shape and extract morphometric features"""
        try:
            # Load and preprocess image
            image = self._load_image(request.image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Extract shape features
            shape_features = self._extract_shape_features(image)
            
            # Extract geometric morphometric features
            geometric_features = self._extract_geometric_features(image)
            
            # Extract traditional morphometric features
            traditional_features = self._extract_traditional_features(image)
            
            # Combine all features
            all_features = {
                **shape_features,
                **geometric_features,
                **traditional_features
            }
            
            # Perform shape analysis
            shape_analysis = self._analyze_shape(image, all_features)
            
            # Generate analysis report
            analysis_result = {
                "shape_features": shape_features,
                "geometric_features": geometric_features,
                "traditional_features": traditional_features,
                "shape_analysis": shape_analysis,
                "quality_metrics": self._calculate_quality_metrics(image),
                "recommendations": self._generate_recommendations(all_features)
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in otolith analysis: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess otolith image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            processed = self._preprocess_image(gray)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features from otolith image"""
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
        
        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        
        # Extent
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "aspect_ratio": float(aspect_ratio),
            "solidity": float(solidity),
            "equivalent_diameter": float(equivalent_diameter),
            "extent": float(extent),
            "width": float(w),
            "height": float(h)
        }
    
    def _extract_geometric_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract geometric morphometric features"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        contour = max(contours, key=cv2.contourArea)
        
        # Landmark points (simplified - in practice, you'd use more sophisticated methods)
        landmarks = self._extract_landmarks(contour)
        
        # Procrustes analysis (simplified)
        procrustes_coords = self._procrustes_analysis(landmarks)
        
        # Principal component analysis
        pca_features = self._pca_analysis(procrustes_coords)
        
        return {
            "landmarks": landmarks.tolist(),
            "procrustes_coordinates": procrustes_coords.tolist(),
            "pca_features": pca_features.tolist(),
            "landmark_count": len(landmarks)
        }
    
    def _extract_landmarks(self, contour: np.ndarray) -> np.ndarray:
        """Extract landmark points from contour"""
        # Simplified landmark extraction
        # In practice, you'd use more sophisticated methods like Procrustes landmarks
        
        # Get contour points
        points = contour.reshape(-1, 2)
        
        # Sample points evenly
        n_landmarks = 20
        if len(points) < n_landmarks:
            return points
        
        indices = np.linspace(0, len(points) - 1, n_landmarks, dtype=int)
        landmarks = points[indices]
        
        return landmarks
    
    def _procrustes_analysis(self, landmarks: np.ndarray) -> np.ndarray:
        """Perform Procrustes analysis on landmarks"""
        # Simplified Procrustes analysis
        # Center the landmarks
        centered = landmarks - np.mean(landmarks, axis=0)
        
        # Scale to unit size
        scale = np.sqrt(np.sum(centered**2))
        if scale > 0:
            scaled = centered / scale
        else:
            scaled = centered
        
        return scaled
    
    def _pca_analysis(self, coords: np.ndarray) -> np.ndarray:
        """Perform PCA on landmark coordinates"""
        from sklearn.decomposition import PCA
        
        # Flatten coordinates
        flat_coords = coords.flatten()
        
        # Reshape for PCA (assuming we have multiple samples)
        if len(flat_coords.shape) == 1:
            flat_coords = flat_coords.reshape(1, -1)
        
        # Perform PCA
        pca = PCA(n_components=min(10, flat_coords.shape[1]))
        pca_result = pca.fit_transform(flat_coords)
        
        return pca_result[0]  # Return first sample
    
    def _extract_traditional_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract traditional morphometric features"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate moments
        moments = cv2.moments(contour)
        
        # Hu moments
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Fourier descriptors
        fourier_descriptors = self._calculate_fourier_descriptors(contour)
        
        return {
            "hu_moments": hu_moments.tolist(),
            "fourier_descriptors": fourier_descriptors.tolist(),
            "m00": float(moments['m00']),
            "m10": float(moments['m10']),
            "m01": float(moments['m01']),
            "m20": float(moments['m20']),
            "m11": float(moments['m11']),
            "m02": float(moments['m02'])
        }
    
    def _calculate_fourier_descriptors(self, contour: np.ndarray) -> np.ndarray:
        """Calculate Fourier descriptors for shape analysis"""
        # Convert contour to complex numbers
        complex_contour = contour[:, 0, 0] + 1j * contour[:, 0, 1]
        
        # Apply FFT
        fft = np.fft.fft(complex_contour)
        
        # Take first 10 descriptors (excluding DC component)
        descriptors = np.abs(fft[1:11])
        
        # Normalize
        if descriptors[0] > 0:
            descriptors = descriptors / descriptors[0]
        
        return descriptors
    
    def _analyze_shape(self, image: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze otolith shape characteristics"""
        # Shape classification based on features
        shape_type = self._classify_shape_type(features)
        
        # Asymmetry analysis
        asymmetry = self._calculate_asymmetry(image)
        
        # Growth pattern analysis
        growth_pattern = self._analyze_growth_pattern(image)
        
        return {
            "shape_type": shape_type,
            "asymmetry": asymmetry,
            "growth_pattern": growth_pattern,
            "complexity": self._calculate_complexity(features),
            "regularity": self._calculate_regularity(features)
        }
    
    def _classify_shape_type(self, features: Dict[str, Any]) -> str:
        """Classify otolith shape type"""
        circularity = features.get('circularity', 0)
        aspect_ratio = features.get('aspect_ratio', 1)
        
        if circularity > 0.8:
            return "circular"
        elif aspect_ratio > 2.0:
            return "elongated"
        elif aspect_ratio < 0.5:
            return "compressed"
        else:
            return "oval"
    
    def _calculate_asymmetry(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate asymmetry metrics"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated methods
        
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"bilateral": 0.0, "radial": 0.0}
        
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return {"bilateral": 0.0, "radial": 0.0}
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Calculate asymmetry (simplified)
        bilateral_asymmetry = 0.0  # Placeholder
        radial_asymmetry = 0.0     # Placeholder
        
        return {
            "bilateral": bilateral_asymmetry,
            "radial": radial_asymmetry
        }
    
    def _analyze_growth_pattern(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze growth pattern in otolith"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated methods like ring detection
        
        return {
            "growth_rings": 0,  # Placeholder
            "growth_rate": 0.0,  # Placeholder
            "age_estimate": 0    # Placeholder
        }
    
    def _calculate_complexity(self, features: Dict[str, Any]) -> float:
        """Calculate shape complexity"""
        perimeter = features.get('perimeter', 0)
        area = features.get('area', 1)
        
        if area > 0:
            complexity = (perimeter * perimeter) / (4 * np.pi * area)
        else:
            complexity = 0
        
        return float(complexity)
    
    def _calculate_regularity(self, features: Dict[str, Any]) -> float:
        """Calculate shape regularity"""
        circularity = features.get('circularity', 0)
        solidity = features.get('solidity', 0)
        
        # Combine circularity and solidity for regularity measure
        regularity = (circularity + solidity) / 2
        
        return float(regularity)
    
    def _calculate_quality_metrics(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate image quality metrics"""
        # Calculate image quality metrics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Calculate contrast
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        # Calculate sharpness (simplified)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        return {
            "mean_intensity": float(mean_intensity),
            "std_intensity": float(std_intensity),
            "contrast": float(contrast),
            "sharpness": float(laplacian_var),
            "quality_score": float(min(contrast * 10, 1.0))  # Normalized quality score
        }
    
    def _generate_recommendations(self, features: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check image quality
        if features.get('quality_score', 0) < 0.5:
            recommendations.append("Image quality is low. Consider retaking the image with better lighting and focus.")
        
        # Check shape characteristics
        circularity = features.get('circularity', 0)
        if circularity < 0.3:
            recommendations.append("Otolith shape is highly irregular. Verify specimen integrity.")
        
        # Check size
        area = features.get('area', 0)
        if area < 100:
            recommendations.append("Otolith is very small. Consider using higher magnification.")
        
        return recommendations
    
    async def classify_species(self, request) -> Dict[str, Any]:
        """Classify species based on otolith morphology"""
        try:
            # Extract features
            image = self._load_image(request.image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            features = self._extract_shape_features(image)
            geometric_features = self._extract_geometric_features(image)
            traditional_features = self._extract_traditional_features(image)
            
            # Combine features
            all_features = {
                **features,
                **geometric_features,
                **traditional_features
            }
            
            # Prepare feature vector for classification
            feature_vector = self._prepare_feature_vector(all_features)
            
            # Classify species
            if 'species_classifier' in self.models:
                prediction = self.models['species_classifier'].predict([feature_vector])[0]
                probabilities = self.models['species_classifier'].predict_proba([feature_vector])[0]
                
                # Get class names
                class_names = self.models['species_classifier'].classes_
                
                # Create result
                result = {
                    "predicted_species": prediction,
                    "confidence": float(max(probabilities)),
                    "all_probabilities": dict(zip(class_names, probabilities.tolist())),
                    "features_used": list(all_features.keys())
                }
            else:
                result = {
                    "predicted_species": "unknown",
                    "confidence": 0.0,
                    "message": "Species classification model not available"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in species classification: {e}")
            raise
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for machine learning models"""
        # Extract numerical features
        numerical_features = []
        
        # Add basic shape features
        for key in ['area', 'perimeter', 'circularity', 'aspect_ratio', 'solidity', 
                   'equivalent_diameter', 'extent', 'width', 'height']:
            if key in features:
                numerical_features.append(features[key])
        
        # Add Hu moments
        if 'hu_moments' in features:
            numerical_features.extend(features['hu_moments'])
        
        # Add Fourier descriptors
        if 'fourier_descriptors' in features:
            numerical_features.extend(features['fourier_descriptors'])
        
        # Add PCA features
        if 'pca_features' in features:
            numerical_features.extend(features['pca_features'])
        
        return np.array(numerical_features)
