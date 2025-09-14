"""
Molecular Analysis Service
Handles DNA sequence analysis, eDNA processing, and molecular species identification
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import requests
import json
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio.Phylo import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import io
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class MolecularAnalysisService:
    """Service for molecular data analysis and species identification"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        self.model_path = Path("models/molecular")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # External database configurations
        self.genbank_api_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.bold_api_url = "http://www.boldsystems.org/index.php/API_Public"
        self.uniprot_api_url = "https://rest.uniprot.org"
        
        # Sequence analysis parameters
        self.sequence_quality_threshold = 0.7
        self.min_sequence_length = 100
        self.max_sequence_length = 10000
        
    async def load_models(self):
        """Load pre-trained models for molecular analysis"""
        try:
            # Load species identification model
            species_model_path = self.model_path / "species_identifier.pkl"
            if species_model_path.exists():
                self.models['species_identifier'] = joblib.load(species_model_path)
                logger.info("Loaded species identification model")
            
            # Load eDNA analysis model
            edna_model_path = self.model_path / "edna_analyzer.pkl"
            if edna_model_path.exists():
                self.models['edna_analyzer'] = joblib.load(edna_model_path)
                logger.info("Loaded eDNA analysis model")
            
            # Load phylogenetic analysis model
            phylo_model_path = self.model_path / "phylogenetic_analyzer.pkl"
            if phylo_model_path.exists():
                self.models['phylogenetic_analyzer'] = joblib.load(phylo_model_path)
                logger.info("Loaded phylogenetic analysis model")
            
            # Load feature scalers
            scaler_path = self.model_path / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scalers['feature_scaler'] = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            
            logger.info("Molecular analysis models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading molecular models: {e}")
            # Initialize with default models if loading fails
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models if pre-trained models are not available"""
        self.models['species_identifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['edna_analyzer'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.models['phylogenetic_analyzer'] = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        self.scalers['feature_scaler'] = StandardScaler()
        logger.info("Initialized default molecular models")
    
    async def analyze_sequence(self, request) -> Dict[str, Any]:
        """Analyze DNA/RNA sequence data"""
        try:
            sequence_data = request.sequence_data
            sequence_type = request.sequence_type
            is_edna = request.is_edna
            
            # Validate sequence
            validation_result = self._validate_sequence(sequence_data)
            if not validation_result['valid']:
                return {
                    "success": False,
                    "error": validation_result['error'],
                    "sequence_analysis": None
                }
            
            # Extract sequence features
            sequence_features = self._extract_sequence_features(sequence_data, sequence_type)
            
            # Perform quality assessment
            quality_assessment = self._assess_sequence_quality(sequence_data, sequence_features)
            
            # Perform species identification
            species_identification = await self._identify_species_from_sequence(
                sequence_data, sequence_type, sequence_features)
            
            # Perform genetic diversity analysis
            genetic_diversity = self._analyze_genetic_diversity(sequence_features)
            
            # Perform phylogenetic analysis
            phylogenetic_analysis = await self._perform_phylogenetic_analysis(
                sequence_data, sequence_type, sequence_features)
            
            # Perform eDNA analysis if applicable
            edna_analysis = None
            if is_edna:
                edna_analysis = await self._analyze_edna_data(sequence_data, sequence_features)
            
            # Perform barcode analysis
            barcode_results = await self._analyze_barcode(sequence_data, sequence_type)
            
            return {
                "success": True,
                "sequence_analysis": {
                    "sequence_length": len(sequence_data),
                    "sequence_type": sequence_type,
                    "gc_content": sequence_features.get('gc_content', 0),
                    "at_content": sequence_features.get('at_content', 0),
                    "nucleotide_frequencies": {
                        'A': sequence_features.get('a_frequency', 0),
                        'T': sequence_features.get('t_frequency', 0),
                        'G': sequence_features.get('g_frequency', 0),
                        'C': sequence_features.get('c_frequency', 0)
                    }
                },
                "quality_assessment": quality_assessment,
                "species_identification": species_identification,
                "genetic_diversity": genetic_diversity,
                "phylogenetic_analysis": phylogenetic_analysis,
                "edna_analysis": edna_analysis,
                "barcode_results": barcode_results
            }
            
        except Exception as e:
            logger.error(f"Error in sequence analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "sequence_analysis": None
            }
    
    def _validate_sequence(self, sequence: str) -> Dict[str, Any]:
        """Validate DNA/RNA sequence"""
        if not sequence:
            return {"valid": False, "error": "Empty sequence"}
        
        # Check sequence length
        if len(sequence) < self.min_sequence_length:
            return {"valid": False, "error": f"Sequence too short (minimum {self.min_sequence_length} bp)"}
        
        if len(sequence) > self.max_sequence_length:
            return {"valid": False, "error": f"Sequence too long (maximum {self.max_sequence_length} bp)"}
        
        # Check for valid nucleotides
        valid_nucleotides = set('ATGCUatgcu')
        sequence_nucleotides = set(sequence.upper())
        
        if not sequence_nucleotides.issubset(valid_nucleotides):
            invalid_chars = sequence_nucleotides - valid_nucleotides
            return {"valid": False, "error": f"Invalid nucleotides found: {invalid_chars}"}
        
        # Check for ambiguous bases
        ambiguous_bases = set('NnRrYyWwSsKkMmBbDdHhVv')
        ambiguous_count = sum(1 for char in sequence if char in ambiguous_bases)
        ambiguous_percentage = ambiguous_count / len(sequence)
        
        if ambiguous_percentage > 0.1:  # More than 10% ambiguous bases
            return {"valid": False, "error": f"Too many ambiguous bases ({ambiguous_percentage:.1%})"}
        
        return {"valid": True, "ambiguous_percentage": ambiguous_percentage}
    
    def _extract_sequence_features(self, sequence: str, sequence_type: str) -> Dict[str, Any]:
        """Extract features from DNA/RNA sequence"""
        sequence = sequence.upper()
        length = len(sequence)
        
        # Basic composition
        a_count = sequence.count('A')
        t_count = sequence.count('T')
        g_count = sequence.count('G')
        c_count = sequence.count('C')
        u_count = sequence.count('U')
        n_count = sequence.count('N')
        
        # Calculate frequencies
        a_freq = a_count / length if length > 0 else 0
        t_freq = t_count / length if length > 0 else 0
        g_freq = g_count / length if length > 0 else 0
        c_freq = c_count / length if length > 0 else 0
        u_freq = u_count / length if length > 0 else 0
        n_freq = n_count / length if length > 0 else 0
        
        # Calculate GC content
        gc_content = (g_count + c_count) / length if length > 0 else 0
        at_content = (a_count + t_count) / length if length > 0 else 0
        
        # Calculate dinucleotide frequencies
        dinucleotide_freqs = self._calculate_dinucleotide_frequencies(sequence)
        
        # Calculate codon usage (for protein-coding sequences)
        codon_usage = {}
        if sequence_type.upper() in ['COI', 'CYTB', 'ND1', 'ND2', 'ND4', 'ND5']:
            codon_usage = self._calculate_codon_usage(sequence)
        
        # Calculate sequence complexity
        complexity = self._calculate_sequence_complexity(sequence)
        
        # Calculate repeat content
        repeat_content = self._calculate_repeat_content(sequence)
        
        return {
            "sequence_length": length,
            "a_frequency": a_freq,
            "t_frequency": t_freq,
            "g_frequency": g_freq,
            "c_frequency": c_freq,
            "u_frequency": u_freq,
            "n_frequency": n_freq,
            "gc_content": gc_content,
            "at_content": at_content,
            "dinucleotide_frequencies": dinucleotide_freqs,
            "codon_usage": codon_usage,
            "complexity": complexity,
            "repeat_content": repeat_content
        }
    
    def _calculate_dinucleotide_frequencies(self, sequence: str) -> Dict[str, float]:
        """Calculate dinucleotide frequencies"""
        dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC',
                        'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        
        dinucleotide_freqs = {}
        total_dinucleotides = len(sequence) - 1
        
        for dinucleotide in dinucleotides:
            count = sequence.count(dinucleotide)
            frequency = count / total_dinucleotides if total_dinucleotides > 0 else 0
            dinucleotide_freqs[dinucleotide] = frequency
        
        return dinucleotide_freqs
    
    def _calculate_codon_usage(self, sequence: str) -> Dict[str, float]:
        """Calculate codon usage frequencies"""
        # Ensure sequence length is multiple of 3
        if len(sequence) % 3 != 0:
            sequence = sequence[:-(len(sequence) % 3)]
        
        codons = {}
        total_codons = len(sequence) // 3
        
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if 'N' not in codon:  # Skip codons with ambiguous bases
                codons[codon] = codons.get(codon, 0) + 1
        
        # Calculate frequencies
        codon_freqs = {}
        for codon, count in codons.items():
            codon_freqs[codon] = count / total_codons if total_codons > 0 else 0
        
        return codon_freqs
    
    def _calculate_sequence_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity using Shannon entropy"""
        # Calculate nucleotide frequencies
        nucleotides = ['A', 'T', 'G', 'C']
        frequencies = [sequence.count(n) / len(sequence) for n in nucleotides]
        
        # Calculate Shannon entropy
        entropy = -sum(f * np.log2(f) for f in frequencies if f > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(nucleotides))
        complexity = entropy / max_entropy if max_entropy > 0 else 0
        
        return complexity
    
    def _calculate_repeat_content(self, sequence: str) -> Dict[str, float]:
        """Calculate repeat content in sequence"""
        # Simple repeat detection (can be enhanced)
        repeat_content = {}
        
        # Check for simple repeats (2-6 bp)
        for repeat_length in range(2, 7):
            repeats = 0
            for i in range(len(sequence) - repeat_length + 1):
                pattern = sequence[i:i+repeat_length]
                if sequence.count(pattern) > 1:
                    repeats += 1
            
            repeat_content[f"repeat_{repeat_length}bp"] = repeats / len(sequence) if len(sequence) > 0 else 0
        
        return repeat_content
    
    def _assess_sequence_quality(self, sequence: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Assess sequence quality"""
        # Calculate quality metrics
        length = len(sequence)
        gc_content = features.get('gc_content', 0)
        n_frequency = features.get('n_frequency', 0)
        complexity = features.get('complexity', 0)
        
        # Quality scoring
        quality_score = 1.0
        
        # Penalize for ambiguous bases
        if n_frequency > 0.05:  # More than 5% N's
            quality_score -= n_frequency * 2
        
        # Penalize for extreme GC content
        if gc_content < 0.2 or gc_content > 0.8:
            quality_score -= 0.2
        
        # Penalize for low complexity
        if complexity < 0.5:
            quality_score -= 0.3
        
        # Penalize for very short sequences
        if length < 200:
            quality_score -= 0.2
        
        quality_score = max(0, min(1, quality_score))  # Clamp between 0 and 1
        
        # Determine quality level
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.4:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "length": length,
            "gc_content": gc_content,
            "ambiguous_bases": n_frequency,
            "complexity": complexity,
            "recommendations": self._generate_quality_recommendations(quality_score, features)
        }
    
    def _generate_quality_recommendations(self, quality_score: float, features: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if features.get('n_frequency', 0) > 0.05:
            recommendations.append("High percentage of ambiguous bases. Consider resequencing.")
        
        if features.get('gc_content', 0) < 0.2 or features.get('gc_content', 0) > 0.8:
            recommendations.append("Extreme GC content. Verify sequence identity.")
        
        if features.get('complexity', 0) < 0.5:
            recommendations.append("Low sequence complexity. Check for contamination.")
        
        if features.get('sequence_length', 0) < 200:
            recommendations.append("Short sequence length. Consider longer reads.")
        
        if quality_score < 0.6:
            recommendations.append("Overall sequence quality is low. Consider quality control measures.")
        
        return recommendations
    
    async def _identify_species_from_sequence(self, sequence: str, sequence_type: str, 
                                            features: Dict[str, Any]) -> Dict[str, Any]:
        """Identify species from sequence data"""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_sequence_feature_vector(features)
            
            # Use species identification model
            if 'species_identifier' in self.models:
                prediction = self.models['species_identifier'].predict([feature_vector])[0]
                probabilities = self.models['species_identifier'].predict_proba([feature_vector])[0]
                
                # Get class names
                class_names = self.models['species_identifier'].classes_
                
                return {
                    "predicted_species": prediction,
                    "confidence": float(max(probabilities)),
                    "all_probabilities": dict(zip(class_names, probabilities.tolist())),
                    "method": "machine_learning"
                }
            else:
                # Fallback to BLAST-like search
                return await self._blast_like_search(sequence, sequence_type)
                
        except Exception as e:
            logger.error(f"Error in species identification: {e}")
            return {"error": str(e), "predicted_species": "unknown", "confidence": 0.0}
    
    def _prepare_sequence_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for sequence classification"""
        # Define expected features
        expected_features = [
            'sequence_length', 'a_frequency', 't_frequency', 'g_frequency',
            'c_frequency', 'gc_content', 'at_content', 'complexity'
        ]
        
        feature_vector = []
        for feature in expected_features:
            feature_vector.append(features.get(feature, 0.0))
        
        return np.array(feature_vector)
    
    async def _blast_like_search(self, sequence: str, sequence_type: str) -> Dict[str, Any]:
        """Perform BLAST-like search against reference databases"""
        try:
            # This is a simplified implementation
            # In practice, you would use BLAST or similar tools
            
            # Query external databases
            genbank_results = await self._query_genbank(sequence, sequence_type)
            bold_results = await self._query_bold(sequence, sequence_type)
            
            # Combine results
            all_results = genbank_results + bold_results
            
            if all_results:
                # Find best match
                best_match = max(all_results, key=lambda x: x.get('similarity', 0))
                
                return {
                    "predicted_species": best_match.get('species', 'unknown'),
                    "confidence": best_match.get('similarity', 0.0),
                    "database_matches": all_results,
                    "method": "database_search"
                }
            else:
                return {
                    "predicted_species": "unknown",
                    "confidence": 0.0,
                    "method": "database_search"
                }
                
        except Exception as e:
            logger.error(f"Error in BLAST-like search: {e}")
            return {"error": str(e), "predicted_species": "unknown", "confidence": 0.0}
    
    async def _query_genbank(self, sequence: str, sequence_type: str) -> List[Dict[str, Any]]:
        """Query GenBank for sequence matches"""
        try:
            # Simplified GenBank query
            # In practice, you would use the NCBI E-utilities API
            return []
        except Exception as e:
            logger.error(f"Error querying GenBank: {e}")
            return []
    
    async def _query_bold(self, sequence: str, sequence_type: str) -> List[Dict[str, Any]]:
        """Query BOLD for sequence matches"""
        try:
            # Simplified BOLD query
            # In practice, you would use the BOLD API
            return []
        except Exception as e:
            logger.error(f"Error querying BOLD: {e}")
            return []
    
    def _analyze_genetic_diversity(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze genetic diversity metrics"""
        # This is a simplified implementation
        # In practice, you would analyze multiple sequences
        
        gc_content = features.get('gc_content', 0)
        complexity = features.get('complexity', 0)
        
        # Calculate diversity metrics
        nucleotide_diversity = complexity  # Simplified
        heterozygosity = 1 - (gc_content**2 + (1-gc_content)**2)  # Simplified
        
        return {
            "nucleotide_diversity": nucleotide_diversity,
            "heterozygosity": heterozygosity,
            "gc_content": gc_content,
            "at_content": 1 - gc_content
        }
    
    async def _perform_phylogenetic_analysis(self, sequence: str, sequence_type: str, 
                                           features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform phylogenetic analysis"""
        try:
            # This is a simplified implementation
            # In practice, you would use proper phylogenetic tools
            
            # Calculate pairwise distances (simplified)
            distances = self._calculate_pairwise_distances(sequence)
            
            # Perform clustering
            clusters = self._perform_sequence_clustering(features)
            
            return {
                "pairwise_distances": distances,
                "clusters": clusters,
                "phylogenetic_signal": self._calculate_phylogenetic_signal(features)
            }
            
        except Exception as e:
            logger.error(f"Error in phylogenetic analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_pairwise_distances(self, sequence: str) -> Dict[str, float]:
        """Calculate pairwise distances (simplified)"""
        # This is a placeholder implementation
        return {"average_distance": 0.1, "max_distance": 0.2, "min_distance": 0.05}
    
    def _perform_sequence_clustering(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sequence clustering"""
        # This is a placeholder implementation
        return {"cluster_count": 1, "cluster_assignments": [0]}
    
    def _calculate_phylogenetic_signal(self, features: Dict[str, Any]) -> float:
        """Calculate phylogenetic signal strength"""
        # This is a placeholder implementation
        return 0.5
    
    async def _analyze_edna_data(self, sequence: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environmental DNA data"""
        try:
            # eDNA-specific analysis
            abundance = self._estimate_abundance(sequence, features)
            diversity = self._calculate_edna_diversity(features)
            contamination = self._assess_contamination(sequence, features)
            
            return {
                "abundance_estimate": abundance,
                "diversity_metrics": diversity,
                "contamination_assessment": contamination,
                "detection_confidence": self._calculate_detection_confidence(features)
            }
            
        except Exception as e:
            logger.error(f"Error in eDNA analysis: {e}")
            return {"error": str(e)}
    
    def _estimate_abundance(self, sequence: str, features: Dict[str, Any]) -> float:
        """Estimate species abundance from eDNA"""
        # Simplified abundance estimation
        length = len(sequence)
        complexity = features.get('complexity', 0)
        
        # Higher complexity and length suggest higher abundance
        abundance = (length / 1000) * complexity
        return min(abundance, 1.0)  # Normalize to 0-1
    
    def _calculate_edna_diversity(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate eDNA diversity metrics"""
        complexity = features.get('complexity', 0)
        gc_content = features.get('gc_content', 0)
        
        return {
            "shannon_diversity": complexity,
            "simpson_diversity": 1 - (gc_content**2 + (1-gc_content)**2),
            "species_richness": int(complexity * 10)  # Simplified
        }
    
    def _assess_contamination(self, sequence: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Assess sequence contamination"""
        n_frequency = features.get('n_frequency', 0)
        complexity = features.get('complexity', 0)
        
        contamination_score = n_frequency + (1 - complexity) * 0.5
        contamination_level = "low" if contamination_score < 0.3 else "medium" if contamination_score < 0.6 else "high"
        
        return {
            "contamination_score": contamination_score,
            "contamination_level": contamination_level,
            "recommendations": self._generate_contamination_recommendations(contamination_score)
        }
    
    def _generate_contamination_recommendations(self, contamination_score: float) -> List[str]:
        """Generate contamination control recommendations"""
        recommendations = []
        
        if contamination_score > 0.5:
            recommendations.append("High contamination detected. Consider additional filtering.")
        
        if contamination_score > 0.3:
            recommendations.append("Moderate contamination. Review sample preparation.")
        
        return recommendations
    
    def _calculate_detection_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate eDNA detection confidence"""
        quality_score = features.get('complexity', 0)
        length = features.get('sequence_length', 0)
        
        # Confidence based on quality and length
        confidence = quality_score * min(length / 500, 1.0)
        return min(confidence, 1.0)
    
    async def _analyze_barcode(self, sequence: str, sequence_type: str) -> Dict[str, Any]:
        """Analyze DNA barcode data"""
        try:
            # Barcode-specific analysis
            barcode_features = self._extract_barcode_features(sequence, sequence_type)
            
            # Identify barcode region
            barcode_region = self._identify_barcode_region(sequence, sequence_type)
            
            # Calculate barcode metrics
            barcode_metrics = self._calculate_barcode_metrics(sequence, barcode_features)
            
            return {
                "barcode_features": barcode_features,
                "barcode_region": barcode_region,
                "barcode_metrics": barcode_metrics,
                "barcode_quality": self._assess_barcode_quality(sequence, barcode_features)
            }
            
        except Exception as e:
            logger.error(f"Error in barcode analysis: {e}")
            return {"error": str(e)}
    
    def _extract_barcode_features(self, sequence: str, sequence_type: str) -> Dict[str, Any]:
        """Extract barcode-specific features"""
        features = self._extract_sequence_features(sequence, sequence_type)
        
        # Add barcode-specific features
        features["barcode_length"] = len(sequence)
        features["barcode_gc_content"] = features.get('gc_content', 0)
        
        return features
    
    def _identify_barcode_region(self, sequence: str, sequence_type: str) -> Dict[str, Any]:
        """Identify barcode region in sequence"""
        # Simplified barcode region identification
        return {
            "start": 0,
            "end": len(sequence),
            "length": len(sequence),
            "region_type": sequence_type
        }
    
    def _calculate_barcode_metrics(self, sequence: str, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate barcode-specific metrics"""
        return {
            "barcode_length": len(sequence),
            "gc_content": features.get('gc_content', 0),
            "complexity": features.get('complexity', 0),
            "quality_score": features.get('complexity', 0)
        }
    
    def _assess_barcode_quality(self, sequence: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Assess barcode quality"""
        quality_score = features.get('complexity', 0)
        length = len(sequence)
        
        # Barcode quality assessment
        if quality_score >= 0.8 and length >= 500:
            quality_level = "excellent"
        elif quality_score >= 0.6 and length >= 300:
            quality_level = "good"
        elif quality_score >= 0.4 and length >= 200:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "length": length,
            "recommendations": self._generate_barcode_recommendations(quality_score, length)
        }
    
    def _generate_barcode_recommendations(self, quality_score: float, length: int) -> List[str]:
        """Generate barcode quality recommendations"""
        recommendations = []
        
        if length < 500:
            recommendations.append("Barcode length is below recommended minimum (500 bp)")
        
        if quality_score < 0.6:
            recommendations.append("Barcode quality is low. Consider resequencing")
        
        return recommendations
    
    async def analyze_edna(self, request) -> Dict[str, Any]:
        """Analyze environmental DNA data"""
        # This is essentially the same as analyze_sequence but with eDNA focus
        return await self.analyze_sequence(request)
