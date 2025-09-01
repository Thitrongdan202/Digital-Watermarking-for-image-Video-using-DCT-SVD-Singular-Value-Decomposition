import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class WatermarkDetector:
    """Statistical watermark detector based on DCT-SVD singular value analysis."""

    def __init__(self) -> None:
        self.models = {}  # Store models for different feature lengths
        self.scalers = {}  # Store scalers for each feature length
        self.trained_features = set()  # Track which feature lengths we've trained

    def _ensure_trained(self, feature_length: int) -> None:
        """Ensure we have a trained model for this specific feature length."""
        if feature_length not in self.trained_features:
            # Create model and scaler for this feature length
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                class_weight='balanced'
            )
            scaler = StandardScaler()
            
            # Generate training data based on realistic DCT singular value distributions
            n_samples = 600
            
            # Generate "clean" images - natural DCT singular value patterns
            X_clean = []
            for _ in range(n_samples):
                # More realistic singular values for DCT
                # First value much larger, then exponential decay with more variation
                first_val = np.random.uniform(15000, 60000)
                decay_rate = np.random.uniform(4, 10)  # Variable decay
                decay_values = first_val * np.exp(-np.linspace(0, decay_rate, feature_length-1))
                
                # Add realistic texture variation
                texture_variation = np.random.uniform(0.7, 1.4, feature_length-1)
                decay_values *= texture_variation
                
                base_values = np.concatenate([[first_val], decay_values])
                
                # Add natural image noise
                noise_level = np.random.uniform(0.02, 0.08)
                noise = np.random.normal(0, base_values * noise_level)
                s_clean = np.maximum(base_values + noise, 0.1)
                X_clean.append(s_clean)
            X_clean = np.array(X_clean)
            
            # Generate "watermarked" images - with typical embedding effects
            X_watermarked = []
            for _ in range(n_samples):
                # Start with a clean pattern (similar to above)
                first_val = np.random.uniform(15000, 60000)
                decay_rate = np.random.uniform(4, 10)
                decay_values = first_val * np.exp(-np.linspace(0, decay_rate, feature_length-1))
                texture_variation = np.random.uniform(0.7, 1.4, feature_length-1)
                decay_values *= texture_variation
                base_values = np.concatenate([[first_val], decay_values])
                
                noise_level = np.random.uniform(0.02, 0.08)
                base_noise = np.random.normal(0, base_values * noise_level)
                s_base = np.maximum(base_values + base_noise, 0.1)
                
                # Add watermark effect - more conservative and realistic
                alpha = np.random.uniform(0.04, 0.10)  # More conservative embedding
                
                # Watermark pattern - based on actual DCT-SVD behavior
                watermark_strength = np.zeros(feature_length)
                
                # Primary effect: boost in early-middle frequencies (where watermark energy concentrates)
                significant_components = min(feature_length//3, 30)
                for i in range(1, significant_components):  # Skip first component
                    # Exponential decay of watermark strength
                    strength = np.random.exponential(scale=50) * np.exp(-i/10)
                    watermark_strength[i] = strength
                
                s_watermarked = s_base + alpha * watermark_strength
                X_watermarked.append(s_watermarked)
            X_watermarked = np.array(X_watermarked)
            
            # Combine training data
            X = np.vstack([X_clean, X_watermarked])
            y = np.array([0] * n_samples + [1] * n_samples)
            
            # Add statistical features that are discriminative
            X_engineered = []
            for sample in X:
                features = self._extract_statistical_features(sample)
                X_engineered.append(features)
            X_engineered = np.array(X_engineered)
            
            # Fit scaler and transform data
            X_scaled = scaler.fit_transform(X_engineered)
            
            # Train the model
            model.fit(X_scaled, y)
            
            # Store the trained model and scaler
            self.models[feature_length] = model
            self.scalers[feature_length] = scaler
            self.trained_features.add(feature_length)
    
    def _extract_statistical_features(self, singular_values: np.ndarray) -> np.ndarray:
        """Extract discriminative statistical features from singular values."""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(singular_values),
            np.std(singular_values),
            np.var(singular_values),
            np.max(singular_values),
            np.min(singular_values),
        ])
        
        if len(singular_values) > 10:
            # Frequency domain analysis
            n = len(singular_values)
            low_freq = singular_values[:n//4]
            mid_freq = singular_values[n//4:3*n//4]  
            high_freq = singular_values[3*n//4:]
            
            # Energy distribution
            total_energy = np.sum(singular_values**2) + 1e-10
            features.extend([
                np.sum(low_freq**2) / total_energy,   # Low frequency energy ratio
                np.sum(mid_freq**2) / total_energy,   # Mid frequency energy ratio  
                np.sum(high_freq**2) / total_energy,  # High frequency energy ratio
            ])
            
            # Critical DCT-SVD watermark indicators
            # First singular value ratio (should be less affected)
            features.extend([
                singular_values[0] / (np.sum(singular_values[1:min(10, n)]) + 1e-10),
                np.mean(singular_values[1:min(20, n//4)]) / (singular_values[0] + 1e-10),  # Early components boost
            ])
            
            # Ratios that might indicate watermarking
            features.extend([
                np.mean(mid_freq) / (np.mean(low_freq) + 1e-10),    # Mid/Low ratio
                np.std(mid_freq) / (np.std(low_freq) + 1e-10),     # Std ratio
                np.var(mid_freq) / (np.var(low_freq) + 1e-10),     # Variance ratio
            ])
            
            # Spectral characteristics
            features.extend([
                np.sum(np.diff(singular_values)**2),  # Roughness/smoothness
                np.corrcoef(np.arange(n), singular_values)[0,1] if n > 1 else 0,  # Trend correlation
            ])
            
            # Watermark-specific patterns
            # Check for unnatural spikes in early frequencies (typical of DCT-SVD embedding)
            if n > 20:
                early_vals = singular_values[1:20]
                expected_decay = singular_values[0] * np.exp(-np.linspace(0, 3, 19))
                deviation = np.mean(np.abs(early_vals - expected_decay) / (expected_decay + 1e-10))
                features.append(deviation)
                
                # Entropy changes in different regions
                def safe_entropy(x):
                    x_norm = x / (np.sum(x) + 1e-10)
                    return -np.sum(x_norm * np.log(x_norm + 1e-10))
                
                features.extend([
                    safe_entropy(low_freq),
                    safe_entropy(mid_freq),
                    safe_entropy(high_freq),
                ])
            
        return np.array(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict whether watermark is present based on singular values."""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        original_feature_length = features.shape[1]
        self._ensure_trained(original_feature_length)
        
        # Extract statistical features
        X_features = []
        for row in features:
            statistical_features = self._extract_statistical_features(row)
            X_features.append(statistical_features)
        X_features = np.array(X_features)
        
        # Scale features
        scaler = self.scalers[original_feature_length]
        X_scaled = scaler.transform(X_features)
        
        # Get the appropriate model and predict
        model = self.models[original_feature_length]
        return model.predict(X_scaled)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability of watermark presence."""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        original_feature_length = features.shape[1]
        self._ensure_trained(original_feature_length)
        
        # Extract statistical features
        X_features = []
        for row in features:
            statistical_features = self._extract_statistical_features(row)
            X_features.append(statistical_features)
        X_features = np.array(X_features)
        
        # Scale features
        scaler = self.scalers[original_feature_length]
        X_scaled = scaler.transform(X_features)
        
        # Get the appropriate model and predict probabilities
        model = self.models[original_feature_length]
        return model.predict_proba(X_scaled)
