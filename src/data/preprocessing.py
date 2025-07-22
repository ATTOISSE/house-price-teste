# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HousePricePreprocessor:
    """
    Classe pour le préprocessing des données de prix immobilier
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None
        
    def load_data(self, file_path):
        """
        Charge les données depuis un fichier CSV
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Données chargées: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise
    
    def remove_outliers_iqr(self, df, columns, multiplier=1.5):
        """
        Supprime les valeurs aberrantes en utilisant la méthode IQR
        """
        df_clean = df.copy()
        outliers_removed = 0
        
        for col in columns:
            if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                df_clean = df_clean[~outliers_mask]
                outliers_removed += outliers_count
                
                logger.info(f"Colonne {col}: {outliers_count} valeurs aberrantes supprimées")
        
        logger.info(f"Total des valeurs aberrantes supprimées: {outliers_removed}")
        logger.info(f"Forme après suppression des outliers: {df_clean.shape}")
        
        return df_clean
    
    def remove_high_correlation_features(self, df, target_col, threshold=0.95):
        """
        Supprime les features ayant une corrélation élevée entre elles
        """
        # Sélectionner seulement les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        # Calculer la matrice de corrélation
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Identifier les paires de features hautement corrélées
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Trouver les features à supprimer
        to_remove = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        logger.info(f"Features supprimées pour corrélation élevée (>{threshold}): {to_remove}")
        
        # Retourner le dataframe sans les features hautement corrélées
        return df.drop(columns=to_remove), to_remove
    
    def remove_low_variance_features(self, df, target_col, threshold=0.01):
        """
        Supprime les features avec une faible variance
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        low_variance_cols = []
        for col in numeric_cols:
            if df[col].var() < threshold:
                low_variance_cols.append(col)
        
        logger.info(f"Features supprimées pour faible variance (<{threshold}): {low_variance_cols}")
        
        return df.drop(columns=low_variance_cols), low_variance_cols
    
    def encode_categorical_features(self, df, target_col):
        """
        Encode les variables catégorielles
        """
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            logger.info(f"Variable {col} encodée")
        
        return df_encoded
    
    def select_best_features(self, X, y, k=10):
        """
        Sélectionne les k meilleures features
        """
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        logger.info(f"Features sélectionnées ({len(self.selected_features)}): {self.selected_features}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def scale_features(self, X):
        """
        Standardise les features
        """
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def preprocess_pipeline(self, df, target_col='price', remove_outliers=True, 
                          correlation_threshold=0.95, variance_threshold=0.01, 
                          select_k_best=15):
        """
        Pipeline complet de preprocessing
        """
        logger.info("=== DÉBUT DU PREPROCESSING ===")
        logger.info(f"Forme initiale: {df.shape}")
        
        # 1. Copie du dataframe
        df_processed = df.copy()
        
        # 2. Suppression des valeurs aberrantes
        if remove_outliers:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != target_col]
            df_processed = self.remove_outliers_iqr(df_processed, numeric_cols)
        
        # 3. Encodage des variables catégorielles
        df_processed = self.encode_categorical_features(df_processed, target_col)
        
        # 4. Suppression des features avec faible variance
        df_processed, removed_variance = self.remove_low_variance_features(
            df_processed, target_col, variance_threshold
        )
        
        # 5. Suppression des features hautement corrélées
        df_processed, removed_corr = self.remove_high_correlation_features(
            df_processed, target_col, correlation_threshold
        )
        
        # 6. Séparation X et y
        if target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
        else:
            raise ValueError(f"Colonne target '{target_col}' non trouvée")
        
        # 7. Sélection des meilleures features
        if select_k_best and select_k_best > 0:
            X = self.select_best_features(X, y, select_k_best)
        
        # 8. Standardisation
        X_scaled = self.scale_features(X)
        
        logger.info(f"Forme finale: X={X_scaled.shape}, y={y.shape}")
        logger.info("=== FIN DU PREPROCESSING ===")
        
        return X_scaled, y, {
            'removed_outliers': remove_outliers,
            'removed_variance_features': removed_variance if remove_outliers else [],
            'removed_correlation_features': removed_corr if remove_outliers else [],
            'selected_features': self.selected_features,
            'final_shape': X_scaled.shape
        }
    
    def save_processed_data(self, X, y, output_dir='data/processed', info=None):
        """
        Sauvegarde les données préprocessées
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarde des features et target
        X.to_csv(os.path.join(output_dir, 'X_processed.csv'), index=False)
        y.to_csv(os.path.join(output_dir, 'y_processed.csv'), index=False)
        
        # Sauvegarde des informations de preprocessing
        if info:
            info_df = pd.DataFrame([info])
            info_df.to_csv(os.path.join(output_dir, 'preprocessing_info.csv'), index=False)
        
        logger.info(f"Données sauvegardées dans {output_dir}")
    
    def generate_preprocessing_report(self, df_original, X_final, y_final, info):
        """
        Génère un rapport de preprocessing
        """
        report = {
            'original_shape': df_original.shape,
            'final_shape': (X_final.shape[0], X_final.shape[1] + 1),  # +1 pour target
            'features_removed': len(df_original.columns) - len(X_final.columns) - 1,  # -1 pour target
            'selected_features': info.get('selected_features', []),
            'removed_variance_features': info.get('removed_variance_features', []),
            'removed_correlation_features': info.get('removed_correlation_features', [])
        }
        
        return report

def main():
    """
    Fonction principale pour exécuter le preprocessing
    """
    # Initialisation du preprocessor
    preprocessor = HousePricePreprocessor()
    
    # Chargement des données (à adapter selon votre dataset)
    try:
        # Modifiez le chemin selon votre structure
        data_path = "../../data/raw/data.csv"  # ou le nom de votre fichier
        df = preprocessor.load_data(data_path)
        
        # Affichage des informations initiales
        logger.info("=== INFORMATIONS SUR LE DATASET ===")
        logger.info(f"Forme: {df.shape}")
        logger.info(f"Colonnes: {list(df.columns)}")
        logger.info(f"Types de données:\n{df.dtypes}")
        logger.info(f"Valeurs manquantes:\n{df.isnull().sum()}")
        
        # Preprocessing (adaptez le nom de la colonne target)
        target_column = 'price'  # Modifiez selon votre dataset
        X_processed, y_processed, preprocessing_info = preprocessor.preprocess_pipeline(
            df, 
            target_col=target_column,
            remove_outliers=True,
            correlation_threshold=0.95,
            variance_threshold=0.01,
            select_k_best=15
        )
        
        # Sauvegarde
        preprocessor.save_processed_data(X_processed, y_processed, info=preprocessing_info)
        
        # Génération du rapport
        report = preprocessor.generate_preprocessing_report(df, X_processed, y_processed, preprocessing_info)
        
        logger.info("=== RAPPORT DE PREPROCESSING ===")
        for key, value in report.items():
            logger.info(f"{key}: {value}")
            
        return X_processed, y_processed, preprocessor
        
    except Exception as e:
        logger.error(f"Erreur dans le preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()