"""
Configuration centralisée pour le projet House Price Prediction
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Chemin racine du projet
PROJECT_ROOT = Path(__file__).parent


@dataclass
class DataConfig:
    """Configuration pour les données"""
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_path: Path = PROJECT_ROOT / "data" / "processed"
    dataset_filename: str = "data.csv"
    
    # Paramètres de nettoyage
    missing_threshold: float = 0.5  # Seuil pour supprimer les colonnes avec trop de valeurs manquantes
    outlier_method: str = "iqr"  # Méthode de détection des outliers: 'iqr', 'zscore', 'isolation'
    outlier_threshold: float = 1.5  # Multiplicateur IQR ou seuil Z-score
    
    # Validation des données
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    @property
    def raw_data_file(self) -> Path:
        return self.raw_data_path / self.dataset_filename
    
    @property
    def processed_data_file(self) -> Path:
        return self.processed_data_path / "processed_data.csv"


@dataclass 
class ModelConfig:
    """Configuration pour les modèles ML"""
    models_path: Path = PROJECT_ROOT / "models"
    
    # Modèles à tester
    models_to_test: List[str] = None
    
    # Hyperparamètres par défaut
    default_params: Dict[str, Dict[str, Any]] = None
    
    # Validation croisée
    cv_folds: int = 5
    scoring: str = "neg_mean_squared_error"
    
    # Seuils de performance
    min_r2_score: float = 0.7
    max_rmse: float = 50000  # En fonction de l'échelle des prix
    
    # Sauvegarde
    model_filename: str = "best_model.joblib"
    scaler_filename: str = "scaler.joblib"
    
    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = [
                "linear_regression",
                "random_forest", 
                "gradient_boosting",
                "xgboost"
            ]
            
        if self.default_params is None:
            self.default_params = {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                },
                "gradient_boosting": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42
                },
                "xgboost": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42
                }
            }
    
    @property
    def model_file(self) -> Path:
        return self.models_path / self.model_filename
    
    @property 
    def scaler_file(self) -> Path:
        return self.models_path / self.scaler_filename


@dataclass
class APIConfig:
    """Configuration pour l'API"""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("API_DEBUG", "False").lower() == "true"
    
    # Limites de l'API
    max_request_size: int = 1024 * 1024  # 1MB
    rate_limit: int = 100  # Requêtes par minute
    
    # Validation des inputs
    required_features: List[str] = None
    feature_ranges: Dict[str, tuple] = None
    
    def __post_init__(self):
        # Ces valeurs seront mises à jour après l'exploration des données
        if self.required_features is None:
            self.required_features = []
            
        if self.feature_ranges is None:
            self.feature_ranges = {}


@dataclass
class LoggingConfig:
    """Configuration pour les logs"""
    logs_path: Path = PROJECT_ROOT / "logs"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Format des logs
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Fichiers de logs
    app_log_file: str = "app.log"
    api_log_file: str = "api.log" 
    model_log_file: str = "model.log"
    
    # Rotation des logs
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Logs structurés pour l'API
    structured_logging: bool = True
    log_requests: bool = True
    log_predictions: bool = True
    
    @property
    def app_log_path(self) -> Path:
        return self.logs_path / self.app_log_file
    
    @property
    def api_log_path(self) -> Path:
        return self.logs_path / self.api_log_file
        
    @property 
    def model_log_path(self) -> Path:
        return self.logs_path / self.model_log_file


@dataclass
class DockerConfig:
    """Configuration pour Docker"""
    image_name: str = "house-price-api"
    image_tag: str = os.getenv("IMAGE_TAG", "latest")
    container_port: int = 8000
    
    # Build
    dockerfile_path: str = "Dockerfile"
    docker_context: str = "."
    
    # Environment
    python_version: str = "3.11"
    base_image: str = f"python:{python_version}-slim"


@dataclass
class Config:
    """Configuration principale du projet"""
    
    # Environnement
    environment: str = os.getenv("ENVIRONMENT", "development")
    project_name: str = "house-price-prediction"
    version: str = "0.1.0"
    
    # Sous-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    
    def __post_init__(self):
        # Créer les dossiers nécessaires
        self._create_directories()
    
    def _create_directories(self):
        """Crée les dossiers nécessaires pour le projet"""
        directories = [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.model.models_path,
            self.logging.logs_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property 
    def is_development(self) -> bool:
        return self.environment.lower() == "development"


# Instance globale de configuration
config = Config()

# Fonction utilitaire pour recharger la configuration
def reload_config():
    """Recharge la configuration depuis les variables d'environnement"""
    global config
    load_dotenv(override=True)
    config = Config()
    return config

# Validation de la configuration
def validate_config() -> List[str]:
    """
    Valide la configuration et retourne une liste d'erreurs
    """
    errors = []
    
    # Vérifier que les chemins critiques existent
    if not config.data.raw_data_path.exists():
        errors.append(f"Le dossier de données brutes n'existe pas: {config.data.raw_data_path}")
    
    # Vérifier les valeurs numériques
    if not 0 < config.data.test_size < 1:
        errors.append("test_size doit être entre 0 et 1")
        
    if not 0 < config.data.validation_size < 1:
        errors.append("validation_size doit être entre 0 et 1")
    
    if config.model.cv_folds < 2:
        errors.append("cv_folds doit être >= 2")
        
    # Vérifier les ports
    if not 1024 <= config.api.port <= 65535:
        errors.append("Le port API doit être entre 1024 et 65535")
    
    return errors

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration du projet House Price Prediction")
    print("=" * 60)
    
    print(f"📁 Projet: {config.project_name} v{config.version}")
    print(f"🌍 Environnement: {config.environment}")
    print(f"📊 Données: {config.data.raw_data_file}")
    print(f"🤖 Modèles: {config.model.models_path}")
    print(f"🚀 API: {config.api.host}:{config.api.port}")
    print(f"📝 Logs: {config.logging.logs_path}")
    
    # Validation
    errors = validate_config()
    if errors:
        print("\n❌ Erreurs de configuration:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("\n✅ Configuration valide!")
    
    # Afficher les modèles à tester
    print(f"\n🔬 Modèles à tester: {', '.join(config.model.models_to_test)}")