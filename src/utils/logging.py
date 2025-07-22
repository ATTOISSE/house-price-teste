"""
Configuration du logging structuré
"""

import logging
import logging.handlers
import structlog
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from config import config


def setup_logging():
    """
    Configure le logging structuré pour l'application
    """
    
    # Créer le dossier de logs s'il n'existe pas
    config.logging.logs_path.mkdir(parents=True, exist_ok=True)
    
    # Configuration de structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configuration du logger standard Python
    logging.basicConfig(
        format=config.logging.log_format,
        datefmt=config.logging.date_format,
        level=getattr(logging, config.logging.log_level.upper())
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Retourne un logger structuré
    
    Args:
        name: Nom du logger
        
    Returns:
        Logger structuré
    """
    return structlog.get_logger(name)


def setup_file_handlers():
    """
    Configure les handlers de fichiers pour les différents types de logs
    """
    
    # Handler pour les logs généraux
    app_handler = logging.handlers.RotatingFileHandler(
        config.logging.app_log_path,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count
    )
    app_handler.setLevel(logging.INFO)
    
    # Handler pour les logs API
    api_handler = logging.handlers.RotatingFileHandler(
        config.logging.api_log_path,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count
    )
    api_handler.setLevel(logging.INFO)
    
    # Handler pour les logs de modèles
    model_handler = logging.handlers.RotatingFileHandler(
        config.logging.model_log_path,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count
    )
    model_handler.setLevel(logging.INFO)
    
    # Formateur JSON pour les logs structurés
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    for handler in [app_handler, api_handler, model_handler]:
        handler.setFormatter(formatter)
    
    # Ajouter les handlers aux loggers appropriés
    logging.getLogger("app").addHandler(app_handler)
    logging.getLogger("api").addHandler(api_handler)
    logging.getLogger("model").addHandler(model_handler)


class APILogger:
    """
    Logger spécialisé pour l'API avec format structuré
    """
    
    def __init__(self):
        self.logger = get_logger("api")
    
    def log_request(self, 
                   method: str,
                   endpoint: str,
                   client_ip: str,
                   user_agent: str = None,
                   request_id: str = None):
        """Log d'une requête entrante"""
        self.logger.info(
            "API request received",
            method=method,
            endpoint=endpoint,
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_prediction(self,
                      features: Dict[str, Any],
                      prediction: float,
                      confidence: float = None,
                      processing_time: float = None,
                      request_id: str = None):
        """Log d'une prédiction"""
        self.logger.info(
            "Prediction made",
            features=features,
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=processing_time * 1000 if processing_time else None,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_error(self,
                  error: str,
                  error_type: str,
                  endpoint: str = None,
                  request_id: str = None,
                  stack_trace: str = None):
        """Log d'une erreur"""
        self.logger.error(
            "API error occurred",
            error=error,
            error_type=error_type,
            endpoint=endpoint,
            request_id=request_id,
            stack_trace=stack_trace,
            timestamp=datetime.utcnow().isoformat()
        )


class ModelLogger:
    """
    Logger spécialisé pour les opérations de modèles ML
    """
    
    def __init__(self):
        self.logger = get_logger("model")
    
    def log_training_start(self, 
                          model_type: str,
                          dataset_shape: tuple,
                          parameters: Dict[str, Any]):
        """Log du début d'entraînement"""
        self.logger.info(
            "Model training started",
            model_type=model_type,
            dataset_rows=dataset_shape[0],
            dataset_cols=dataset_shape[1],
            parameters=parameters,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_training_complete(self,
                             model_type: str,
                             metrics: Dict[str, float],
                             training_time: float):
        """Log de fin d'entraînement"""
        self.logger.info(
            "Model training completed",
            model_type=model_type,
            metrics=metrics,
            training_time_seconds=training_time,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_model_saved(self,
                       model_type: str,
                       model_path: str,
                       model_size_mb: float):
        """Log de sauvegarde de modèle"""
        self.logger.info(
            "Model saved",
            model_type=model_type,
            model_path=model_path,
            model_size_mb=model_size_mb,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_prediction_batch(self,
                           model_type: str,
                           batch_size: int,
                           prediction_time: float):
        """Log de prédiction en lot"""
        self.logger.info(
            "Batch prediction completed",
            model_type=model_type,
            batch_size=batch_size,
            prediction_time_seconds=prediction_time,
            avg_time_per_prediction=prediction_time / batch_size,
            timestamp=datetime.utcnow().isoformat()
        )


# Initialisation automatique du logging
if config.logging.structured_logging:
    setup_logging()
    setup_file_handlers()

# Instances globales des loggers
api_logger = APILogger()
model_logger = ModelLogger()

# Logger général pour l'application
app_logger = get_logger("app")