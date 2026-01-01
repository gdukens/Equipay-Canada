"""
Custom Exceptions for EquiPay Canada
====================================

Centralized exception classes for better error handling and debugging.
"""


class EquiPayError(Exception):
    """Base exception for EquiPay Canada project."""
    pass


class DataLoadError(EquiPayError):
    """Raised when data loading fails."""
    
    def __init__(self, message: str, source: str = None, path: str = None):
        self.source = source
        self.path = path
        super().__init__(f"{message} (source: {source}, path: {path})")


class DataValidationError(EquiPayError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, column: str = None, expected: str = None, actual: str = None):
        self.column = column
        self.expected = expected
        self.actual = actual
        details = f"column: {column}, expected: {expected}, actual: {actual}"
        super().__init__(f"{message} ({details})")


class MissingColumnError(DataValidationError):
    """Raised when a required column is missing from DataFrame."""
    
    def __init__(self, column: str, available_columns: list = None):
        self.available_columns = available_columns or []
        message = f"Required column '{column}' not found"
        if available_columns:
            message += f". Available columns: {available_columns[:10]}..."
        super().__init__(message, column=column)


class ModelError(EquiPayError):
    """Raised when model operations fail."""
    pass


class ModelNotTrainedError(ModelError):
    """Raised when trying to use a model that hasn't been trained."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        message = f"Model '{model_name}' has not been trained" if model_name else "Model has not been trained"
        super().__init__(message)


class ModelNotFoundError(ModelError):
    """Raised when a saved model cannot be found."""
    
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Model file not found: {path}")


class ConfigurationError(EquiPayError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(f"{message} (key: {config_key})")


class AnalysisError(EquiPayError):
    """Raised when statistical analysis fails."""
    pass


class InsufficientDataError(AnalysisError):
    """Raised when there's not enough data for analysis."""
    
    def __init__(self, required: int, actual: int, context: str = None):
        self.required = required
        self.actual = actual
        self.context = context
        message = f"Insufficient data: need at least {required} samples, got {actual}"
        if context:
            message = f"{context}: {message}"
        super().__init__(message)


class FairnessError(EquiPayError):
    """Raised when fairness analysis fails."""
    pass


class APIError(EquiPayError):
    """Raised for API-related errors."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.status_code = status_code
        super().__init__(message)
