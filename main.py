

"""CGM Service - Capteur de glycemie continu.
Service FastAPI pour lire les donnees de glycemie depuis un Patient Simulator."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Optional

import httpx
from dynaconf import Dynaconf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from google.cloud import firestore
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ============================================================================
# CONFIGURATION (Dynaconf)
# ============================================================================
settings = Dynaconf(
    settings_files=["settings.toml"],
    envvar_prefix="CGM",  # Variables d'env: CGM_PATIENT_URL, CGM_RETRY_ATTEMPTS, etc.
)

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMAS PYDANTIC
# ============================================================================
class GlucoseReading(BaseModel):
    """Modele de lecture de glycemie."""
    value: float = Field(..., description="Glycemie en mg/dL")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Horodatage de la lecture",
    )
    sensor_id: str = Field(
        default_factory=lambda: settings.DEFAULT_SENSOR_ID,
        description="Identifiant du capteur",
    )

    @field_validator("value")
    @classmethod
    def validate_glucose_range(cls, v: float) -> float:
        """Valide que la glycemie est dans la plage acceptable."""
        if v < settings.GLUCOSE_MIN or v > settings.GLUCOSE_MAX:
            raise ValueError(
                f"Glycémie doit être entre {settings.GLUCOSE_MIN} et {settings.GLUCOSE_MAX} mg/dL"
            )
        return v

class HealthStatus(BaseModel):
    """Modele de statut de sante du service."""
    status: str = Field(..., description="Statut du service (healthy/degraded)")
    service: str = Field(..., description="Nom du service")
    version: str = Field(..., description="Version du service")
    patient_simulator_url: str = Field(..., description="URL du Patient Simulator")
    patient_simulator_reachable: bool = Field(
        ..., description="Connectivite avec le Patient Simulator"
    )

# ============================================================================
# EXCEPTIONS
# ============================================================================
class PatientSimulatorError(Exception):
    """Exception pour les erreurs du Patient Simulator."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

# ============================================================================
# FIRESTORE CLIENT
# ============================================================================
db = firestore.Client(database=settings.FIRESTORE_DATABASE)
CACHE_COLLECTION = settings.CGM_CACHE_COLLECTION
CACHE_DOC_ID = settings.CGM_CACHE_DOCUMENT_ID

def save_glucose_cache(reading: GlucoseReading) -> None:
    """Sauvegarde la dernière lecture dans Firestore."""
    doc_ref = db.collection(CACHE_COLLECTION).document(CACHE_DOC_ID)
    doc_ref.set({
        "value": reading.value,
        "timestamp": reading.timestamp,
        "sensor_id": reading.sensor_id,
        "updated_at": datetime.now(timezone.utc),
    })
    logger.info("Cache Firestore mis à jour: %s mg/dL", reading.value)

def get_glucose_cache() -> Optional[GlucoseReading]:
    """Récupère la dernière lecture depuis Firestore."""
    doc_ref = db.collection(CACHE_COLLECTION).document(CACHE_DOC_ID)
    doc = doc_ref.get()
    
    if not doc.exists:
        return None
        
    data = doc.to_dict()
    return GlucoseReading(
        value=data["value"],
        timestamp=data["timestamp"],
        sensor_id=data.get("sensor_id", settings.DEFAULT_SENSOR_ID),
    )

# ============================================================================
# CLIENT PATIENT SIMULATOR
# ============================================================================
class PatientSimulatorClient:
    """Client pour le Patient Simulator avec retries automatiques."""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.PATIENT_URL
        self.timeout = httpx.Timeout(
            settings.HTTP_TIMEOUT_TOTAL,
            connect=settings.HTTP_TIMEOUT_CONNECT,
        )

    @retry(
        stop=stop_after_attempt(settings.RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=settings.RETRY_MULTIPLIER,
            min=settings.RETRY_MIN_WAIT,
            max=settings.RETRY_MAX_WAIT,
        ),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def read_glucose(self) -> GlucoseReading:
        """Lit la glycemie depuis le Patient Simulator."""
        logger.info("Lecture de la glycemie depuis %s/status", self.base_url)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(f"{self.base_url}/status")
            except httpx.TimeoutException:
                logger.error("Timeout lors de la connexion au Patient Simulator")
                raise
            except httpx.ConnectError:
                logger.error("Erreur de connexion au Patient Simulator")
                raise
            return self._parse_glucose_response(response)

    def _parse_glucose_response(self, response: httpx.Response) -> GlucoseReading:
        """Parse et valide la réponse du Patient Simulator."""
        if response.status_code != HTTPStatus.OK:
            logger.error(
                "Patient Simulator a retourne un status %d: %s",
                response.status_code,
                response.text,
            )
            raise PatientSimulatorError(
                f"Patient Simulator indisponible (HTTP {response.status_code})",
                status_code=response.status_code,
            )
        try:
            data = response.json()
        except ValueError:
            logger.error("Réponse JSON invalide du Patient Simulator")
            raise PatientSimulatorError("Réponse JSON invalide du Patient Simulator")
            
        if "glucose_level" not in data:
            logger.error("Champ 'glucose_level' manquant dans la réponse: %s", data)
            raise PatientSimulatorError(
                "Champ 'glucose_level' manquant dans la réponse"
            )
            
        glucose_value = data["glucose_level"]
        if glucose_value is None:
            logger.error("Valeur de glycemie nulle recue")
            raise PatientSimulatorError("Valeur de glycemie nulle")
            
        try:
            reading = GlucoseReading(value=glucose_value)
            logger.info("Glycemie lue avec succes: %s mg/dL", reading.value)
            return reading
        except ValueError:
            logger.error("Valeur de glycemie hors plage: %s", glucose_value)
            raise PatientSimulatorError(
                f"Valeur de glycemie hors plage valide (20-600 mg/dL): {glucose_value}"
            )

    async def check_health(self) -> bool:
        """Vérifie la connectivité avec le Patient Simulator."""
        try:
            timeout = httpx.Timeout(settings.HEALTH_CHECK_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{self.base_url}/status")
                return response.status_code == HTTPStatus.OK
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
            logger.warning("Health check: Patient Simulator non joignable: %s", e)
            return False

# Instance globale du client
patient_client = PatientSimulatorClient()

# ============================================================================
# APPLICATION FASTAPI
# ============================================================================
def get_remote_address(request: Request) -> str:
    """Récupère l'adresse IP pour le rate limiting."""
    return request.client.host if request.client else "127.0.0.1"

limiter = Limiter(key_func=get_remote_address)

async def polling_loop():
    """Boucle de polling automatique - lit la glycemie toutes les 5 minutes."""
    logger.info("Démarrage de la boucle de polling automatique")
    while True:
        try:
            reading = await patient_client.read_glucose()
            save_glucose_cache(reading)  # Écriture Firestore au lieu de variable globale
            logger.info(
                "Polling automatique: glycemie %s mg/dL",
                reading.value,
            )
        except Exception as e:
            logger.error("Erreur dans la boucle de polling: %s", e)
        await asyncio.sleep(settings.POLLING_INTERVAL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    # Startup
    logger.info("Démarrage du CGM Service")
    asyncio.create_task(polling_loop())
    yield
    # Shutdown (si nécessaire)
    logger.info("Arrêt du CGM Service")

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description="Service de capteur de glycemie continu (CGM).",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/glucose", response_model=GlucoseReading)
@limiter.limit(settings.RATE_LIMIT)
async def get_current_glucose(request: Request) -> GlucoseReading:
    """Retourne la derniere lecture de glycemie depuis le cache Firestore.
    Le polling en arriere-plan met a jour le cache toutes les 5 minutes.
    Si aucune lecture n'est disponible, fait un appel direct au Patient Simulator.
    """
    # 1. Essayer de lire depuis le cache Firestore
    cached_reading = get_glucose_cache()
    
    if cached_reading is not None:
        logger.info("Retour de la glycemie depuis le cache Firestore: %s mg/dL", cached_reading.value)
        return cached_reading
        
    # 2. Fallback : appel direct au Patient Simulator si cache vide
    logger.warning("Cache Firestore vide, appel direct au Patient Simulator")
    try:
        reading = await patient_client.read_glucose()
        # Sauvegarder dans le cache pour les prochaines requêtes
        save_glucose_cache(reading)
        logger.info("Lecture directe Patient Simulator: %s mg/dL", reading.value)
        return reading
    except Exception as e:
        logger.error("Erreur lors de l'appel direct au Patient Simulator: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Aucune lecture disponible - le service demarre et Patient Simulator indisponible",
        )

@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Endpoint de verification de sante du service."""
    patient_reachable = await patient_client.check_health()
    status = "healthy" if patient_reachable else "degraded"
    
    logger.info(
        "Health check: %s, Patient Simulator reachable: %s",
        status,
        patient_reachable,
    )
    
    return HealthStatus(
        status=status,
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        patient_simulator_url=settings.PATIENT_URL,
        patient_simulator_reachable=patient_reachable,
    )

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Gestionnaire global d'exceptions non gerees."""
    logger.exception("Erreur non geree: %s", exc)
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content={"detail": "Erreur interne du service"},
    )




