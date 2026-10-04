# backward-compat shim — use models.jepa_world_model directly
from models.jepa_world_model import *  # noqa: F401, F403
from models.jepa_world_model import JEPAWorldModel as SensorimotorWorldModel  # noqa: F401
