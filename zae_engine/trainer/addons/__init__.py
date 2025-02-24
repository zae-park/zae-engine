from .core import AddOnBase
from .mpu import MultiGPUAddon
from .web_logger import NeptuneLoggerAddon, WandBLoggerAddon
from .state_manager import StateManagerAddon
from .mix_precision import PrecisionMixerAddon
from .signal_handler import SignalHandlerAddon


# Add-on case definitions
ADDON_CASE_CORE = 0  # Trainer의 핵심 로직 변경 (예: MultiGPUAddon)
ADDON_CASE_INDEPENDENT = 1  # 독립적으로 동작하는 Add-on (예: StateManagerAddon)
ADDON_CASE_TRIGGER = 2  # 이벤트 트리거 역할 (예: SignalHandlerAddon)
ADDON_CASE_DEPENDENT = 3  # 이전 Add-on의 데이터를 받아야 동작하는 Add-on (예: WebLoggerAddon)
