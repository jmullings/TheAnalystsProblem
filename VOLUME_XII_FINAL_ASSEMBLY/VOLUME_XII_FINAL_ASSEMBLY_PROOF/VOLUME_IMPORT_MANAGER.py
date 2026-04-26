# ============================================================================
# VOLUME_IMPORT_MANAGER.py
# Centralized import orchestration for The Analyst's Problem volumes
# ============================================================================


from __future__ import annotations


import os
import sys
import importlib
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union


logger = logging.getLogger(__name__)


T = TypeVar("T")


class VolumeStatus(Enum):
    NOT_ATTEMPTED = auto()
    IMPORTING = auto()
    AVAILABLE = auto()
    MISSING = auto()
    PARTIAL = auto()
    ERROR = auto()


@dataclass
class FunctionSpec:
    name: str
    required: bool = True
    default: Optional[Callable] = None
    alias: Optional[str] = None

    @property
    def local_name(self) -> str:
        return self.alias or self.name


@dataclass
class VolumeConfig:
    volume_id: str
    module_path: str
    functions: List[FunctionSpec] = field(default_factory=list)
    optional: bool = False
    post_import_hook: Optional[Callable[[Dict[str, Any]], bool]] = None

    def __post_init__(self):
        if not self.module_path:
            raise ValueError(f"Volume {self.volume_id} requires a module_path")


class VolumeImportError(Exception):
    def __init__(self, volume_id: str, message: str, original: Optional[Exception] = None):
        super().__init__(f"[{volume_id}] {message}")
        self.volume_id = volume_id
        self.original = original


class VolumeImporter:
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).parent.parent
        )
        self._volumes: Dict[str, VolumeConfig] = {}
        self._status: Dict[str, VolumeStatus] = {}
        self._modules: Dict[str, Any] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}
        self._errors: Dict[str, str] = {}

        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

    # Convenience access to underlying modules (used for debugging / alignment)
    @property
    def modules(self) -> Dict[str, Any]:
        return self._modules

    def register_volume(self, config: VolumeConfig) -> "VolumeImporter":
        self._volumes[config.volume_id] = config
        self._status[config.volume_id] = VolumeStatus.NOT_ATTEMPTED
        self._functions[config.volume_id] = {}
        return self

    def register_volumes(self, configs: List[VolumeConfig]) -> "VolumeImporter":
        for config in configs:
            self.register_volume(config)
        return self

    def import_volume(self, volume_id: str) -> bool:
        if volume_id not in self._volumes:
            raise KeyError(f"Volume '{volume_id}' not registered.")

        config = self._volumes[volume_id]
        self._status[volume_id] = VolumeStatus.IMPORTING

        try:
            logger.info(f"Importing {volume_id} from {config.module_path}")
            module = importlib.import_module(config.module_path)
            self._modules[volume_id] = module

            available_count = 0
            for func_spec in config.functions:
                func = getattr(module, func_spec.name, None)
                if func is not None:
                    self._functions[volume_id][func_spec.local_name] = func
                    available_count += 1
                elif func_spec.required:
                    raise AttributeError(
                        f"Required function '{func_spec.name}' not found in {config.module_path}"
                    )
                else:
                    self._functions[volume_id][func_spec.local_name] = func_spec.default

            total_required = sum(1 for f in config.functions if f.required)

            # Distinguish clean AVAILABLE vs PARTIAL based on required vs optional
            if available_count == len(config.functions):
                self._status[volume_id] = VolumeStatus.AVAILABLE
            elif available_count >= total_required:
                self._status[volume_id] = VolumeStatus.PARTIAL
            else:
                raise VolumeImportError(
                    volume_id,
                    f"Only {available_count}/{len(config.functions)} functions available",
                )

            if config.post_import_hook:
                if not config.post_import_hook(self._functions[volume_id]):
                    raise VolumeImportError(volume_id, "Post-import validation failed")

            logger.info(
                f"✓ {volume_id} imported successfully ({self._status[volume_id].name})"
            )
            return True

        except ImportError as e:
            self._status[volume_id] = VolumeStatus.MISSING
            self._errors[volume_id] = f"Module not found: {config.module_path}"
            if config.optional:
                logger.warning(f"⚠ {volume_id} optional dependency missing: {e}")
                return True
            logger.error(f"✗ {volume_id} import failed: {e}")
            raise VolumeImportError(volume_id, str(e), e) from e

        except (AttributeError, VolumeImportError) as e:
            # AttributeError usually means missing required function
            total_required = sum(1 for f in config.functions if f.required)
            have_required = sum(
                1
                for f in config.functions
                if f.required and self._functions[volume_id].get(f.local_name) is not None
            )
            if config.optional and have_required == total_required:
                # All required functions are present: treat as PARTIAL, not ERROR
                self._status[volume_id] = VolumeStatus.PARTIAL
                self._errors[volume_id] = str(e)
                logger.warning(f"⚠ {volume_id} partial import: {e}")
                return True

            self._status[volume_id] = VolumeStatus.ERROR
            self._errors[volume_id] = str(e)
            if config.optional:
                logger.warning(f"⚠ {volume_id} import error (optional volume): {e}")
                return True
            logger.error(f"✗ {volume_id} import error: {e}")
            raise

        except Exception as e:
            self._status[volume_id] = VolumeStatus.ERROR
            self._errors[volume_id] = f"Unexpected error: {type(e).__name__}: {e}"
            if config.optional:
                logger.warning(f"⚠ {volume_id} unexpected error: {e}")
                return True
            logger.error(f"✗ {volume_id} unexpected error: {e}")
            raise VolumeImportError(volume_id, str(e), e) from e

    def import_all(self, raise_on_missing: bool = False) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for volume_id in self._volumes:
            try:
                results[volume_id] = self.import_volume(volume_id)
            except VolumeImportError:
                results[volume_id] = False
                if raise_on_missing and not self._volumes[volume_id].optional:
                    raise
        return results

    def is_available(self, volume_id: str) -> bool:
        # Treat PARTIAL as available so hooks can decide what to do
        return self._status.get(volume_id) in (
            VolumeStatus.AVAILABLE,
            VolumeStatus.PARTIAL,
        )

    def get_status(self, volume_id: str) -> VolumeStatus:
        return self._status.get(volume_id, VolumeStatus.NOT_ATTEMPTED)

    def get_function(self, volume_id: str, function_name: str, default: Any = None) -> Any:
        return self._functions.get(volume_id, {}).get(function_name, default)

    def get_module(self, volume_id: str) -> Optional[Any]:
        return self._modules.get(volume_id)

    def get_error(self, volume_id: str) -> Optional[str]:
        return self._errors.get(volume_id)

    def summary(self) -> str:
        lines = ["\n" + "=" * 60, "VOLUME IMPORT SUMMARY", "=" * 60]
        for vid, config in self._volumes.items():
            status = self._status.get(vid, VolumeStatus.NOT_ATTEMPTED)
            symbol = {
                VolumeStatus.AVAILABLE: "✓",
                VolumeStatus.PARTIAL: "⚠",
                VolumeStatus.MISSING: "○",
                VolumeStatus.ERROR: "✗",
            }.get(status, "?")
            line = f"{symbol} {vid:20s} [{status.name:12s}]"
            if status == VolumeStatus.ERROR and vid in self._errors:
                line += f" — {self._errors[vid]}"
            lines.append(line)
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    def require_functions(
        self,
        volume_id: str,
        *function_names: str,
        raise_if_missing: bool = True,
    ) -> Dict[str, Callable]:
        if not self.is_available(volume_id):
            if raise_if_missing:
                raise VolumeImportError(volume_id, "Volume not available")
            return {}

        result: Dict[str, Callable] = {}
        missing: List[str] = []
        for name in function_names:
            func = self.get_function(volume_id, name)
            if func is not None:
                result[name] = func
            else:
                missing.append(name)

        if missing and raise_if_missing:
            raise VolumeImportError(volume_id, f"Missing required functions: {missing}")

        return result