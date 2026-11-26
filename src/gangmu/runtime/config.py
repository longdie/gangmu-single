"""Strict, offline configuration loading and credential-free route snapshots.

Loading configuration never imports a provider SDK or invokes a model. Provider
protocols are labels: a configured protocol is unavailable until an execution
adapter explicitly declares that it implements the protocol. There is no
implicit provider failover, model substitution, or environment-file loading.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
import tomllib
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REDACTED = "[REDACTED]"
_ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TOP_FIELDS = frozenset({"config_version", "providers", "models", "parameters"})
_PROVIDER_FIELDS = frozenset({"api_key", "api_key_env", "base_url", "protocol"})
_MODEL_FIELDS = frozenset({"provider", "name"})
_SENSITIVE = frozenset({
    "apikey", "key", "authorization", "proxyauthorization", "auth",
    "bearer", "token", "accesstoken", "refreshtoken", "idtoken",
    "password", "passwd", "secret", "clientsecret", "secretkey",
    "privatekey", "credential", "credentials", "cookie", "setcookie",
    "signature", "xamzcredential", "xamzsecuritytoken", "xamzsignature",
    "xgoogsignature",
})


class ConfigError(ValueError):
    """Configuration is invalid; messages deliberately omit supplied values."""


class UnavailableRouteError(ConfigError):
    """A requested route is unconfigured or has no declared implementation."""


def _sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in _SENSITIVE or normalized.endswith((
        "apikey", "password", "clientsecret", "accesstoken", "refreshtoken",
        "credential", "credentials", "privatekey",
    ))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def redact_config(value: Any, *, secrets: Iterable[str] = ()) -> Any:
    """Return a JSON-compatible copy with nested credentials and URLs redacted.

    Values of known resolved credentials are also removed wherever they occur.
    A snapshot should still contain only explicitly nonsecret parameters; this
    helper is defense in depth, rather than a classifier for arbitrary secrets.
    """
    secret_values = sorted({item for item in secrets if item}, key=len, reverse=True)

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): REDACTED if _sensitive_key(str(key)) else visit(child)
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        if isinstance(item, str):
            result = item
            for secret in secret_values:
                result = result.replace(secret, REDACTED)
            # Also sanitize credentials in URL-valued parameters, including URLs
            # outside providers. The loader rejects them in provider base URLs.
            if result.strip().lower().startswith(("http://", "https://")):
                try:
                    parts = urlsplit(result.strip())
                    netloc = parts.netloc.rsplit("@", 1)[-1]
                    if "@" in parts.netloc:
                        netloc = f"{REDACTED}@{netloc}"
                    query = urlencode([
                        (key, REDACTED if _sensitive_key(key) else visit(child))
                        for key, child in parse_qsl(parts.query, keep_blank_values=True)
                    ])
                    result = urlunsplit((parts.scheme, netloc, parts.path, query,
                                        REDACTED if parts.fragment else ""))
                except ValueError:
                    return REDACTED
            return result
        return item

    return visit(value)


@dataclass(frozen=True)
class ProviderConfig:
    alias: str
    base_url: str
    protocol: str
    api_key_env: str = ""


@dataclass(frozen=True)
class ModelConfig:
    role: str
    provider: str
    name: str


@dataclass(frozen=True)
class ResolvedRoute:
    requested_role: str
    configured_role: str
    provider: ProviderConfig
    model: ModelConfig

    @property
    def fallback_used(self) -> bool:
        return self.requested_role != self.configured_role


@dataclass(frozen=True)
class RouteStatus:
    role: str
    status: str
    missing: tuple[str, ...] = ()


class Credentials:
    """Secrets isolated from routing configuration and safe object reprs.

    ``get`` is the only ordinary accessor for a resolved key. Callers must not
    log or serialize its result. Existing instances do not observe subsequent
    environment changes: reload explicitly for a later run.
    """

    __slots__ = ("__values", "__sources")

    def __init__(self, values: Mapping[str, str], sources: Mapping[str, str]) -> None:
        self.__values = MappingProxyType(dict(values))
        self.__sources = MappingProxyType(dict(sources))

    def __repr__(self) -> str:
        return "Credentials(<redacted>)"

    def get(self, provider: str) -> str:
        if provider not in self.__values:
            raise ConfigError("Unknown credential provider")
        return self.__values[provider]

    def configured(self, provider: str) -> bool:
        return bool(self.get(provider).strip())

    def source(self, provider: str) -> str:
        if provider not in self.__sources:
            raise ConfigError("Unknown credential provider")
        return self.__sources[provider]

    def redact(self, value: Any) -> Any:
        return redact_config(value, secrets=self.__values.values())


@dataclass(frozen=True)
class RuntimeConfig:
    config_version: int
    providers: Mapping[str, ProviderConfig]
    models: Mapping[str, ModelConfig]
    parameters: Mapping[str, Any]
    credentials: Credentials = field(repr=False, compare=False)
    project_root: Path = field(repr=False)
    sources: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return (f"RuntimeConfig(config_version={self.config_version}, "
                f"provider_count={len(self.providers)}, role_count={len(self.models)})")

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve an explicitly selected runtime path inside this project."""
        return _local_path(self.project_root, value)

    def resolve_route(self, role: str, *, fallback_to_default: bool = False) -> ResolvedRoute:
        """Find an explicit task/role route, optionally using its named default."""
        _alias(role)
        configured_role = role
        if role not in self.models:
            if fallback_to_default and "default" in self.models:
                configured_role = "default"
            else:
                raise ConfigError("No model route is configured for the requested role")
        model = self.models[configured_role]
        return ResolvedRoute(role, configured_role, self.providers[model.provider], model)

    def route_status(self, role: str, *, implemented_protocols: Iterable[str] = (),
                     fallback_to_default: bool = False) -> RouteStatus:
        route = self.resolve_route(role, fallback_to_default=fallback_to_default)
        missing = tuple(name for name, present in (
            ("model", bool(route.model.name.strip())),
            ("base_url", bool(route.provider.base_url)),
            ("protocol", bool(route.provider.protocol)),
            ("credential", self.credentials.configured(route.provider.alias)),
        ) if not present)
        if missing:
            return RouteStatus(role, "unconfigured", missing)
        if route.provider.protocol not in frozenset(implemented_protocols):
            return RouteStatus(role, "not_implemented")
        return RouteStatus(role, "configured")

    def require_route(self, role: str, *, implemented_protocols: Iterable[str] = (),
                      fallback_to_default: bool = False) -> ResolvedRoute:
        status = self.route_status(role, implemented_protocols=implemented_protocols,
                                   fallback_to_default=fallback_to_default)
        if status.status != "configured":
            raise UnavailableRouteError(f"Requested model route is {status.status}")
        return self.resolve_route(role, fallback_to_default=fallback_to_default)

    def snapshot(self) -> dict[str, Any]:
        """Effective configuration without keys, private file contents, or root paths."""
        result = {
            "config_version": self.config_version,
            "providers": {
                alias: {
                    "base_url": provider.base_url,
                    "protocol": provider.protocol,
                    "api_key_env": provider.api_key_env,
                    "credential_configured": self.credentials.configured(alias),
                    "credential_source": self.credentials.source(alias),
                } for alias, provider in self.providers.items()
            },
            "models": {role: {"provider": model.provider, "name": model.name}
                       for role, model in self.models.items()},
            "parameters": _plain(self.parameters),
            "sources": list(self.sources),
        }
        return self.credentials.redact(result)


def _alias(value: Any) -> str:
    if not isinstance(value, str) or not _ALIAS.fullmatch(value):
        raise ConfigError("Provider and role names must be bounded identifier strings")
    return value


def _table(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ConfigError("Configuration sections must be string-keyed tables")
    return value


def _json_parameters(value: Any) -> None:
    if isinstance(value, Mapping):
        _table(value)
        for key, child in value.items():
            if _sensitive_key(key):
                if child not in ("", None):
                    raise ConfigError("Credentials are forbidden in public or runtime parameters")
            _json_parameters(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _json_parameters(child)
    elif isinstance(value, str):
        if value.strip().lower().startswith(("http://", "https://")):
            try:
                parts = urlsplit(value.strip())
                if (parts.username is not None or parts.password is not None
                        or parts.fragment or any(_sensitive_key(key) for key, _
                                                 in parse_qsl(parts.query))):
                    raise ConfigError("Credential-bearing URLs are forbidden in runtime parameters")
            except ValueError:
                raise ConfigError("Invalid URL in runtime parameters") from None
        return
    elif value is None or isinstance(value, (bool, int)):
        return
    elif isinstance(value, float) and math.isfinite(value):
        return
    else:
        raise ConfigError("Runtime parameters must contain finite JSON-compatible values")


def _validate_layer(layer: Mapping[str, Any], *, private: bool = False) -> None:
    _table(layer)
    if set(layer) - _TOP_FIELDS:
        raise ConfigError("Unknown top-level configuration section")
    if "config_version" in layer and (type(layer["config_version"]) is not int
                                      or layer["config_version"] != 1):
        raise ConfigError("Only config_version = 1 is supported")
    for section, fields in (("providers", _PROVIDER_FIELDS), ("models", _MODEL_FIELDS)):
        entries = _table(layer.get(section, {}))
        for alias, entry in entries.items():
            _alias(alias)
            _table(entry)
            if set(entry) - fields:
                raise ConfigError("Unknown field in provider or model route")
            if any(not isinstance(value, str) for value in entry.values()):
                raise ConfigError("Provider and model route fields must be strings")
            if section == "providers":
                if not private and entry.get("api_key", ""):
                    raise ConfigError("Actual api_key values are allowed only in private configuration")
                env_name = entry.get("api_key_env", "")
                if env_name and not _ENV_NAME.fullmatch(env_name):
                    raise ConfigError("api_key_env must name an environment variable")
                protocol = entry.get("protocol", "")
                if protocol and not _ALIAS.fullmatch(protocol):
                    raise ConfigError("Provider protocols must be bounded identifier strings")
                _base_url(entry.get("base_url", ""))
    parameters = _table(layer.get("parameters", {}))
    # Parameters never hold credentials, including in a selected private file.
    _json_parameters(parameters)


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = _plain(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = _plain(value)
    return result


def _local_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    path = (root / path if not path.is_absolute() else path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ConfigError("Configuration paths must remain inside the project root") from None
    return path


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        # Parser errors may quote a secret-containing source line.
        raise ConfigError("Configuration file is missing, unreadable, or invalid TOML") from None


def _base_url(value: str) -> str:
    if not value:
        return ""
    try:
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or any(char.isspace() for char in value)):
            raise ValueError
        # Accessing port also detects nonnumeric or out-of-range ports.
        parts.port
    except ValueError:
        raise ConfigError("Provider base_url must be an HTTP(S) address without credentials or query") from None
    return value.rstrip("/")


def load_config(project_root: str | Path, *,
                public_config: str | Path = "config.example.toml",
                public_defaults: Sequence[str | Path] = (),
                private_config: str | Path | None = "config.toml",
                experiment_config: str | Path | None = None,
                overrides: Mapping[str, Any] | None = None,
                environ: Mapping[str, str] | None = None,
                require_private: bool = False) -> RuntimeConfig:
    """Load one private file and freeze effective configuration, without calls.

    Precedence for nonsecret fields is template < public defaults < private
    configuration < experiment < explicit overrides. Credentials are selected
    from the one private file, then a nonempty declared environment value takes
    precedence. Public layers cannot supply an actual key or erase a private
    key with a placeholder. Relative paths are project-relative and cannot
    escape the project root, including through symlinks.
    """
    root = Path(project_root).resolve()
    if isinstance(public_defaults, (str, bytes, Path)):
        raise ConfigError("public_defaults must be a sequence of project-local paths")
    effective: dict[str, Any] = {}
    sources: list[str] = []
    private_keys: dict[str, str] = {}

    def add_file(value: str | Path, *, private: bool = False) -> None:
        nonlocal effective
        path = _local_path(root, value)
        layer = _read(path)
        _validate_layer(layer, private=private)
        layer = _plain(layer)
        for alias, entry in layer.get("providers", {}).items():
            if private:
                private_keys[alias] = entry.get("api_key", "")
            entry.pop("api_key", None)
        effective = _merge(effective, layer)
        sources.append("private" if private else path.relative_to(root).as_posix())

    add_file(public_config)
    for value in public_defaults:
        add_file(value)
    if private_config is not None:
        private_path = _local_path(root, private_config)
        if private_path.is_file() or require_private:
            add_file(private_path, private=True)
    elif require_private:
        raise ConfigError("A private configuration file is required")
    if experiment_config is not None:
        add_file(experiment_config)
    if overrides is not None:
        _validate_layer(overrides)
        clean_overrides = _plain(overrides)
        for entry in clean_overrides.get("providers", {}).values():
            entry.pop("api_key", None)
        effective = _merge(effective, clean_overrides)
        sources.append("explicit_overrides")
    if effective.get("config_version") != 1:
        raise ConfigError("The effective configuration must declare config_version = 1")

    providers: dict[str, ProviderConfig] = {}
    key_values: dict[str, str] = {}
    key_sources: dict[str, str] = {}
    environment = os.environ if environ is None else environ
    for alias, entry in effective.get("providers", {}).items():
        providers[alias] = ProviderConfig(alias=alias, base_url=_base_url(entry.get("base_url", "")),
                                          protocol=entry.get("protocol", ""),
                                          api_key_env=entry.get("api_key_env", ""))
        env_value = environment.get(providers[alias].api_key_env, "") if providers[alias].api_key_env else ""
        if not isinstance(env_value, str):
            raise ConfigError("Credential environment values must be strings")
        key_values[alias] = env_value if env_value.strip() else private_keys.get(alias, "")
        key_sources[alias] = ("environment" if env_value.strip() else
                              "private" if key_values[alias].strip() else "missing")
    models: dict[str, ModelConfig] = {}
    for role, entry in effective.get("models", {}).items():
        provider = entry.get("provider", "")
        if provider not in providers:
            raise ConfigError("A model route must refer to a declared provider")
        models[role] = ModelConfig(role, provider, entry.get("name", ""))
    return RuntimeConfig(1, MappingProxyType(providers), MappingProxyType(models),
                         _freeze(effective.get("parameters", {})),
                         Credentials(key_values, key_sources), root, tuple(sources))
