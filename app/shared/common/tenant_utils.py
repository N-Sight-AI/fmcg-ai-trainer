import os
import json
from typing import Optional
from dataclasses import dataclass

@dataclass
class TenantConfig:
    db_server: str
    db_port: int
    db_name: str
    db_driver: str
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_trusted_connection: bool = False
    db_encrypt: bool = True
    db_trust_server_cert: bool = False

def _load_config() -> dict:
    """Load configuration from config.json file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def get_tenant_config_from_cache_only(tenant: str) -> Optional[TenantConfig]:
    """Get tenant configuration from config.json file."""
    config = _load_config()
    tenant_key = tenant.upper()
    
    if 'tenants' not in config or tenant_key not in config['tenants']:
        return None
    
    tenant_config = config['tenants'][tenant_key]
    
    return TenantConfig(
        db_server=tenant_config.get('db_server', ''),
        db_port=int(tenant_config.get('db_port', 1433)),
        db_name=tenant_config.get('db_name', ''),
        db_driver=tenant_config.get('db_driver', 'ODBC Driver 17 for SQL Server'),
        db_user=tenant_config.get('db_user'),
        db_password=tenant_config.get('db_password'),
        db_trusted_connection=tenant_config.get('db_trusted_connection', False),
        db_encrypt=tenant_config.get('db_encrypt', True),
        db_trust_server_cert=tenant_config.get('db_trust_server_cert', False),
    )

def get_logging_config(profile: str = "default") -> dict:
    """Get logging configuration from config.json file."""
    config = _load_config()
    
    if 'logging' not in config:
        return {"level": "INFO", "format": "json"}
    
    logging_config = config['logging']
    
    # Return specific profile or default
    if profile in logging_config:
        return logging_config[profile]
    elif 'default' in logging_config:
        return logging_config['default']
    else:
        return {"level": "INFO", "format": "json"}

def get_tenant_config_sync(tenant: str) -> Optional[TenantConfig]:
    return get_tenant_config_from_cache_only(tenant)
