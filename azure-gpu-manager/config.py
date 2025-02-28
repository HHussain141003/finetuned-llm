import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Azure Configuration
    SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
    RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
    VM_NAME = os.getenv("AZURE_VM_NAME")
    TENANT_ID = os.getenv("AZURE_TENANT_ID")
    CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
    CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
    
    # Application Configuration
    LLM_SERVICE_PORT = int(os.getenv("LLM_SERVICE_PORT", "8000"))
    IDLE_TIMEOUT_MINUTES = int(os.getenv("IDLE_TIMEOUT_MINUTES", "30"))
    
    # Derived values
    @classmethod
    def get_vm_url(cls, ip_address):
        return f"http://{ip_address}:{cls.LLM_SERVICE_PORT}"
