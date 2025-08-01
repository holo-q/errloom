"""
API key management for different providers.
"""
import os
from pathlib import Path

from errloom.paths import userdir


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment or user config."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    # Check user config file
    config_file = userdir / "userconf.py"
    if config_file.exists():
        # Try to read from config file
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                # Simple parsing for OPENAI_API_KEY
                if 'OPENAI_API_KEY' in content:
                    for line in content.split('\n'):
                        if 'OPENAI_API_KEY' in line and '=' in line:
                            key = line.split('=')[1].strip().strip('"').strip("'")
                            if key:
                                return key
        except:
            pass
    
    raise ValueError("OPENAI_API_KEY not found in environment or user config")


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment or user config."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        return api_key
    
    # Check user config file
    config_file = userdir / "userconf.py"
    if config_file.exists():
        # Try to read from config file
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                # Simple parsing for OPENROUTER_API_KEY
                if 'OPENROUTER_API_KEY' in content:
                    for line in content.split('\n'):
                        if 'OPENROUTER_API_KEY' in line and '=' in line:
                            key = line.split('=')[1].strip().strip('"').strip("'")
                            if key:
                                return key
        except:
            pass
    
    raise ValueError("OPENROUTER_API_KEY not found in environment or user config")


def prompt_and_store_api_key(provider: str, env_var: str) -> str:
    """
    Prompt user for API key and store it in user config.
    
    Args:
        provider: Name of the provider (e.g., "OpenAI", "OpenRouter")
        env_var: Environment variable name for the key
        
    Returns:
        The API key
    """
    import getpass
    
    print(f"Please enter your {provider} API key (or set {env_var} environment variable):")
    api_key = getpass.getpass(f"{provider} API Key: ")
    
    if not api_key:
        raise ValueError(f"{provider} API key is required")
    
    # Store in user config file
    config_file = userdir / "userconf.py"
    config_content = ""
    
    # Read existing config if it exists
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_content = f.read()
    
    # Add or update the API key
    key_line = f'{env_var} = "{api_key}"\n'
    if env_var in config_content:
        # Replace existing key
        import re
        config_content = re.sub(f'{env_var} = ".*"', key_line.strip(), config_content)
    else:
        # Add new key
        config_content += key_line
    
    # Write config file
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    # Also set in environment for current session
    os.environ[env_var] = api_key
    
    return api_key


def get_or_prompt_openai_api_key() -> str:
    """Get OpenAI API key or prompt user for it."""
    try:
        return get_openai_api_key()
    except ValueError:
        return prompt_and_store_api_key("OpenAI", "OPENAI_API_KEY")


def get_or_prompt_openrouter_api_key() -> str:
    """Get OpenRouter API key or prompt user for it."""
    try:
        return get_openrouter_api_key()
    except ValueError:
        return prompt_and_store_api_key("OpenRouter", "OPENROUTER_API_KEY")