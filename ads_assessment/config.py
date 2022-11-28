from typing import Optional, TypedDict
import dotenv
import os


def get_env(key: str) -> str:
    val = os.getenv(key)
    assert val is not None, f"Please make sure to set the {key} environment variable, or put it in .env"
    return val


class Credentials(TypedDict):
    """Helper class to make the type annotations clearer."""
    username: str
    password: str
    host: str
    database_name: str


creds: Credentials


def load_config(env_file: Optional[str] = None) -> None:
    if env_file is not None:
        dotenv.load_dotenv(dotenv_path=env_file)
    global creds
    creds = {
        "username": get_env("ADS_DB_USERNAME"),
        "password": get_env("ADS_DB_PASSWORD"),
        "host": get_env("ADS_DB_HOST"),
        "database_name": "property_prices"
    }
