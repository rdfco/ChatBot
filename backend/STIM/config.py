import sys
from pathlib import Path

from pydantic_settings import BaseSettings

from STIM.utils.appdirs import user_data_dir

# https://pyinstaller.org/en/v6.6.0/runtime-information.html
IS_BUNDLED = bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))

USER_DATA_DIR = user_data_dir(appname="STIM")


class EnvironmentType(str):
    development = "development"
    production = "production"


class Config(BaseSettings):
    # SQLite database will be mounted in the configuration directory
    # This is where all STIM data is stored
    # Current dir / db.sqlite3
    sqlite_path: str = str(Path(USER_DATA_DIR) / "db.sqlite3")
    sqlite_echo: bool = False

    # This is where all uploaded files are stored (ex. uploaded sqlite DBs)
    data_directory: str = str(Path(USER_DATA_DIR) / "data")

    sample_dvdrental_path: str = str(Path(__file__).parent.parent / "samples" / "dvd_rental.sqlite3")
    sample_netflix_path: str = str(Path(__file__).parent.parent / "samples" / "netflix.sqlite3")
    sample_titanic_path: str = str(Path(__file__).parent.parent / "samples" / "titanic.sqlite3")
    sample_spotify_path: str = str(Path(__file__).parent.parent / "samples" / "spotify.sqlite3")

    # to be consumed by the bundled frontend app
    base_api_url: str = ""  # e.g. http://localhost:7377

    default_model: str = "gpt-3.5-turbo"
    templates_path: Path = Path(__file__).parent.parent / "templates"
    assets_path: Path = Path(__file__).parent.parent / "assets"

    environment: str = EnvironmentType.development if not IS_BUNDLED else EnvironmentType.production
    release: str | None = None

    # HTTP Basic Authentication
    auth_username: str | None = None
    auth_password: str | None = None

    spa_mode: bool = False

    # CORS settings
    allowed_origins: str = (
        "http://localhost:7377,http://localhost:5173,http://0.0.0.0:7377,http://0.0.0.0:5173,http://127.0.0.1:7377,http://127.0.0.1:5173"  # comma separated list of origins
    )

    @property
    def has_auth(self) -> bool:
        return bool(self.auth_username and self.auth_password)


config = Config()
