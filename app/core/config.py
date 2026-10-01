from pydantic_settings import BaseSettings, SettingsConfigDict

# creating a configuration class, with name Settings
class Settings(BaseSettings):
    PROJECT_NAME: str = "Real Estate Housing Predict"
    MODEL_VERSION: str = "v1-regression"
    ENVIRONMENT: str = "development"

    HUGGINGFACE_API_KEY: str = "Null"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# create an instance of the class, which shall be exported and 
# reused by other source files
settings = Settings()
