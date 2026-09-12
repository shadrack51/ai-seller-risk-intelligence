import os

from dotenv import load_dotenv


load_dotenv()


def test_environment_variable():
    app_environment = os.getenv(
        "APP_ENV",
        "development",
    )

    print(f"Application environment: {app_environment}")


if __name__ == "__main__":
    test_environment_variable()