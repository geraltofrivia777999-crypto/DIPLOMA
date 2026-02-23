import os
import sys

# Load .env BEFORE anything else so DB_BACKEND / POSTGRES_DSN are available
from dotenv import load_dotenv
load_dotenv()

# Fix SSL certificate verification on macOS
if sys.platform == "darwin":
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    except ImportError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

from anti_terror.runner import main


if __name__ == "__main__":
    main()
