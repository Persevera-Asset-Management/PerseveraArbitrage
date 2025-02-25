import logging

# Configure logging for the entire package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger for the package
logger = logging.getLogger('persevera_arbitrage')

# Set default level
logger.setLevel(logging.INFO)

# Create a null handler to avoid "No handler found" warnings
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__version__ = "0.5.11"