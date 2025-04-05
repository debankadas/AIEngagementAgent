import logging
import logging.handlers
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

# --- Constants ---
LOG_DIR = Path(__file__).resolve().parent.parent / "logs" # Place logs dir in project root
LOG_FILENAME_PREFIX = "app"
# LOG_FILE will be determined dynamically in setup_logging
LOG_MAX_BYTES = 5 * 1024 * 1024 # 5 MB
LOG_BACKUP_COUNT = 5 # Keep 5 backup files (app.log.1, app.log.2, ...)
LOG_RETENTION_DAYS = 2 # Delete logs older than 2 days

# --- Setup Function ---
def setup_logging(use_utc=False): # Add option for UTC time
    """Configures logging for the application with date-stamped filenames."""
    LOG_DIR.mkdir(exist_ok=True) # Create logs directory if it doesn't exist

    # Determine current log filename based on date
    now = datetime.utcnow() if use_utc else datetime.now()
    log_filename = f"{LOG_FILENAME_PREFIX}_{now.strftime('%Y-%m-%d')}.log"
    current_log_file_path = LOG_DIR / log_filename

    # Configure the root logger
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if use_utc:
        log_formatter.converter = time.gmtime # Use UTC time for logs


    # Rotating File Handler (based on size for the *current day's* log file)
    rotating_handler = logging.handlers.RotatingFileHandler(
        filename=current_log_file_path, # Use the date-stamped filename
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    rotating_handler.setFormatter(log_formatter)

    # Console Handler (optional, but useful for seeing logs during development)
    console_handler = logging.StreamHandler(sys.stdout) # Use stdout for console
    console_handler.setFormatter(log_formatter)

    # Get the root logger and add handlers
    logger = logging.getLogger() # Get root logger
    # Clear existing handlers (important if this function is called multiple times or in tests)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO) # Set minimum logging level
    logger.addHandler(rotating_handler)
    logger.addHandler(console_handler) # Add console output

    logging.info("Logging configured successfully.") # Log confirmation
    return logger # Return the configured logger instance

# --- Cleanup Function for Old Logs ---
def cleanup_old_logs():
    """Deletes log files older than LOG_RETENTION_DAYS."""
    now_ts = time.time()
    cutoff_ts = now_ts - (LOG_RETENTION_DAYS * 86400) # 86400 seconds in a day
    logging.info(f"Running log cleanup. Deleting files in {LOG_DIR} older than {LOG_RETENTION_DAYS} days...")
    deleted_count = 0
    try:
        if not LOG_DIR.exists():
            logging.warning(f"Log directory {LOG_DIR} not found. Skipping cleanup.")
            return

        # Glob pattern to match app_YYYY-MM-DD.log and app_YYYY-MM-DD.log.1 etc.
        for file_path in LOG_DIR.glob(f"{LOG_FILENAME_PREFIX}_*.log*"):
            if file_path.is_file():
                try:
                    file_mod_time = file_path.stat().st_mtime
                    if file_mod_time < cutoff_ts:
                        file_path.unlink()
                        logging.info(f"Deleted old log file: {file_path}")
                        deleted_count += 1
                except OSError as e:
                    logging.error(f"Error deleting log file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error during log cleanup process: {e}")
    logging.info(f"Log cleanup finished. Deleted {deleted_count} file(s).")