"""
Centralized logging configuration for the server using RichHandler.
"""
import logging
from rich.logging import RichHandler
from rich.console import Console

# --- Module Level Variables ---
_handler_configured = False
_console = Console() # Shared console instance

# --- Custom Filter ---
class ModuleBlockerFilter(logging.Filter):
    """Blocks logs from specified modules when active."""
    def __init__(self, name=''):
        super().__init__(name)
        self.blocked_prefixes = []
        self.blocking_active = False # Default to not blocking

    def set_blocked_prefixes(self, prefixes: list[str]):
        self.blocked_prefixes = prefixes

    def set_blocking(self, active: bool):
        self.blocking_active = active

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter implementation. Returns False to block, True to allow.
        """
        if self.blocking_active:
            for prefix in self.blocked_prefixes:
                if record.name.startswith(prefix):
                    return False # Block the record
        return True # Allow the record by default or if blocking is inactive

# Create a single instance of the filter
_module_blocker = ModuleBlockerFilter()

# --- Setup Function ---
def setup_logging(debug_mode: bool = False):
    """
    Configures the root logger with RichHandler.

    Ensures configuration happens only once.

    Args:
        debug_mode: If True, sets level to DEBUG and enables richer tracebacks/paths.
                    Otherwise, sets level to CRITICAL.
    """
    global _handler_configured
    if _handler_configured:
        logging.debug("Logger already configured by RichHandler.")
        return

    log_level = logging.DEBUG if debug_mode else logging.CRITICAL
    root_logger = logging.getLogger() # Get root logger

    # Remove any pre-existing handlers (e.g., basicConfig from imports)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create and configure RichHandler
    rich_handler = RichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=debug_mode, # Show path only in debug mode
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=debug_mode # Show locals in tracebacks only in debug
    )

    # Add the custom filter to the handler
    rich_handler.addFilter(_module_blocker)

    # Configure root logger with RichHandler
    root_logger.setLevel(log_level)
    root_logger.addHandler(rich_handler)
    root_logger.propagate = False # Prevent propagation to avoid duplicates

    # Specifically prevent Werkzeug logs from propagating to our handler
    logging.getLogger("werkzeug").propagate = False

    _handler_configured = True

def get_console() -> Console:
    """Returns the shared Rich Console instance."""
    return _console

def get_module_blocker_filter() -> ModuleBlockerFilter:
    """Returns the shared module blocking filter instance."""
    return _module_blocker

# Ensure logging is configured at least minimally if module is imported elsewhere early
# This is less ideal than calling setup_logging explicitly, but acts as a safeguard.
# if not _handler_configured:
#     setup_logging() # Call with default (INFO) if not configured yet 