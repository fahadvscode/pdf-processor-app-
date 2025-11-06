"""
File tracking module for PDF redaction system.
"""

import logging

logger = logging.getLogger(__name__)

class FileTracker:
    """
    Simple file tracker for monitoring processed files.
    """
    
    def __init__(self):
        self.processed_files = set()
        logger.info("FileTracker initialized")
    
    def is_processed(self, file_id: str) -> bool:
        """Check if a file has been processed."""
        return file_id in self.processed_files
    
    def mark_processed(self, file_id: str):
        """Mark a file as processed."""
        self.processed_files.add(file_id)
        logger.debug(f"Marked file {file_id} as processed")
    
    def get_processed_count(self) -> int:
        """Get the number of processed files."""
        return len(self.processed_files)
