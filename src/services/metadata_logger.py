"""Metadata logging for generated images."""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MetadataLogger:
    """Log metadata for generated images."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metadata logger.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_generation(
        self,
        output_path: str,
        prompt: str,
        pipeline_type: str,
        parameters: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metadata for a generated image.

        Args:
            output_path: Path to the generated image
            prompt: The prompt used
            pipeline_type: Type of pipeline used
            parameters: Generation parameters
            additional_metadata: Any additional metadata
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "output_path": output_path,
            "prompt": prompt,
            "pipeline_type": pipeline_type,
            "parameters": parameters,
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        self._write_metadata(metadata)
        logger.info(f"Logged metadata for {output_path}")

    def _write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write metadata to log file."""
        try:
            log_file = self.log_dir / "generation_log.jsonl"

            with open(log_file, "a") as f:
                f.write(json.dumps(metadata) + "\n")

        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")

    def log_error(
        self,
        error_message: str,
        pipeline_type: str,
        parameters: Dict[str, Any],
    ) -> None:
        """Log an error during generation."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_message": error_message,
            "pipeline_type": pipeline_type,
            "parameters": parameters,
        }

        self._write_metadata(metadata)
        logger.error(f"Logged error: {error_message}")

    def get_generation_log(self) -> list:
        """
        Read all generation logs.

        Returns:
            List of metadata dictionaries
        """
        log_file = self.log_dir / "generation_log.jsonl"

        if not log_file.exists():
            return []

        logs = []
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read generation log: {e}")

        return logs
