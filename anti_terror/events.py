import datetime as dt
from pathlib import Path
from typing import Iterable

from loguru import logger

from .config import EventConfig


class EventSink:
    def __init__(self, cfg: EventConfig):
        self.cfg = cfg
        if cfg.enable_file_logging:
            cfg.log_dir.mkdir(parents=True, exist_ok=True)
            logger.add(cfg.log_dir / "events.log", rotation="10 MB")

    def emit(self, events: Iterable[dict]):
        for event in events:
            timestamp = event.get("time") or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            camera = event.get("camera_id") or self.cfg.camera_id
            event_type = event.get("type", "unknown")

            payload = {
                "timestamp": timestamp,
                "camera": camera,
                **event,
            }

            if event_type == "Abandoned Bag":
                logger.opt(colors=True).warning(
                    f"<red>[WARNING]</red> {timestamp} | "
                    f"ABANDONED BAG: {event.get('bag_id','?')} | "
                    f"Camera: {camera} | "
                    f"Owner: {event.get('person_id','?')} left {event.get('away_for_s',0)}s ago | "
                    f"Static: {event.get('static_for_s',0)}s"
                )
            elif event_type == "Bag Ownership Transfer":
                logger.opt(colors=True).warning(
                    f"<yellow>[WARNING]</yellow> {timestamp} | "
                    f"BAG TRANSFER: {event.get('bag_id','?')} | "
                    f"Camera: {camera} | "
                    f"{event.get('previous_owner','?')} -> {event.get('person_id','?')}"
                )
            else:
                logger.info(
                    f"[ALERT] {timestamp} | {event_type} "
                    f"PersonID: {event.get('person_id','?')} BagID: {event.get('bag_id','?')} "
                    f"Camera: {camera} details={payload}"
                )
