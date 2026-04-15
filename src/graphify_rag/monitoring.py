from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter


@dataclass(slots=True)
class MetricsRegistry:
    request_count: int = 0
    ingest_count: int = 0
    question_count: int = 0
    request_latency_seconds: list[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def record_request(self, duration_seconds: float) -> None:
        with self._lock:
            self.request_count += 1
            self.request_latency_seconds.append(duration_seconds)

    def record_ingest(self) -> None:
        with self._lock:
            self.ingest_count += 1

    def record_question(self) -> None:
        with self._lock:
            self.question_count += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            count = len(self.request_latency_seconds)
            average = sum(self.request_latency_seconds) / count if count else 0.0
            return {
                "requests_total": self.request_count,
                "ingestions_total": self.ingest_count,
                "questions_total": self.question_count,
                "avg_request_latency_seconds": round(average, 4),
            }


class Timer:
    def __enter__(self) -> "Timer":
        self.started_at = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        self.elapsed = perf_counter() - self.started_at
