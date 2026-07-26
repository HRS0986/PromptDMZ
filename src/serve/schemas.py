"""Pydantic request/response models for the demonstration server.

    DetectRequest   {"prompt": "<text>"}
    DetectResponse  {"verdict": "INJECTION" | "BENIGN",
                     "score": float, "threshold": float,
                     "category_scores": {"role_violation": float,
                                         "privilege_escalation": float,
                                         "obfuscation_evasion": float},
                     "latency_ms": float}

Implemented by P9.2.
"""
