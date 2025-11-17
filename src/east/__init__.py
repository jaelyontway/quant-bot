"""
EAST - Event-Driven AI Sentiment Trigger Model

A modular system for detecting news events, analyzing sentiment,
and generating trading signals based on event-driven strategies.
"""

__version__ = "0.1.0"
__author__ = "EAST Development Team"

from . import sentiment
from . import margin_predictor
from . import train_margins
from . import integrate_margins

__all__ = [
    "sentiment",
    "margin_predictor",
    "train_margins",
    "integrate_margins",
]

