"""Maintenance recommendations derived from model confidence and coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


def calculate_efficiency_loss(
    clean_power_w: float,
    dusty_power_w: float,
    clean_irradiance_w_m2: float,
    dusty_irradiance_w_m2: float,
) -> float:
    """Calculate loss using (eta_clean - eta_dusty) / eta_clean * 100.

    Efficiency is represented by irradiance-normalized power:
    ``eta = power_w / irradiance_w_m2``.
    """
    values = (clean_power_w, dusty_power_w, clean_irradiance_w_m2, dusty_irradiance_w_m2)
    if any(float(value) <= 0 for value in values):
        raise ValueError("Power and irradiance measurements must be greater than zero.")
    clean_normalized = float(clean_power_w) / float(clean_irradiance_w_m2)
    dusty_normalized = float(dusty_power_w) / float(dusty_irradiance_w_m2)
    loss = (1.0 - dusty_normalized / clean_normalized) * 100.0
    return round(max(0.0, min(100.0, loss)), 2)


@dataclass(frozen=True)
class DustAnalysis:
    confidence: float
    dust_coverage: float
    dust_level: str
    level_color: str
    efficiency_loss: float | None
    maintenance_action: str
    maintenance_priority: str
    next_cleaning_date: str
    recommendations: list[str]


def analyze_dust(
    confidence_percent: float, dust_coverage_percent: float, label: str
) -> DustAnalysis:
    """Return maintenance values without inventing electrical loss data."""
    confidence = round(max(0.0, min(100.0, float(confidence_percent))), 1)
    coverage = round(max(0.0, min(100.0, float(dust_coverage_percent))), 1)

    if label == "Clean":
        level = "Clean"
        color = "#22c55e"
        loss = None
        action = "No cleaning required"
        priority = "None"
        days = 30
        recommendations = [
            "Continue routine visual inspections.",
            "Keep the panel surface free from new obstructions.",
        ]
    elif coverage >= 35:
        level = "Heavy Dust"
        color = "#ef4444"
        loss = None
        action = "Immediate cleaning required"
        priority = "High"
        days = 1
        recommendations = [
            "Clean the panel surface as soon as safely possible.",
            "Check for persistent stains or physical damage after cleaning.",
        ]
    elif coverage >= 15:
        level = "Moderate Dust"
        color = "#f97316"
        loss = None
        action = "Cleaning required within 7 days"
        priority = "Medium"
        days = 7
        recommendations = [
            "Schedule a cleaning within the next week.",
            "Inspect the panel again after the next dusty-weather event.",
        ]
    else:
        level = "Light Dust"
        color = "#eab308"
        loss = None
        action = "Routine cleaning recommended"
        priority = "Low"
        days = 14
        recommendations = [
            "Plan a routine cleaning during the next maintenance window.",
            "Monitor coverage over time for a worsening trend.",
        ]

    next_date = (date.today() + timedelta(days=days)).strftime("%d %B %Y")
    return DustAnalysis(
        confidence=confidence,
        dust_coverage=coverage,
        dust_level=level,
        level_color=color,
        efficiency_loss=loss,
        maintenance_action=action,
        maintenance_priority=priority,
        next_cleaning_date=next_date,
        recommendations=recommendations,
    )
