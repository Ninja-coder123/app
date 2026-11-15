"""Shared data schemas for the wealth management application."""

from dataclasses import dataclass


@dataclass
class MarketAssumptions:
    """
    Market assumptions driving financial projections.

    Attributes:
        inflation_rate: Annual inflation rate (decimal, e.g., 0.06 for 6%).
        step_up_rate: Annual step-up rate for SIP contributions (decimal).
        equity_return: Expected annual equity return (decimal).
        debt_return: Expected annual debt return (decimal).
    """
    inflation_rate: float
    step_up_rate: float
    equity_return: float
    debt_return: float

    @classmethod
    def from_preset(cls, preset_name: str) -> "MarketAssumptions":
        """
        Create market assumptions from a preset configuration.

        Args:
            preset_name: One of "Conservative", "Normal", or "Strong".

        Returns:
            MarketAssumptions instance with preset values.

        Raises:
            ValueError: If preset_name is not recognized.
        """
        presets = {
            "Conservative": {
                "inflation_rate": 0.05,
                "step_up_rate": 0.05,
                "equity_return": 0.10,
                "debt_return": 0.06,
            },
            "Normal": {
                "inflation_rate": 0.06,
                "step_up_rate": 0.07,
                "equity_return": 0.12,
                "debt_return": 0.07,
            },
            "Strong": {
                "inflation_rate": 0.07,
                "step_up_rate": 0.10,
                "equity_return": 0.15,
                "debt_return": 0.08,
            },
        }

        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Choose from {list(presets.keys())}"
            )

        config = presets[preset_name]
        return cls(**config)


@dataclass
class Goal:
    """A financial goal specification."""

    target_amount: float
    years_to_goal: float
    is_critical: bool
    risk_capacity: str = "medium"


@dataclass
class PortfolioAllocation:
    """
    Portfolio asset allocation percentages.

    Attributes:
        equity_pct: Percentage allocated to equities (0-100).
        debt_pct: Percentage allocated to debt (0-100).
    """
    equity_pct: float
    debt_pct: float

    def __post_init__(self):
        """Validate that equity and debt percentages sum to 100."""
        total = self.equity_pct + self.debt_pct
        tolerance = 1e-6
        if abs(total - 100.0) > tolerance:
            raise ValueError(
                f"Allocation percentages must sum to 100.0, "
                f"got {total} ({self.equity_pct}% + {self.debt_pct}%)"
            )


@dataclass
class TaxConfig:
    """
    Taxation configuration for investment returns.

    Attributes:
        equity_ltcg_rate: Long-term capital gains tax rate for equities (default 0.125).
        equity_exemption: Annual tax-free exemption for equity gains (default 125000).
        debt_slab_rate: Tax slab rate for debt returns (default 0.30).
    """
    equity_ltcg_rate: float = 0.125
    equity_exemption: float = 125000.0
    debt_slab_rate: float = 0.30
