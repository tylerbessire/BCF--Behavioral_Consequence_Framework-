"""Network effects analyzer - models how behaviors spread through social networks."""

from typing import Dict, List, Optional

import numpy as np


class NetworkEffectsAnalyzer:
    """Analyzes how behaviors cascade through social networks."""

    def __init__(self, network_size: int = 100):
        """
        Initialize network effects analyzer.

        Args:
            network_size: Size of social network to model
        """
        self.network_size = network_size

    def estimate_social_contagion(
        self,
        initial_adopters: int = 1,
        transmission_rate: float = 0.3,
        recovery_rate: float = 0.1,
        time_steps: int = 50,
    ) -> Dict[str, List[float]]:
        """
        Estimate spread of behavior through network using SIR-like model.

        Args:
            initial_adopters: Number of initial adopters
            transmission_rate: Probability of transmission per contact
            recovery_rate: Rate of discontinuation
            time_steps: Number of time steps to simulate

        Returns:
            Dict with time series of adopters, susceptible, and recovered
        """
        susceptible = [self.network_size - initial_adopters]
        infected = [initial_adopters]
        recovered = [0]

        for _ in range(time_steps - 1):
            s = susceptible[-1]
            i = infected[-1]
            r = recovered[-1]

            # New infections
            new_infected = transmission_rate * (s * i) / self.network_size

            # New recoveries
            new_recovered = recovery_rate * i

            susceptible.append(max(0, s - new_infected))
            infected.append(max(0, i + new_infected - new_recovered))
            recovered.append(r + new_recovered)

        return {
            "susceptible": susceptible,
            "adopters": infected,
            "discontinued": recovered,
            "peak_adoption": max(infected),
            "peak_time": infected.index(max(infected)),
        }

    def calculate_network_amplification(
        self, direct_impact: float, network_density: float = 0.3, influence_decay: float = 0.5
    ) -> float:
        """
        Calculate how network effects amplify individual behavior impact.

        Args:
            direct_impact: Direct impact of individual behavior
            network_density: How connected the network is (0-1)
            influence_decay: How much influence decays with distance

        Returns:
            Amplified impact accounting for network effects
        """
        # Simple model: impact = direct + network_effect
        network_multiplier = 1 + (network_density * influence_decay)
        return direct_impact * network_multiplier
