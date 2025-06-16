# coding=utf-8
"""
Functions to calculate the positions/velocities of exoplanet/exomoon.
Created on 18. October 2021 by Andrea Gebek.
"""

from typing import Tuple

import numpy as np


class Grid:
    def __init__(
        self,
        x_midpoint: float,
        x_border: float,
        x_steps: int,
        rho_border: float,
        rho_steps: int,
        phi_steps: int,
        orbphase_border: float,
        orbphase_steps: int
    ) -> None:
        self.x_midpoint = x_midpoint
        self.x_border = x_border
        self.x_steps = x_steps
        self.rho_border = rho_border
        self.rho_steps = rho_steps
        self.phi_steps = phi_steps
        self.orbphase_border = orbphase_border
        self.orbphase_steps = orbphase_steps

    @staticmethod
    def getCartesianFromCylinder(phi: float, rho: float) -> Tuple[float, float]:
        y = rho * np.sin(phi)
        z = rho * np.cos(phi)
        return y, z

    def getDeltaX(self) -> float:
        return 2. * self.x_border / float(self.x_steps)

    def getDeltaRho(self) -> float:
        return self.rho_border / float(self.rho_steps)

    def getDeltaPhi(self) -> float:
        return 2. * np.pi / float(self.phi_steps)

    def constructXaxis(self, midpoints: bool = True) -> np.ndarray:
        if midpoints:  # Gas cell midpoints
            x_axis = np.linspace(
                self.x_midpoint - self.x_border,
                self.x_midpoint + self.x_border,
                int(self.x_steps),
                endpoint=False
            ) + self.x_border / float(self.x_steps)
        else:  # Return an array with x_steps + 1 entries, marking the cell edges
            x_axis = np.linspace(
                self.x_midpoint - self.x_border,
                self.x_midpoint + self.x_border,
                int(self.x_steps) + 1
            )
        return x_axis

    def constructRhoAxis(self, midpoints: bool = True) -> np.ndarray:
        if midpoints:
            rho_axis = np.linspace(
                0., self.rho_border, int(self.rho_steps), endpoint=False
            ) + 0.5 * self.rho_border / float(self.rho_steps)
        else:
            rho_axis = np.linspace(
                0., self.rho_border, int(self.rho_steps) + 1
            )
        return rho_axis

    def constructPhiAxis(self, midpoints: bool = True) -> np.ndarray:
        if midpoints:
            phi_axis = np.linspace(
                0, 2 * np.pi, int(self.phi_steps), endpoint=False
            ) + np.pi / float(self.phi_steps)
        else:
            phi_axis = np.linspace(
                0, 2 * np.pi, int(self.phi_steps) + 1
            )
        return phi_axis

    def constructOrbphaseAxis(self) -> np.ndarray:
        orbphase_axis = np.linspace(
            -self.orbphase_border, self.orbphase_border, int(self.orbphase_steps)
        )
        return orbphase_axis

    def getChordGrid(self) -> np.ndarray:
        phi_axis = self.constructPhiAxis()
        rho_axis = self.constructRhoAxis()
        orbphase_axis = self.constructOrbphaseAxis()
        phiGrid, rhoGrid, orbphaseGrid = np.meshgrid(
            phi_axis, rho_axis, orbphase_axis, indexing='ij'
        )
        chordGrid = np.stack(
            (phiGrid.flatten(), rhoGrid.flatten(), orbphaseGrid.flatten()), axis=-1
        )
        return chordGrid
