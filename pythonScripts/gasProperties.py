
"""
This file defines classes and functions related to the properties of the
gaseous medium in an exoplanet's atmosphere or exosphere. It includes
various models for number density distributions (e.g., barometric, power-law),
classes for atomic and molecular absorbers, and the main `Transit` class that
orchestrates the simulation of a transit light curve.

Created on 19. October 2021 by Andrea Gebek.
"""

import os
from copy import deepcopy
from typing import Any, Callable, List, Tuple, Union

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter as gauss
from scipy.special import erf, voigt_profile

from . import constants as const
from . import geometryHandler as geom

lineListPath: str = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + '/Resources/LineList.txt'
molecularLookupPath: str = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))) + '/molecularResources/'


class CollisionalAtmosphere:
    """Base class for a collisional atmosphere with a defined temperature and pressure.

    Attributes:
        T (float): Temperature of the atmosphere in Kelvin.
        P_0 (float): Reference pressure at the base of the atmosphere in cgs units.
        constituents (List[Union['AtmosphericConstituent', 'MolecularConstituent']]):
            List of absorbing species in the atmosphere.
        hasMoon (bool): Flag indicating if a moon is present in this model.
    """

    def __init__(self, T: float, P_0: float):
        """Initializes the CollisionalAtmosphere.

        Args:
            T (float): Temperature of the atmosphere in Kelvin.
            P_0 (float): Reference pressure at the base in cgs units.
        """
        self.T: float = T
        self.P_0: float = P_0
        self.constituents: List[Union['AtmosphericConstituent',
                                      'MolecularConstituent']] = []
        self.hasMoon: bool = False

    def getReferenceNumberDensity(self) -> float:
        """Calculates the number density at the reference pressure and temperature.

        Returns:
            float: The reference number density (n_0) in cm^-3.
        """
        n_0 = self.P_0 / (const.k_B * self.T)
        return n_0

    def getVelDispersion(self, m: float) -> float:
        """Calculates the thermal velocity dispersion for a given particle mass.

        Args:
            m (float): Mass of the particle in grams.

        Returns:
            float: The 1D thermal velocity dispersion (sigma_v) in cm/s.
        """
        sigma_v = np.sqrt(self.T * const.k_B / m)
        return sigma_v

    def addConstituent(self, speciesName: str, chi: float) -> None:
        """Adds an atomic or ionic constituent to the atmosphere.

        Args:
            speciesName (str): The name of the species (e.g., 'NaI').
            chi (float): The mixing ratio of this species.
        """
        species = const.AvailableSpecies().findSpecies(speciesName)
        m = species.mass
        sigma_v = self.getVelDispersion(m)
        constituent = AtmosphericConstituent(species, chi, sigma_v)
        self.constituents.append(constituent)

    def addMolecularConstituent(self, speciesName: str, chi: float) -> None:
        """Adds a molecular constituent to the atmosphere.

        Args:
            speciesName (str): The name of the molecule.
            chi (float): The mixing ratio of this molecule.
        """
        constituent = MolecularConstituent(speciesName, chi)
        self.constituents.append(constituent)


class BarometricAtmosphere(CollisionalAtmosphere):
    """An isothermal atmosphere with a density profile following the barometric formula.

    Attributes:
        mu (float): The mean molecular weight of the atmosphere in grams.
        planet (Any): The Planet object this atmosphere belongs to.
    """

    def __init__(self, T: float, P_0: float, mu: float, planet: Any):
        """Initializes the BarometricAtmosphere.

        Args:
            T (float): Temperature of the atmosphere in Kelvin.
            P_0 (float): Reference pressure at the base in cgs units.
            mu (float): The mean molecular weight of the atmosphere in grams.
            planet (Any): The Planet object this atmosphere belongs to.
        """
        super().__init__(T, P_0)
        self.mu: float = mu
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = BarometricAtmosphere.getReferenceNumberDensity(self)
        H = const.k_B * self.T * self.planet.R**2 / \
            (const.G * self.mu * self.planet.M)
        n = n_0 * np.exp((self.planet.R - r) / H) * \
            np.heaviside(r - self.planet.R, 1.)
        return n


class HydrostaticAtmosphere(CollisionalAtmosphere):
    """An isothermal atmosphere in hydrostatic equilibrium.

    Attributes:
        mu (float): The mean molecular weight of the atmosphere in grams.
        planet (Any): The Planet object this atmosphere belongs to.
    """

    def __init__(self, T: float, P_0: float, mu: float, planet: Any):
        """Initializes the HydrostaticAtmosphere.

        Args:
            T (float): Temperature of the atmosphere in Kelvin.
            P_0 (float): Reference pressure at the base in cgs units.
            mu (float): The mean molecular weight of the atmosphere in grams.
            planet (Any): The Planet object this atmosphere belongs to.
        """
        super().__init__(T, P_0)
        self.mu: float = mu
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = HydrostaticAtmosphere.getReferenceNumberDensity(self)
        Jeans_0 = const.G * self.mu * self.planet.M / \
            (const.k_B * self.T * self.planet.R)
        Jeans = const.G * self.mu * self.planet.M / \
            (const.k_B * self.T * r) * np.heaviside(r - self.planet.R, 1.)
        n = n_0 * np.exp(Jeans - Jeans_0)
        return n


class PowerLawAtmosphere(CollisionalAtmosphere):
    """An atmosphere with a density profile following a power law.

    Attributes:
        q (float): The power-law index for the density profile.
        planet (Any): The Planet object this atmosphere belongs to.
    """

    def __init__(self, T: float, P_0: float, q: float, planet: Any):
        """Initializes the PowerLawAtmosphere.

        Args:
            T (float): Temperature of the atmosphere in Kelvin.
            P_0 (float): Reference pressure at the base in cgs units.
            q (float): The power-law index.
            planet (Any): The Planet object this atmosphere belongs to.
        """
        super().__init__(T, P_0)
        self.q: float = q
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = PowerLawAtmosphere.getReferenceNumberDensity(self)
        n = n_0 * (self.planet.R / r)**self.q * \
            np.heaviside(r - self.planet.R, 1.)
        return n


class EvaporativeExosphere:
    """Base class for an exosphere model normalized by total particle number.

    Attributes:
        N (float): Total number of particles in the exosphere.
        hasMoon (bool): Flag indicating if a moon is present in this model.
    """

    def __init__(self, N: float):
        """Initializes the EvaporativeExosphere.

        Args:
            N (float): Total number of particles in the exosphere.
        """
        self.N: float = N
        self.hasMoon: bool = False

    def addConstituent(self, speciesName: str, sigma_v: float) -> None:
        """Adds an atomic or ionic constituent to the exosphere.

        Note: An evaporative exosphere can only have one constituent.

        Args:
            speciesName (str): The name of the species (e.g., 'NaI').
            sigma_v (float): The velocity dispersion of the species in cm/s.
        """
        species = const.AvailableSpecies().findSpecies(speciesName)
        constituent = AtmosphericConstituent(species, 1., sigma_v)
        self.constituents: List[AtmosphericConstituent] = [constituent]

    def addMolecularConstituent(self, speciesName: str, T: float) -> None:
        """Adds a molecular constituent to the exosphere.

        Note: An evaporative exosphere can only have one constituent.

        Args:
            speciesName (str): The name of the molecule.
            T (float): The pseudo-temperature for the molecular cross-sections.
        """
        constituent = MolecularConstituent(speciesName, 1.0)
        self.constituents: List[MolecularConstituent] = [constituent]
        self.T: float = T


class PowerLawExosphere(EvaporativeExosphere):
    """An exosphere with a density profile following a power law.

    Attributes:
        q (float): The power-law index for the density profile.
        planet (Any): The Planet object this exosphere belongs to.
    """

    def __init__(self, N: float, q: float, planet: Any):
        """Initializes the PowerLawExosphere.

        Args:
            N (float): Total number of particles in the exosphere.
            q (float): The power-law index.
            planet (Any): The Planet object this exosphere belongs to.
        """
        super().__init__(N)
        self.q: float = q
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.planet.R**3) * self.N
        n = n_0 * (self.planet.R / r)**self.q * \
            np.heaviside(r - self.planet.R, 1.)
        return n


class MoonExosphere(EvaporativeExosphere):
    """An exosphere sourced from a moon, with a power-law density profile.

    Attributes:
        q (float): The power-law index for the density profile.
        moon (Any): The Moon object this exosphere belongs to.
        planet (Any): The host planet of the moon.
    """

    def __init__(self, N: float, q: float, moon: Any):
        """Initializes the MoonExosphere.

        Args:
            N (float): Total number of particles in the exosphere.
            q (float): The power-law index.
            moon (Any): The Moon object this exosphere belongs to.
        """
        super().__init__(N)
        self.q: float = q
        self.moon: Any = moon
        self.hasMoon: bool = True
        self.planet: Any = moon.hostPlanet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        r = self.moon.getDistanceFromMoon(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.moon.R**3) * self.N
        n = n_0 * (self.moon.R / r)**self.q * np.heaviside(r - self.moon.R, 1.)
        return n


class TidallyHeatedMoon(EvaporativeExosphere):
    """A moon exosphere with a variable source rate dependent on orbital phase.

    This model is designed to simulate phenomena like volcanic activity on a
    tidally heated moon, where the outgassing rate changes with orbital position.

    Attributes:
        q (float): The power-law index for the density profile.
        moon (Any): The Moon object this exosphere belongs to.
        planet (Any): The host planet of the moon.
        N_function (Callable[[float], float]): An interpolation function that
            returns the total number of particles as a function of the moon's
            orbital phase.
    """

    def __init__(self, q: float, moon: Any):
        """Initializes the TidallyHeatedMoon model.

        Args:
            q (float): The power-law index for the density profile.
            moon (Any): The Moon object this exosphere belongs to.
        """
        self.q: float = q
        self.moon: Any = moon
        self.hasMoon: bool = True
        self.planet: Any = moon.hostPlanet

    def addSourceRateFunction(self, filename: str, tau_photoionization: float, mass_absorber: float) -> None:
        """Loads a source rate profile and creates an interpolation function.

        The total number of particles `N` at any time is calculated as
        `M_dot * tau / m`, where M_dot is the mass loss rate, tau is the
        photoionization lifetime, and m is the particle mass.

        Args:
            filename (str): Path to the file containing the mass loss rate (M_dot)
                as a function of the moon's orbital phase.
            tau_photoionization (float): The photoionization lifetime of the
                absorbing species in seconds.
            mass_absorber (float): The mass of a single absorbing particle in grams.
        """
        Mdot = np.loadtxt(filename)
        Mdot = np.concatenate((Mdot, Mdot[::-1]))
        phi_moon = np.linspace(0., 2. * np.pi, len(Mdot))
        N_function = interp1d(phi_moon, np.log10(
            Mdot * tau_photoionization / mass_absorber))
        self.N_function: Callable[[float], float] = N_function

    def calculateAbsorberNumber(self, orbphase: float) -> float:
        """Calculates the total number of absorbers for a given planetary orbital phase.

        Args:
            orbphase (float): The planet's orbital phase in radians.

        Returns:
            float: The total number of particles `N` in the exosphere.
        """
        orbphase_moon = self.moon.getOrbphase(orbphase) % (2. * np.pi)
        N = 10**self.N_function(orbphase_moon)
        return N

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        N = self.calculateAbsorberNumber(orbphase)
        r = self.moon.getDistanceFromMoon(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.moon.R**3) * N
        n = n_0 * (self.moon.R / r)**self.q * np.heaviside(r - self.moon.R, 1.)
        return n


class TorusExosphere(EvaporativeExosphere):
    """An exosphere model of a neutral gas torus around a planet.

    The density profile is Gaussian in both the radial and vertical directions
    of the torus.

    Attributes:
        a_torus (float): The radius of the torus centerline in cm.
        v_ej (float): The ejection velocity of particles, which determines the
            torus scale height, in cm/s.
        planet (Any): The Planet object this torus surrounds.
    """

    def __init__(self, N: float, a_torus: float, v_ej: float, planet: Any):
        """Initializes the TorusExosphere.

        Args:
            N (float): Total number of particles in the torus.
            a_torus (float): The radius of the torus centerline in cm.
            v_ej (float): The ejection velocity of particles in cm/s.
            planet (Any): The Planet object this torus surrounds.
        """
        super().__init__(N)
        self.a_torus: float = a_torus
        self.v_ej: float = v_ej
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        a, z = self.planet.getTorusCoords(x, phi, rho, orbphase)
        v_orbit = np.sqrt(const.G * self.planet.M / self.a_torus)
        H_torus = self.a_torus * self.v_ej / v_orbit
        n_a = np.exp(-((a - self.a_torus) / (4. * H_torus))**2)
        n_z = np.exp(-(z / H_torus)**2)
        term1 = 8. * H_torus**2 * np.exp(-self.a_torus**2 / (16. * H_torus**2))
        term2 = 2. * np.sqrt(np.pi) * self.a_torus * H_torus * \
            (erf(self.a_torus / (4. * H_torus)) + 1.)
        n_0 = 1. / (2. * np.pi**1.5 * H_torus * (term1 + term2)) * self.N
        n = n_0 * np.multiply(n_a, n_z)
        return n


class SerpensExosphere(EvaporativeExosphere):
    """An exosphere model based on particle data from the SERPENS simulation code.

    This class loads particle positions from a SERPENS output file, histograms
    them onto a 3D grid, and creates an interpolation function for the
    number density.

    Attributes:
        filename (str): Path to the SERPENS output file.
        planet (Any): The Planet object this exosphere belongs to.
        sigmaSmoothing (float): The sigma for Gaussian smoothing of the
            histogrammed density grid.
        InterpolatedDensity (Callable): A 3D interpolation function for number density.
    """

    def __init__(self, filename: str, N: float, planet: Any, sigmaSmoothing: float):
        """Initializes the SerpensExosphere.

        Args:
            filename (str): Path to the SERPENS output file.
            N (float): Total number of particles to scale the simulation to.
            planet (Any): The Planet object this exosphere belongs to.
            sigmaSmoothing (float): The sigma for Gaussian smoothing in grid units.
        """
        super().__init__(N)
        self.filename: str = filename
        self.planet: Any = planet
        self.sigmaSmoothing: float = sigmaSmoothing

    def addInterpolatedDensity(self, spatialGrid: Any) -> None:
        """Loads SERPENS data and creates the density interpolation function.

        Args:
            spatialGrid (Any): The `geometryHandler.Grid` object defining the
                simulation grid.
        """
        serpensOutput = np.loadtxt(self.filename) * 1e2
        particlePos = serpensOutput[:, 0:3]
        xBins = spatialGrid.constructXaxis(midpoints=False)
        yBins = np.linspace(-spatialGrid.rho_border,
                            spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps) + 1)
        zBins = np.linspace(-spatialGrid.rho_border,
                            spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps) + 1)
        cellVolume = (xBins[1] - xBins[0]) * \
            (yBins[1] - yBins[0]) * (zBins[1] - zBins[0])
        n_histogram = np.histogramdd(particlePos, bins=[xBins, yBins, zBins])[
            0] * self.N / (np.size(particlePos, axis=0) * cellVolume)
        if self.sigmaSmoothing > 0.:
            n_histogram = gauss(n_histogram, sigma=self.sigmaSmoothing)
        print('Sum over all particles, potentially smoothed with a Gaussian:', np.sum(
            n_histogram) * cellVolume)
        xPoints = spatialGrid.constructXaxis()
        yPoints = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(
            spatialGrid.rho_steps), endpoint=False) + 2. * spatialGrid.rho_border / (4. * spatialGrid.rho_steps)
        zPoints = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(
            spatialGrid.rho_steps), endpoint=False) + 2. * spatialGrid.rho_border / (4. * spatialGrid.rho_steps)
        x, y, z = np.meshgrid(xPoints, yPoints, zPoints, indexing='ij')
        SEL = ((y**2 + z**2) > self.planet.R**2) * \
            ((y**2 + z**2) < self.planet.hostStar.R**2)
        print('Sum over all particles outside of the planetary disk but inside the stellar disk:', np.sum(
            n_histogram[SEL]) * cellVolume)
        n_function = RegularGridInterpolator(
            (xPoints, yPoints, zPoints), n_histogram)
        self.InterpolatedDensity: Callable[[
            np.ndarray], np.ndarray] = n_function

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates number density using the pre-computed interpolation function.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        coordArray = np.array(
            [x, np.repeat(y, np.size(x)), np.repeat(z, np.size(x))]).T
        n = self.InterpolatedDensity(coordArray)
        return n


"""
Calculate absorption cross sections
"""


class AtmosphericConstituent:
    """Represents an atomic or ionic absorbing species.

    This class handles the calculation of absorption cross-sections for a given
    species by computing Voigt profiles for its spectral lines.

    Attributes:
        isMolecule (bool): Flag indicating this is not a molecule.
        species (Any): The `constants.Species` object.
        chi (float): The mixing ratio of this species.
        sigma_v (float): The velocity dispersion in cm/s.
        lookupFunction (Callable): An interpolation function for the absorption
            cross-section (`log10(sigma_abs)`) vs. wavelength.
    """

    def __init__(self, species: Any, chi: float, sigma_v: float):
        """Initializes the AtmosphericConstituent.

        Args:
            species (Any): The `constants.Species` object.
            chi (float): The mixing ratio of this species.
            sigma_v (float): The velocity dispersion in cm/s.
        """
        self.isMolecule: bool = False
        self.species: Any = species
        self.chi: float = chi
        self.sigma_v: float = sigma_v
        self.wavelengthGridRefinement: float = 10.
        self.wavelengthGridExtension: float = 0.01
        self.lookupOffset: float = 1e-50

    def getLineParameters(self, wavelength: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieves spectral line parameters from a data file.

        Reads the line list file and returns the parameters for the lines of this
        species that fall within the specified wavelength range.

        Args:
            wavelength (np.ndarray): A 2-element array with the min and max
                wavelengths of interest [cm].

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - line_wavelength (np.ndarray): Wavelengths of the lines [cm].
                - line_gamma (np.ndarray): Damping parameters (Gamma) of the lines.
                - line_f (np.ndarray): Oscillator strengths (f-values) of the lines.
        """
        lineList = np.loadtxt(lineListPath, dtype=str,
                              usecols=(0, 1, 2, 3, 4), skiprows=1)
        line_wavelength = np.array([x[1:-1] for x in lineList[:, 2]])
        line_A = np.array([x[1:-1] for x in lineList[:, 3]])
        line_f = np.array([x[1:-1] for x in lineList[:, 4]])
        SEL_COMPLETE = (line_wavelength != '') * \
            (line_A != '') * (line_f != '')
        SEL_SPECIES = (lineList[:, 0] == self.species.element) * \
            (lineList[:, 1] == self.species.ionizationState)
        line_wavelength = line_wavelength[SEL_SPECIES *
                                          SEL_COMPLETE].astype(float) * 1e-8
        line_gamma = line_A[SEL_SPECIES *
                            SEL_COMPLETE].astype(float) / (4. * np.pi)
        line_f = line_f[SEL_SPECIES * SEL_COMPLETE].astype(float)
        SEL_WAVELENGTH = (line_wavelength > min(wavelength)) * \
            (line_wavelength < max(wavelength))
        return line_wavelength[SEL_WAVELENGTH], line_gamma[SEL_WAVELENGTH], line_f[SEL_WAVELENGTH]

    def calculateVoigtProfile(self, wavelength: np.ndarray) -> np.ndarray:
        """Calculates the total absorption cross-section from all spectral lines.

        The profile is a sum of Voigt profiles for each line of the species.

        Args:
            wavelength (np.ndarray): Wavelength grid to calculate the profile on [cm].

        Returns:
            np.ndarray: The absorption cross-section (sigma_abs) in cm^2.
        """
        line_wavelength, line_gamma, line_f = self.getLineParameters(
            wavelength)
        sigma_abs = np.zeros_like(wavelength)
        for idx in range(len(line_wavelength)):
            lineProfile = voigt_profile(
                const.c / wavelength - const.c / line_wavelength[idx], self.sigma_v / line_wavelength[idx], line_gamma[idx])
            sigma_abs += np.pi * \
                (const.e)**2 / (const.m_e * const.c) * \
                line_f[idx] * lineProfile
        return sigma_abs

    def constructLookupFunction(self, wavelengthGrid: 'WavelengthGrid') -> Callable[[np.ndarray], np.ndarray]:
        """Creates an interpolation function for the absorption cross-section.

        Calculates the Voigt profile on a refined grid and creates a 1D
        interpolator for `log10(sigma_abs)` to speed up subsequent calculations.

        Args:
            wavelengthGrid (WavelengthGrid): The wavelength grid object for the simulation.

        Returns:
            Callable[[np.ndarray], np.ndarray]: The interpolation function.
        """
        wavelengthGridRefined = deepcopy(wavelengthGrid)
        wavelengthGridRefined.resolutionHigh /= self.wavelengthGridRefinement
        wavelengthGridRefined.lower_w *= (1. - self.wavelengthGridExtension)
        wavelengthGridRefined.upper_w *= (1. + self.wavelengthGridExtension)
        wavelengthRefined = wavelengthGridRefined.constructWavelengthGridSingle(
            self)
        sigma_abs = self.calculateVoigtProfile(wavelengthRefined)
        lookupFunction = interp1d(wavelengthRefined, np.log10(
            sigma_abs + self.lookupOffset), bounds_error=False, fill_value=np.log10(self.lookupOffset))
        return lookupFunction

    def addLookupFunctionToConstituent(self, wavelengthGrid: 'WavelengthGrid') -> None:
        """Constructs and attaches the lookup function to the object instance.

        Args:
            wavelengthGrid (WavelengthGrid): The simulation's wavelength grid object.
        """
        lookupFunction = self.constructLookupFunction(wavelengthGrid)
        self.lookupFunction: Callable[[
            np.ndarray], np.ndarray] = lookupFunction

    def getSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        """Retrieves the absorption cross-section using the lookup function.

        Args:
            wavelength (np.ndarray): Wavelength array [cm].

        Returns:
            np.ndarray: Absorption cross-section array [cm^2].
        """
        sigma_absFlattened = 10**self.lookupFunction(
            wavelength.flatten()) - self.lookupOffset
        sigma_abs = sigma_absFlattened.reshape(wavelength.shape)
        return sigma_abs


class MolecularConstituent:
    """Represents a molecular absorbing species.

    This class handles the retrieval of pre-computed molecular cross-sections
    from HDF5 files, which are functions of pressure, temperature, and wavelength.

    Attributes:
        isMolecule (bool): Flag indicating this is a molecule.
        moleculeName (str): Name of the molecule.
        chi (float): Mixing ratio of the molecule.
        lookupFunction (Callable): A multi-dimensional interpolation function for
            the absorption cross-section.
    """

    def __init__(self, moleculeName: str, chi: float):
        """Initializes the MolecularConstituent.

        Args:
            moleculeName (str): The name of the molecule, corresponding to the
                HDF5 filename (without extension).
            chi (float): The mixing ratio of this molecule.
        """
        self.isMolecule: bool = True
        self.lookupOffset: float = 1e-50
        self.moleculeName: str = moleculeName
        self.chi: float = chi

    def constructLookupFunction(self) -> Callable[[np.ndarray], np.ndarray]:
        """Creates an interpolation function from an HDF5 cross-section file.

        Reads pressure, temperature, wavelength, and cross-section data from
        an HDF5 file and creates a RegularGridInterpolator.

        Returns:
            Callable[[np.ndarray], np.ndarray]: The interpolation function.
        """
        with h5py.File(molecularLookupPath + self.moleculeName + '.h5', 'r+') as f:
            P = f['p'][:] * 10.
            T = f['t'][:]
            wavelength = 1. / f['bin_edges'][:][::-1]
            sigma_abs = f['xsecarr'][:][:, :, ::-1]
            lookupFunction = RegularGridInterpolator((P, T, wavelength), np.log10(
                sigma_abs + self.lookupOffset), bounds_error=False, fill_value=np.log10(self.lookupOffset))
            return lookupFunction

    def addLookupFunctionToConstituent(self) -> None:
        """Constructs and attaches the lookup function to the object instance."""
        lookupFunction = self.constructLookupFunction()
        self.lookupFunction: Callable[[
            np.ndarray], np.ndarray] = lookupFunction

    def getSigmaAbs(self, P: np.ndarray, T: float, wavelength: np.ndarray) -> np.ndarray:
        """Retrieves the absorption cross-section using the lookup function.

        Args:
            P (np.ndarray): Pressure array [cgs units].
            T (float): Temperature [K].
            wavelength (np.ndarray): Wavelength array [cm].

        Returns:
            np.ndarray: Absorption cross-section array [cm^2].
        """
        wavelengthFlattened = wavelength.flatten()
        TFlattened = np.full_like(wavelengthFlattened, T)
        PFlattened = np.repeat(
            np.clip(P, a_min=1e-4, a_max=None), np.size(wavelength, axis=1))
        inputArray = np.array([PFlattened, TFlattened, wavelengthFlattened]).T
        sigma_absFlattened = 10**self.lookupFunction(
            inputArray) - self.lookupOffset
        sigma_abs = sigma_absFlattened.reshape(wavelength.shape)
        return sigma_abs


class Atmosphere:
    """Manages all atmospheric/exospheric density distributions for a simulation.

    This class aggregates multiple density models (e.g., a barometric atmosphere
    plus a moon exosphere) and calculates the total optical depth along a line of sight.

    Attributes:
        densityDistributionList (List[Any]): A list of density distribution
            model objects (e.g., `BarometricAtmosphere`, `PowerLawExosphere`).
        hasOrbitalDopplerShift (bool): Flag indicating whether to include
            Doppler shifts from orbital motion.
    """

    def __init__(self, densityDistributionList: List[Any], hasOrbitalDopplerShift: bool):
        """Initializes the Atmosphere object.

        Args:
            densityDistributionList (List[Any]): A list of density model objects.
            hasOrbitalDopplerShift (bool): Flag for including orbital Doppler shifts.
        """
        self.densityDistributionList: List[Any] = densityDistributionList
        self.hasOrbitalDopplerShift: bool = hasOrbitalDopplerShift

    @staticmethod
    def getAbsorberNumberDensity(densityDistribution: Any, chi: float, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density of a specific absorbing species.

        Args:
            densityDistribution (Any): The density model object.
            chi (float): The mixing ratio of the absorber.
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density of the absorber at each x coordinate [cm^-3].
        """
        n_total = densityDistribution.calculateNumberDensity(
            x, phi, rho, orbphase)
        n_abs = n_total * chi
        return n_abs

    def getAbsorberVelocityField(self, densityDistribution: Any, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the line-of-sight velocity of the gas.

        Args:
            densityDistribution (Any): The density model object.
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The line-of-sight velocity at each x coordinate [cm/s].
        """
        v_los = np.zeros_like(x)
        if self.hasOrbitalDopplerShift:
            if not densityDistribution.hasMoon:
                v_los += densityDistribution.planet.getLOSvelocity(orbphase)
            else:
                v_los += densityDistribution.moon.getLOSvelocity(orbphase)
        return v_los

    def getLOSopticalDepth(self, x: np.ndarray, phi: float, rho: float, orbphase: float, wavelength: np.ndarray, delta_x: float) -> np.ndarray:
        """Calculates the total line-of-sight (LOS) optical depth.

        This method sums the contributions from all constituents in all density
        distributions to compute the total optical depth along a single chord
        through the atmosphere.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).
            wavelength (np.ndarray): The wavelength grid [cm].
            delta_x (float): The step size along the x-axis (line of sight) [cm].

        Returns:
            np.ndarray: The total optical depth (tau) for each wavelength.
        """
        kappa = np.zeros((len(x), len(wavelength)))
        for densityDistribution in self.densityDistributionList:
            for constituent in densityDistribution.constituents:
                v_los = self.getAbsorberVelocityField(
                    densityDistribution, x, phi, rho, orbphase)
                shift = const.calculateDopplerShift(-v_los)
                wavelengthShifted = np.tensordot(shift, wavelength, axes=0)
                if constituent.isMolecule:
                    n_tot = densityDistribution.calculateNumberDensity(
                        x, phi, rho, orbphase)
                    n_abs = n_tot * constituent.chi
                    T = densityDistribution.T
                    P = n_tot * const.k_B * T
                    sigma_abs = constituent.getSigmaAbs(
                        P, T, wavelengthShifted)
                else:
                    n_abs = Atmosphere.getAbsorberNumberDensity(
                        densityDistribution, constituent.chi, x, phi, rho, orbphase)
                    sigma_abs = constituent.getSigmaAbs(wavelengthShifted)
                kappa += np.tile(n_abs, (len(wavelength), 1)).T * sigma_abs
        LOStau = np.sum(kappa, axis=0) * delta_x
        return LOStau


class WavelengthGrid:
    """Creates and manages the wavelength grid for the simulation.

    The grid can be non-uniform, with higher resolution around specified
    spectral lines and lower resolution elsewhere.

    Attributes:
        lower_w (float): The lower bound of the wavelength range [cm].
        upper_w (float): The upper bound of the wavelength range [cm].
        widthHighRes (float): The width of the high-resolution region around
            each spectral line [cm].
        resolutionLow (float): The step size for the low-resolution parts of the grid [cm].
        resolutionHigh (float): The step size for the high-resolution parts of the grid [cm].
    """

    def __init__(self, lower_w: float, upper_w: float, widthHighRes: float, resolutionLow: float, resolutionHigh: float):
        """Initializes the WavelengthGrid.

        Args:
            lower_w (float): Lower wavelength bound [cm].
            upper_w (float): Upper wavelength bound [cm].
            widthHighRes (float): High-resolution region width [cm].
            resolutionLow (float): Low-resolution step size [cm].
            resolutionHigh (float): High-resolution step size [cm].
        """
        self.lower_w: float = lower_w
        self.upper_w: float = upper_w
        self.widthHighRes: float = widthHighRes
        self.resolutionLow: float = resolutionLow
        self.resolutionHigh: float = resolutionHigh

    def arangeWavelengthGrid(self, linesList: List[float]) -> np.ndarray:
        """Constructs a non-uniform wavelength grid.

        Creates a grid with high resolution around the specified line centers
        and low resolution in between.

        Args:
            linesList (List[float]): A list of spectral line center wavelengths [cm].

        Returns:
            np.ndarray: The constructed wavelength grid [cm].
        """
        peaks = np.sort(np.unique(linesList))
        diff = np.concatenate(([np.inf], np.diff(peaks), [np.inf]))
        if len(peaks) == 0:
            print(
                'WARNING: No absorption lines from atoms/ions in the specified wavelength range!')
            return np.arange(self.lower_w, self.upper_w, self.resolutionLow)
        HighResBorders: Tuple[List[float], List[float]] = ([], [])
        for idx, peak in enumerate(peaks):
            if diff[idx] > self.widthHighRes:
                HighResBorders[0].append(peak - self.widthHighRes / 2.)
            if diff[idx + 1] > self.widthHighRes:
                HighResBorders[1].append(peak + self.widthHighRes / 2.)
        grid: List[np.ndarray] = []
        for idx in range(len(HighResBorders[0])):
            grid.append(
                np.arange(HighResBorders[0][idx], HighResBorders[1][idx], self.resolutionHigh))
            if idx == 0:
                if self.lower_w < HighResBorders[0][0]:
                    grid.append(
                        np.arange(self.lower_w, HighResBorders[0][0], self.resolutionLow))
                if len(HighResBorders[0]) == 1 and self.upper_w > HighResBorders[1][-1]:
                    grid.append(
                        np.arange(HighResBorders[1][0], self.upper_w, self.resolutionLow))
            elif idx == len(HighResBorders[0]) - 1:
                grid.append(np.arange(
                    HighResBorders[1][idx - 1], HighResBorders[0][idx], self.resolutionLow))
                if self.upper_w > HighResBorders[1][-1]:
                    grid.append(
                        np.arange(HighResBorders[1][-1], self.upper_w, self.resolutionLow))
            else:
                grid.append(np.arange(
                    HighResBorders[1][idx - 1], HighResBorders[0][idx], self.resolutionLow))
        wavelengthGrid = np.sort(np.concatenate(grid))
        return wavelengthGrid

    def constructWavelengthGridSingle(self, constituent: AtmosphericConstituent) -> np.ndarray:
        """Constructs a wavelength grid for a single atomic/ionic constituent.

        Args:
            constituent (AtmosphericConstituent): The constituent to get lines from.

        Returns:
            np.ndarray: The constructed wavelength grid [cm].
        """
        linesList = constituent.getLineParameters(
            np.array([self.lower_w, self.upper_w]))[0]
        return self.arangeWavelengthGrid(linesList)

    def constructWavelengthGrid(self, densityDistributionList: List[Any]) -> np.ndarray:
        """Constructs a wavelength grid for all atomic/ionic species in the atmosphere.

        Molecular opacities are continuous and do not influence the grid construction.

        Args:
            densityDistributionList (List[Any]): List of all density models.

        Returns:
            np.ndarray: The final wavelength grid for the simulation [cm].
        """
        linesList: List[float] = []
        for densityDistribution in densityDistributionList:
            for constituent in densityDistribution.constituents:
                if constituent.isMolecule:
                    continue
                lines_w = constituent.getLineParameters(
                    np.array([self.lower_w, self.upper_w]))[0]
                linesList.extend(lines_w)
        if len(linesList) == 0:
            return np.arange(self.lower_w, self.upper_w, self.resolutionLow)
        return self.arangeWavelengthGrid(linesList)


class Transit:
    """Main class to orchestrate a transit simulation.

    This class combines the atmospheric model, wavelength grid, and spatial grid
    to calculate the transit light curve or transmission spectrum.

    Attributes:
        atmosphere (Atmosphere): The atmosphere object containing all gas properties.
        wavelengthGrid (WavelengthGrid): The wavelength grid object.
        spatialGrid (Any): The `geometryHandler.Grid` object.
        planet (Any): The primary `Planet` object of the simulation.
        wavelength (np.ndarray): The wavelength array for the simulation.
    """

    def __init__(self, atmosphere: Atmosphere, wavelengthGrid: WavelengthGrid, spatialGrid: geom.Grid):
        """Initializes the Transit simulation object.

        Args:
            atmosphere (Atmosphere): The atmosphere object.
            wavelengthGrid (WavelengthGrid): The wavelength grid object.
            spatialGrid (Any): The `geometryHandler.Grid` object.
        """
        self.atmosphere: Atmosphere = atmosphere
        self.wavelengthGrid: WavelengthGrid = wavelengthGrid
        self.spatialGrid: geom.Grid = spatialGrid
        self.planet: Any = self.atmosphere.densityDistributionList[0].planet

    def addWavelength(self) -> None:
        """Constructs and stores the wavelength grid for the simulation."""
        wavelength = self.wavelengthGrid.constructWavelengthGrid(
            self.atmosphere.densityDistributionList)
        self.wavelength: np.ndarray = wavelength

    def checkBlock(self, phi: float, rho: float, orbphase: float) -> bool:
        """Checks if a line of sight is blocked by the opaque body of a planet or moon.

        Args:
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): The planet's orbital phase (radians).

        Returns:
            bool: True if the line of sight is blocked, False otherwise.
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        y_p = self.planet.getPosition(orbphase)[1]
        blockingPlanet = (np.sqrt((y - y_p)**2 + z**2) < self.planet.R)
        if blockingPlanet:
            return True
        for densityDistribution in self.atmosphere.densityDistributionList:
            if densityDistribution.hasMoon:
                moon = densityDistribution.moon
                y_moon = moon.getPosition(orbphase)[1]
                blockingMoon = ((y - y_moon)**2 + z**2 < moon.R)
                if blockingMoon:
                    return True
        return False

    def evaluateChord(self, phi: float, rho: float, orbphase: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the transmitted flux along a single line of sight (chord).

        This involves getting the unattenuated stellar flux, calculating the
        optical depth through the atmosphere, and applying the Beer-Lambert law.

        Args:
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): The planet's orbital phase (radians).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - F_in (np.ndarray): The attenuated flux received by the observer.
                - F_out (np.ndarray): The unattenuated flux from that point on the star.
        """
        Fstar = self.planet.hostStar.getFstar(phi, rho, self.wavelength)
        F_out = rho * Fstar * self.wavelength / self.wavelength
        if self.checkBlock(phi, rho, orbphase):
            F_in = np.zeros_like(self.wavelength)
            return F_in, F_out
        x = self.spatialGrid.constructXaxis()
        delta_x = self.spatialGrid.getDeltaX()
        tau = self.atmosphere.getLOSopticalDepth(
            x, phi, rho, orbphase, self.wavelength, delta_x)
        F_in = rho * Fstar * np.exp(-tau)
        return F_in, F_out

    def sumOverChords(self) -> np.ndarray:
        """Integrates the flux over all chords to get the final transit depth.

        This method iterates over the entire spatial grid (phi, rho, orbphase),
        calls `evaluateChord` for each point, and sums the results to compute
        the ratio of in-transit flux to out-of-transit flux.

        Returns:
            np.ndarray: An array of the flux ratio (transit depth) for each
                orbital phase and wavelength. Shape is (orbphase_steps, n_wavelengths).
        """
        chordGrid = self.spatialGrid.getChordGrid()
        F_in: List[np.ndarray] = []
        F_out: List[np.ndarray] = []
        for chord in chordGrid:
            Fsingle_in, Fsingle_out = self.evaluateChord(
                chord[0], chord[1], chord[2])
            F_in.append(Fsingle_in)
            F_out.append(Fsingle_out)
        F_in = np.array(F_in).reshape((self.spatialGrid.phi_steps * self.spatialGrid.rho_steps,
                                       self.spatialGrid.orbphase_steps, len(self.wavelength)))
        F_out = np.array(F_out).reshape((self.spatialGrid.phi_steps *
                                         self.spatialGrid.rho_steps, self.spatialGrid.orbphase_steps, len(self.wavelength)))
        R = np.sum(F_in, axis=0) / np.sum(F_out, axis=0)
        return R
