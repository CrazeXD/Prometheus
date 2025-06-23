"""
This file defines classes for celestial bodies (Star, Planet, Moon)
and a utility class to load pre-defined systems from data files.
These classes encapsulate the physical properties and orbital mechanics
required for transit simulations.

Created on 16. September 2022 by Andrea Gebek.
"""

import csv
import os
import pathlib
import shutil
import urllib.request as request
from contextlib import closing
from typing import Any, Callable, Optional, Tuple

import astropy.io.fits as fits
import numpy as np
from scipy.interpolate import interp1d

from . import constants as const
from . import geometryHandler as geom


class Star:
    """Represents a star with its physical and observational properties.

    This class stores stellar parameters, handles the retrieval of stellar spectra,
    and calculates effects like center-to-limb variation (CLV) and the
    Rossiter-McLaughlin (RM) effect.

    Attributes:
        R (float): Stellar radius in cm.
        M (float): Stellar mass in g.
        T_eff (float): Effective temperature in Kelvin.
        log_g (float): Logarithm of the surface gravity (log10(cm/s^2)).
        Z (float): Metallicity [Fe/H].
        alpha (float): Alpha-element enhancement [alpha/Fe].
        CLV_u1 (float): Linear limb-darkening coefficient.
        CLV_u2 (float): Quadratic limb-darkening coefficient.
        vsiniStarrot (float): Projected stellar rotational velocity in cm/s.
        phiStarrot (float): Azimuthal angle of the stellar rotation axis in radians.
        Fstar_function (Optional[Callable]): An interpolation function for the
            stellar flux as a function of wavelength.
    """

    def __init__(self, R: float, dR: float, M: float, dM: float, T_eff: float, dT_eff: float, log_g: float, dlog_g: float, Z: float, dZ: float, alpha: float) -> None:
        """Initializes the Star object with its fundamental properties.

        Args:
            R (float): Stellar radius in cm.
            M (float): Stellar mass in g.
            T_eff (float): Effective temperature in Kelvin.
            log_g (float): Logarithm of the surface gravity (log10(cm/s^2)).
            Z (float): Metallicity [Fe/H].
            alpha (float): Alpha-element enhancement [alpha/Fe].
        """
        self.R: float = R
        self.dR: float = dR
        self.M: float = M
        self.dM: float = dM
        self.T_eff: float = T_eff
        self.dT_eff: float = dT_eff
        self.log_g: float = log_g
        self.dlog_g: float = dlog_g
        self.Z: float = Z
        self.dZ: float = dZ
        self.alpha: float = alpha
        self.CLV_u1: float = 0.
        self.CLV_u2: float = 0.
        self.vsiniStarrot: float = 0.
        self.phiStarrot: float = 0.
        self.Fstar_function: Optional[Callable[[Any], Any]] = None

    def addCLVparameters(self, CLV_u1: float, CLV_u2: float) -> None:
        """Adds quadratic limb-darkening coefficients to the star.

        Args:
            CLV_u1 (float): The linear limb-darkening coefficient.
            CLV_u2 (float): The quadratic limb-darkening coefficient.
        """
        self.CLV_u1 = CLV_u1
        self.CLV_u2 = CLV_u2

    def addRMparameters(self, vsiniStarrot: float, phiStarrot: float) -> None:
        """Adds Rossiter-McLaughlin effect parameters to the star.

        Args:
            vsiniStarrot (float): Projected stellar rotational velocity in cm/s.
            phiStarrot (float): Azimuthal angle of the stellar rotation axis
                in radians.
        """
        self.vsiniStarrot = vsiniStarrot
        self.phiStarrot = phiStarrot

    def getSurfaceVelocity(self, phi: float, rho: float) -> float:
        """Calculates the line-of-sight velocity of a point on the stellar surface.

        This is used for calculating the Rossiter-McLaughlin effect.

        Args:
            phi (float): The azimuthal angle on the stellar disk in radians.
            rho (float): The projected radial distance from the star's center in cm.

        Returns:
            float: The line-of-sight velocity at the specified point in cm/s.
        """
        v_los = self.vsiniStarrot * rho / \
            self.R * np.cos(phi - self.phiStarrot)
        return v_los

    @staticmethod
    def round_to_grid(grid: np.ndarray, value: float) -> float:
        """Finds the value in a grid closest to a given value.

        Args:
            grid (np.ndarray): The array of grid points.
            value (float): The value to match.

        Returns:
            float: The grid point closest to the input value.
        """
        diff = np.subtract(value, grid)
        arg = np.argmin(np.abs(diff))
        return grid[arg]

    def getSpectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Queries a PHOENIX photosphere model, either from disk or from the PHOENIX website
        if the spectrum hasn't been downloaded before, and return the wavelength and flux arrays.

        The function rounds the stellar parameters (effective temperature, surface gravity,
        metallicity, and alpha enhancement) to the nearest available grid values, constructs
        the appropriate URLs for the PHOENIX model FITS files, downloads them, reads the data,
        and returns the wavelength and flux arrays in cgs units.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - Wavelength array (in cm)
            - Flux array (in cgs units, divided by pi)
        """
        # These contain the acceptable values.
        T_grid = np.concatenate(
            (np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)))
        log_g_grid = np.arange(0, 6.5, 0.5)
        Z_grid = np.concatenate(
            (np.arange(-4, -1, 1), np.arange(-1.5, 1.5, 0.5)))
        alpha_grid = np.arange(0, 1.6, 0.2)-0.2

        T_a = Star.round_to_grid(T_grid, self.T_eff)
        log_g_a = Star.round_to_grid(log_g_grid, self.log_g)
        Z_a = Star.round_to_grid(Z_grid, self.Z)
        alpha_a = Star.round_to_grid(alpha_grid, self.alpha)

        # This is where phoenix spectra are located.
        root = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'

        # We assemble a combination of strings to parse the user input into the URL,
        z_string = '{:.1f}'.format(float(Z_a))
        if Z_a > 0:
            z_string = '+' + z_string
        elif Z_a == 0:
            z_string = '-' + z_string
        else:
            z_string = z_string
        a_string = ''
        if alpha_a > 0:
            a_string = '.Alpha=+'+'{:.2f}'.format(float(alpha_a))
        if alpha_a < 0:
            a_string = '.Alpha='+'{:.2f}'.format(float(alpha_a))
        t_string = str(int(T_a))
        if T_a < 10000:
            t_string = '0'+t_string
        g_string = '-'+'{:.2f}'.format(float(log_g_a))

        # These are URLS for the input files.
        waveurl = root+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
        specurl = root+'PHOENIX-ACES-AGSS-COND-2011/Z'+z_string+a_string+'/lte' + \
            t_string+g_string+z_string+a_string+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

        # These are the output filenames, they will also be returned so that the wrapper
        # of this function can take them in.
        wavename = 'WAVE.fits'
        specname = 'lte'+t_string+g_string+z_string + \
            a_string+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

        # Download PHOENIX spectra:
        with closing(request.urlopen(waveurl)) as r:
            with open(wavename, 'wb') as f:
                shutil.copyfileobj(r, f)

        with closing(request.urlopen(specurl)) as r:
            with open(specname, 'wb') as f:
                shutil.copyfileobj(r, f)

        F = fits.getdata(specname)
        w = fits.getdata(wavename)

        os.remove(wavename)
        os.remove(specname)

        # Conversion to cgs-units. Note that Jens divides F by
        return (w * 1e-8, F / np.pi)
        # a seemingly random factor of pi, but this should not bother the transit calculations here.

    def calculateCLV(self, rho: float) -> float:
        """Calculates the center-to-limb variation (CLV) factor for a given radius.

        This uses a quadratic limb-darkening law.

        Args:
            rho (float): The projected radial distance from the star's center in cm.

        Returns:
            float: The CLV intensity factor, normalized to 1 at the center.
        """
        arg = 1. - np.sqrt(1. - rho**2 / self.R**2)
        return 1. - self.CLV_u1 * arg - self.CLV_u2 * arg**2

    def calculateRM(self, phi: float, rho: float, wavelength: np.ndarray) -> np.ndarray:
        """Calculates the Doppler-shifted stellar flux from a point on the star's surface.

        Args:
            phi (float): The azimuthal angle on the stellar disk in radians.
            rho (float): The projected radial distance from the star's center in cm.
            wavelength (np.ndarray): The array of wavelengths at which to calculate the flux.

        Returns:
            np.ndarray: The Doppler-shifted flux at the given location.
        """
        v_los = self.getSurfaceVelocity(phi, rho)
        shift = const.calculateDopplerShift(v_los)
        F_shifted = 10.**self.Fstar_function(wavelength / shift)
        return F_shifted

    def addFstarFunction(self, wavelength: np.ndarray) -> None:
        """Creates and stores an interpolation function for the stellar spectrum.

        This function fetches the PHOENIX spectrum, selects the relevant wavelength
        range, and creates a 1D interpolation function for `log10(flux)` vs.
        `wavelength`.

        Args:
            wavelength (np.ndarray): The wavelength grid for the simulation, used to
                determine the required range of the stellar spectrum.
        """
        PHOENIX_output = self.getSpectrum()
        w_star = PHOENIX_output[0]
        w_max = np.max(wavelength) * \
            const.calculateDopplerShift(-self.vsiniStarrot)
        w_min = np.min(wavelength) * \
            const.calculateDopplerShift(self.vsiniStarrot)
        SEL = (w_star >= w_min) * (w_star <= w_max)
        minArg = max(min(np.argwhere(SEL)).item() - 1, 0)
        maxArg = max(np.argwhere(SEL)).item() + 2
        w_starSEL = w_star[minArg:maxArg]
        F_0 = PHOENIX_output[1][minArg:maxArg]
        Fstar_function = interp1d(w_starSEL, np.log10(F_0), kind='linear')
        self.Fstar_function = Fstar_function

    def getFstarIntegrated(self, wavelength: np.ndarray, grid: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the stellar flux integrated over the entire disk.

        This method computes the total flux from the star, considering CLV and RM effects.
        It also calculates the flux from the upper part of the star, which is used
        for normalization in some contexts.

        Args:
            wavelength (np.ndarray): The array of wavelengths for the calculation.
            grid (Any): The spatial grid object (`geometryHandler.Grid`), used for
                discretization.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The total integrated stellar flux.
                - The integrated flux from the upper portion of the stellar disk.
        """
        if self.vsiniStarrot == 0.:
            FstarIntegrated = np.pi * self.R**2 * \
                (1. - self.CLV_u1 / 3. - self.CLV_u2 / 6.) * \
                np.ones_like(wavelength)
            upperTerm = 0.5 * (-self.CLV_u2 * self.R**2 -
                               self.CLV_u1 * self.R**2 + self.R**2)
            term1 = -4. * self.R**2 * self.CLV_u1 * \
                (1. - grid.rho_border**2 / self.R**2)**1.5
            term2 = self.R**2 * self.CLV_u2 * (6 * grid.rho_border**2 / self.R**2 + 8. * (
                1. - grid.rho_border**2 / self.R**2)**1.5 - 3. * (self.R**2 - grid.rho_border**2)**2 / self.R**4)
            lowerTerm = 1. / 12. * \
                (term1 - term2 - 6. * self.CLV_u1 *
                 grid.rho_border**2 + 6. * grid.rho_border**2)
            FstarUpper = 2. * np.pi * \
                (upperTerm - lowerTerm) * np.ones_like(wavelength)
        else:
            phiArray = grid.constructPhiAxis()
            delta_phi = grid.getDeltaPhi()
            rhoArray = grid.constructRhoAxis()
            delta_rho = grid.getDeltaRho()
            FstarIntegrated = np.zeros_like(wavelength)
            for phi in phiArray:
                for rho in rhoArray:
                    Fstar = self.calculateRM(phi, rho, wavelength)
                    Fstar *= self.calculateCLV(rho)
                    FstarIntegrated += Fstar * delta_phi * delta_rho * rho
                    FstarUpper = np.zeros_like(wavelength)
        return FstarIntegrated, FstarUpper

    def getFstar(self, phi: float, rho: float, wavelength: np.ndarray) -> np.ndarray:
        """Calculates the stellar flux from a specific point on the disk.

        If stellar rotation (`vsiniStarrot`) is non-zero, this includes the RM effect.
        It always includes the CLV effect.

        Args:
            phi (float): The azimuthal angle on the stellar disk in radians.
            rho (float): The projected radial distance from the star's center in cm.
            wavelength (np.ndarray): The array of wavelengths.

        Returns:
            np.ndarray: The flux array from the specified point on the stellar disk.
        """
        if self.vsiniStarrot == 0.:
            Fstar = np.ones_like(wavelength) * self.calculateCLV(rho)
        else:
            Fstar = self.calculateRM(phi, rho, wavelength)
            Fstar *= self.calculateCLV(rho)
        return Fstar


class Planet:
    """Represents a planet and its orbital properties.

    Attributes:
        name (str): The name of the planet.
        R (float): The radius of the planet in cm.
        M (float): The mass of the planet in g.
        a (float): The semi-major axis of the planet's orbit in cm.
        hostStar (Star): The host star object.
        transitDuration (float): The duration of the transit in hours.
        orbitalPeriod (float): The orbital period of the planet in days.
    """

    def __init__(self, name: str, R: float, M: float, a: float, hostStar: Star, transitDuration: float, orbitalPeriod: float) -> None:
        """Initializes the Planet object.

        Args:
            name (str): The name of the planet.
            R (float): The radius of the planet in cm.
            M (float): The mass of the planet in g.
            a (float): The semi-major axis of the planet's orbit in cm.
            hostStar (Star): The host star object.
            transitDuration (float): The duration of the transit in hours.
            orbitalPeriod (float): The orbital period of the planet in days.
        """
        self.name: str = name
        self.R: float = R
        self.M: float = M
        self.a: float = a
        self.hostStar: Star = hostStar
        self.transitDuration: float = transitDuration
        self.orbitalPeriod: float = orbitalPeriod

    def getPosition(self, orbphase: float) -> Tuple[float, float]:
        """Calculates the planet's (x, y) coordinates for a given orbital phase.

        Assumes a circular orbit viewed edge-on. The observer is along the x-axis.

        Args:
            orbphase (float): The orbital phase in radians (0 at mid-transit).

        Returns:
            Tuple[float, float]: The x and y coordinates of the planet in cm.
        """
        x_p = self.a * np.cos(orbphase)
        y_p = self.a * np.sin(orbphase)
        return x_p, y_p

    def getLOSvelocity(self, orbphase: float) -> float:
        """Calculates the planet's line-of-sight velocity.

        Assumes a circular orbit.

        Args:
            orbphase (float): The orbital phase in radians (0 at mid-transit).

        Returns:
            float: The line-of-sight velocity in cm/s.
        """
        v_los = -np.sin(orbphase) * np.sqrt(const.G * self.hostStar.M / self.a)
        return v_los

    def getDistanceFromPlanet(self, x: float, phi: float, rho: float, orbphase: float) -> float:
        """Calculates the 3D distance from a point in space to the planet's center.

        The point is defined in a cylindrical coordinate system (x, phi, rho)
        relative to the observer's line of sight through the star's center.

        Args:
            x (float): The coordinate along the line of sight in cm.
            phi (float): The azimuthal angle on the sky plane in radians.
            rho (float): The projected radial distance from the star's center in cm.
            orbphase (float): The planet's orbital phase in radians.

        Returns:
            float: The distance from the point to the planet's center in cm.
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_p, y_p = self.getPosition(orbphase)
        r_fromPlanet = np.sqrt((x - x_p)**2 + (y - y_p)**2 + z**2)
        return r_fromPlanet

    def getTorusCoords(self, x: float, phi: float, rho: float, orbphase: float) -> Tuple[float, float]:
        """Calculates coordinates relative to the planet for a torus model.

        Args:
            x (float): The coordinate along the line of sight in cm.
            phi (float): The azimuthal angle on the sky plane in radians.
            rho (float): The projected radial distance from the star's center in cm.
            orbphase (float): The planet's orbital phase in radians.

        Returns:
            Tuple[float, float]: A tuple containing:
                - a (float): The projected radial distance from the planet's center in the orbital plane.
                - z (float): The vertical distance from the orbital plane.
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_p, y_p = self.getPosition(orbphase)
        a = np.sqrt((x - x_p)**2 + (y - y_p)**2)
        return a, z


class Moon:
    """Represents a moon orbiting a planet.

    Attributes:
        midTransitOrbphase (float): The orbital phase of the moon (relative to
            its planet) at the time of the planet's mid-transit. In radians.
        R (float): The radius of the moon in cm.
        a (float): The semi-major axis of the moon's orbit around the planet in cm.
        hostPlanet (Planet): The host planet object.
    """

    def __init__(self, midTransitOrbphase: float, R: float, a: float, hostPlanet: Planet) -> None:
        """Initializes the Moon object.

        Args:
            midTransitOrbphase (float): The orbital phase of the moon relative to
                its planet at the time of the planet's mid-transit, in radians.
            R (float): The radius of the moon in cm.
            a (float): The semi-major axis of the moon's orbit around the planet in cm.
            hostPlanet (Planet): The host planet object.
        """
        self.midTransitOrbphase: float = midTransitOrbphase
        self.R: float = R
        self.a: float = a
        self.hostPlanet: Planet = hostPlanet

    def getOrbphase(self, orbphase: float) -> float:
        """Calculates the moon's orbital phase around its planet.

        This is scaled by the relative orbital periods of the planet and moon.

        Args:
            orbphase (float): The orbital phase of the host planet in radians.

        Returns:
            float: The orbital phase of the moon around its planet in radians.
        """
        orbphase_moon = self.midTransitOrbphase + orbphase * \
            np.sqrt((self.hostPlanet.a**3 * self.hostPlanet.M) /
                    (self.a**3 * self.hostPlanet.hostStar.M))
        return orbphase_moon

    def getPosition(self, orbphase: float) -> Tuple[float, float]:
        """Calculates the moon's (x, y) coordinates in the star's frame of reference.

        Args:
            orbphase (float): The orbital phase of the host planet in radians.

        Returns:
            Tuple[float, float]: The x and y coordinates of the moon in cm.
        """
        orbphase_moon = self.getOrbphase(orbphase)
        x_p, y_p = self.hostPlanet.getPosition(orbphase)
        x_moon = x_p + self.a * np.cos(orbphase_moon)
        y_moon = y_p + self.a * np.sin(orbphase_moon)
        return x_moon, y_moon

    def getLOSvelocity(self, orbphase: float) -> float:
        """Calculates the moon's total line-of-sight velocity.

        This is the sum of the planet's velocity and the moon's orbital velocity
        around the planet.

        Args:
            orbphase (float): The orbital phase of the host planet in radians.

        Returns:
            float: The moon's line-of-sight velocity in cm/s.
        """
        v_los_planet = self.hostPlanet.getLOSvelocity(orbphase)
        orbphase_moon = self.getOrbphase(orbphase)
        v_los = v_los_planet - \
            np.sin(orbphase_moon) * \
            np.sqrt(const.G * self.hostPlanet.M / self.a)
        return v_los

    def getDistanceFromMoon(self, x: float, phi: float, rho: float, orbphase: float) -> float:
        """Calculates the 3D distance from a point in space to the moon's center.

        Args:
            x (float): The coordinate along the line of sight in cm.
            phi (float): The azimuthal angle on the sky plane in radians.
            rho (float): The projected radial distance from the star's center in cm.
            orbphase (float): The host planet's orbital phase in radians.

        Returns:
            float: The distance from the point to the moon's center in cm.
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_moon, y_moon = self.getPosition(orbphase)
        r_fromMoon = np.sqrt((x - x_moon)**2 + (y - y_moon)**2 + z**2)
        return r_fromMoon


class AvailablePlanets:
    """A utility class to load and manage pre-defined planet and star data.

    This class reads data from 'stars.csv' and 'planets.csv' to populate
    a list of known exoplanetary systems.

    Attributes:
        stars (dict[str, Star]): A dictionary mapping star names to Star objects.
        planetList (list[Planet]): A list of available Planet objects.
    """

    def __init__(self) -> None:
        """Initializes the class by loading data from CSV files."""
        cwd = pathlib.Path(__file__).parent.resolve()
        self.stars: dict[str, Star] = {}
        with open(os.path.join(cwd, '../Resources/stars.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                R = float(row['R_sun']) * const.R_sun
                dR = float(row['dR_sun']) * const.R_sun
                M = float(row['M_sun']) * const.M_sun
                dM = float(row['dM_sun']) * const.M_sun
                T_eff = float(row['T_eff'])
                dT_eff = float(row['dT_eff'])
                log_g = float(row['log_g'])
                dlog_g = float(row['dlog_g'])
                Z = float(row['Fe_H'])
                dZ = float(row['dFe_H'])
                alpha = 0
                self.stars[name] = Star(
                    R, dR, M, dM, T_eff, dT_eff, log_g, dlog_g, Z, dZ, alpha)
        self.planetList: list[Planet] = []
        with open(os.path.join(cwd, '../Resources/planets.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                R = float(row['R_J']) * const.R_J
                M = float(row['M_J']) * const.M_J
                a = float(row['a_AU']) * const.AU
                transitDuration = float(row['transitDuration']) * 24
                hostStarName = row['hostStar']
                hostStar = self.stars.get(hostStarName)
                orbitalPeriod = float(row['P'])
                if hostStar is not None:
                    planet = Planet(name, R, M, a, hostStar,
                                    transitDuration, orbitalPeriod)
                    self.planetList.append(planet)
                else:
                    print(
                        f"Warning: Host star {hostStarName} not found for planet {name}")

    def listPlanetNames(self) -> list[str]:
        """Returns a list of names of all available planets.

        Returns:
            list[str]: A list of planet names.
        """
        planetNames: list[str] = []
        for planet in self.planetList:
            planetNames.append(planet.name)
        return planetNames

    def findPlanet(self, namePlanet: str) -> Optional[Planet]:
        """Finds a planet by its name.

        Args:
            namePlanet (str): The name of the planet to find.

        Returns:
            Optional[Planet]: The Planet object if found, otherwise None.
        """
        for planet in self.planetList:
            if planet.name == namePlanet:
                return planet
        print('System', namePlanet, 'was not found.')
        return None
