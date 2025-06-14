"""
This file has properties and methods for the
celestial bodies (star, planet, moon) involved
in the calculation.
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
    def __init__(self, R: float, M: float, T_eff: float, log_g: float, Z: float, alpha: float) -> None:
        self.R: float = R
        self.M: float = M
        self.T_eff: float = T_eff
        self.log_g: float = log_g
        self.Z: float = Z
        self.alpha: float = alpha
        self.CLV_u1: float = 0.
        self.CLV_u2: float = 0.
        self.vsiniStarrot: float = 0.
        self.phiStarrot: float = 0.
        self.Fstar_function: Optional[Callable[[Any], Any]] = None

    def addCLVparameters(self, CLV_u1: float, CLV_u2: float) -> None:
        self.CLV_u1 = CLV_u1
        self.CLV_u2 = CLV_u2

    def addRMparameters(self, vsiniStarrot: float, phiStarrot: float) -> None:
        self.vsiniStarrot = vsiniStarrot
        self.phiStarrot = phiStarrot

    def getSurfaceVelocity(self, phi: float, rho: float) -> float:
        v_los = self.vsiniStarrot * rho / self.R * np.cos(phi - self.phiStarrot)
        return v_los

    @staticmethod
    def round_to_grid(grid: np.ndarray, value: float) -> float:
        diff = np.subtract(value, grid)
        arg = np.argmin(np.abs(diff))
        return grid[arg]

    def getSpectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Query a PHOENIX photosphere model, either from disk or from the PHOENIX website
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
        #These contain the acceptable values.
        T_grid = np.concatenate((np.arange(2300,7100,100),np.arange(7200,12200,200)))
        log_g_grid = np.arange(0,6.5,0.5)
        Z_grid = np.concatenate((np.arange(-4,-1,1),np.arange(-1.5,1.5,0.5)))
        alpha_grid = np.arange(0,1.6,0.2)-0.2
        
        T_a = Star.round_to_grid(T_grid, self.T_eff)
        log_g_a = Star.round_to_grid(log_g_grid, self.log_g)
        Z_a = Star.round_to_grid(Z_grid, self.Z)
        alpha_a = Star.round_to_grid(alpha_grid, self.alpha)

        #This is where phoenix spectra are located.
        root = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'

        #We assemble a combination of strings to parse the user input into the URL,
        z_string = '{:.1f}'.format(float(Z_a))
        if Z_a > 0:
            z_string = '+' + z_string
        elif Z_a == 0:
            z_string = '-' + z_string
        else:
            z_string = z_string
        a_string=''
        if alpha_a > 0:
            a_string ='.Alpha=+'+'{:.2f}'.format(float(alpha_a))
        if alpha_a < 0:
            a_string ='.Alpha='+'{:.2f}'.format(float(alpha_a))
        t_string = str(int(T_a))
        if T_a < 10000:
            t_string = '0'+t_string
        g_string = '-'+'{:.2f}'.format(float(log_g_a))

        #These are URLS for the input files.
        waveurl = root+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
        specurl = root+'PHOENIX-ACES-AGSS-COND-2011/Z'+z_string+a_string+'/lte'+t_string+g_string+z_string+a_string+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

        #These are the output filenames, they will also be returned so that the wrapper
        #of this function can take them in.
        wavename = 'WAVE.fits'
        specname = 'lte'+t_string+g_string+z_string+a_string+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

        #Download PHOENIX spectra:
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

        return (w * 1e-8, F / np.pi) # Conversion to cgs-units. Note that Jens divides F by
        # a seemingly random factor of pi, but this should not bother the transit calculations here.

    def calculateCLV(self, rho: float) -> float:
        arg = 1. - np.sqrt(1. - rho**2 / self.R**2)
        return 1. - self.CLV_u1 * arg - self.CLV_u2 * arg**2

    def calculateRM(self, phi: float, rho: float, wavelength: np.ndarray) -> np.ndarray:
        v_los = self.getSurfaceVelocity(phi, rho)
        shift = const.calculateDopplerShift(v_los)
        F_shifted = 10.**self.Fstar_function(wavelength / shift)
        return F_shifted

    def addFstarFunction(self, wavelength: np.ndarray) -> None:
        PHOENIX_output = self.getSpectrum()
        w_star = PHOENIX_output[0]
        w_max = np.max(wavelength) * const.calculateDopplerShift(-self.vsiniStarrot)
        w_min = np.min(wavelength) * const.calculateDopplerShift(self.vsiniStarrot)
        SEL = (w_star >= w_min) * (w_star <= w_max)
        minArg = max(min(np.argwhere(SEL)).item() - 1, 0)
        maxArg = max(np.argwhere(SEL)).item() + 2
        w_starSEL = w_star[minArg:maxArg]
        F_0 = PHOENIX_output[1][minArg:maxArg]
        Fstar_function = interp1d(w_starSEL, np.log10(F_0), kind='linear')
        self.Fstar_function = Fstar_function

    def getFstarIntegrated(self, wavelength: np.ndarray, grid: Any) -> Tuple[np.ndarray, np.ndarray]:
        if self.vsiniStarrot == 0.:
            FstarIntegrated = np.pi * self.R**2 * (1. - self.CLV_u1 / 3. - self.CLV_u2 / 6.) * np.ones_like(wavelength)
            upperTerm = 0.5 * (-self.CLV_u2 * self.R**2 - self.CLV_u1 * self.R**2 + self.R**2)
            term1 = -4. * self.R**2 * self.CLV_u1 * (1. - grid.rho_border**2 / self.R**2)**1.5
            term2 = self.R**2 * self.CLV_u2 * (6 * grid.rho_border**2 / self.R**2 + 8. * (1. - grid.rho_border**2 / self.R**2)**1.5 - 3. * (self.R**2 - grid.rho_border**2)**2 / self.R**4)
            lowerTerm = 1. / 12. * (term1 - term2 - 6. * self.CLV_u1 * grid.rho_border**2 + 6. * grid.rho_border**2)
            FstarUpper = 2. * np.pi * (upperTerm - lowerTerm) * np.ones_like(wavelength)
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
        if self.vsiniStarrot == 0.:
            Fstar = np.ones_like(wavelength) * self.calculateCLV(rho)
        else:
            Fstar = self.calculateRM(phi, rho, wavelength)
            Fstar *= self.calculateCLV(rho)
        return Fstar


class Planet:
    def __init__(self, name: str, R: float, M: float, a: float, hostStar: Star, transitDuration: float, orbitalPeriod: float) -> None:
        self.name: str = name
        self.R: float = R
        self.M: float = M
        self.a: float = a
        self.hostStar: Star = hostStar
        self.transitDuration: float = transitDuration
        self.orbitalPeriod: float = orbitalPeriod

    def getPosition(self, orbphase: float) -> Tuple[float, float]:
        x_p = self.a * np.cos(orbphase)
        y_p = self.a * np.sin(orbphase)
        return x_p, y_p

    def getLOSvelocity(self, orbphase: float) -> float:
        v_los = -np.sin(orbphase) * np.sqrt(const.G * self.hostStar.M / self.a)
        return v_los

    def getDistanceFromPlanet(self, x: float, phi: float, rho: float, orbphase: float) -> float:
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_p, y_p = self.getPosition(orbphase)
        r_fromPlanet = np.sqrt((x - x_p)**2 + (y - y_p)**2 + z**2)
        return r_fromPlanet

    def getTorusCoords(self, x: float, phi: float, rho: float, orbphase: float) -> Tuple[float, float]:
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_p, y_p = self.getPosition(orbphase)
        a = np.sqrt((x - x_p)**2 + (y - y_p)**2)
        return a, z


class Moon:
    def __init__(self, midTransitOrbphase: float, R: float, a: float, hostPlanet: Planet) -> None:
        self.midTransitOrbphase: float = midTransitOrbphase
        self.R: float = R
        self.a: float = a
        self.hostPlanet: Planet = hostPlanet

    def getOrbphase(self, orbphase: float) -> float:
        orbphase_moon = self.midTransitOrbphase + orbphase * np.sqrt((self.hostPlanet.a**3 * self.hostPlanet.M) / (self.a**3 * self.hostPlanet.hostStar.M))
        return orbphase_moon

    def getPosition(self, orbphase: float) -> Tuple[float, float]:
        orbphase_moon = self.getOrbphase(orbphase)
        x_p, y_p = self.hostPlanet.getPosition(orbphase)
        x_moon = x_p + self.a * np.cos(orbphase_moon)
        y_moon = y_p + self.a * np.sin(orbphase_moon)
        return x_moon, y_moon

    def getLOSvelocity(self, orbphase: float) -> float:
        v_los_planet = self.hostPlanet.getLOSvelocity(orbphase)
        orbphase_moon = self.getOrbphase(orbphase)
        v_los = v_los_planet - np.sin(orbphase_moon) * np.sqrt(const.G * self.hostPlanet.M / self.a)
        return v_los

    def getDistanceFromMoon(self, x: float, phi: float, rho: float, orbphase: float) -> float:
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_moon, y_moon = self.getPosition(orbphase)
        r_fromMoon = np.sqrt((x - x_moon)**2 + (y - y_moon)**2 + z**2)
        return r_fromMoon


class AvailablePlanets:
    def __init__(self) -> None:
        cwd = pathlib.Path(__file__).parent.resolve()
        self.stars: dict[str, Star] = {}
        with open(os.path.join(cwd, '../Resources/stars.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                R = float(row['R_sun']) * const.R_sun
                M = float(row['M_sun']) * const.M_sun
                T_eff = float(row['T_eff'])
                log_g = float(row['log_g'])
                Z = float(row['Fe_H'])
                alpha = float(row['alpha'])
                self.stars[name] = Star(R, M, T_eff, log_g, Z, alpha)
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
                    planet = Planet(name, R, M, a, hostStar, transitDuration, orbitalPeriod)
                    self.planetList.append(planet)
                else:
                    print(f"Warning: Host star {hostStarName} not found for planet {name}")

    def listPlanetNames(self) -> list[str]:
        planetNames: list[str] = []
        for planet in self.planetList:
            planetNames.append(planet.name)
        return planetNames

    def findPlanet(self, namePlanet: str) -> Optional[Planet]:
        for planet in self.planetList:
            if planet.name == namePlanet:
                return planet
        print('System', namePlanet, 'was not found.')
        return None
