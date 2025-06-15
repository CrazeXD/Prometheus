# coding=utf-8
"""
This file stores various functions related
to the properties of the gas (e.g. number densities, velocities,
absorption cross sections).
Created on 19. October 2021 by Andrea Gebek.
"""

import numpy as np
from scipy.special import erf, voigt_profile
import os
import h5py
from scipy.interpolate import interp1d
from . import constants as const
from . import geometryHandler as geom
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy
from scipy.ndimage import gaussian_filter as gauss
from typing import List, Callable, Optional, Tuple, Any, Union

lineListPath: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Resources/LineList.txt'
molecularLookupPath: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/molecularResources/'

class CollisionalAtmosphere:
    def __init__(self, T: float, P_0: float):
        self.T: float = T
        self.P_0: float = P_0
        self.constituents: List[Union['AtmosphericConstituent', 'MolecularConstituent']] = []
        self.hasMoon: bool = False

    def getReferenceNumberDensity(self) -> float:
        n_0 = self.P_0 / (const.k_B * self.T)
        return n_0

    def getVelDispersion(self, m: float) -> float:
        sigma_v = np.sqrt(self.T * const.k_B / m)
        return sigma_v

    def addConstituent(self, speciesName: str, chi: float) -> None:
        species = const.AvailableSpecies().findSpecies(speciesName)
        m = species.mass
        sigma_v = self.getVelDispersion(m)
        constituent = AtmosphericConstituent(species, chi, sigma_v)
        self.constituents.append(constituent)

    def addMolecularConstituent(self, speciesName: str, chi: float) -> None:
        constituent = MolecularConstituent(speciesName, chi)
        self.constituents.append(constituent)


class BarometricAtmosphere(CollisionalAtmosphere):
    def __init__(self, T: float, P_0: float, mu: float, planet: Any):
        super().__init__(T, P_0)
        self.mu: float = mu
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = BarometricAtmosphere.getReferenceNumberDensity(self)
        H = const.k_B * self.T * self.planet.R**2 / (const.G * self.mu * self.planet.M)
        n = n_0 * np.exp((self.planet.R - r) / H) * np.heaviside(r - self.planet.R, 1.)
        return n

class HydrostaticAtmosphere(CollisionalAtmosphere):
    def __init__(self, T: float, P_0: float, mu: float, planet: Any):
        super().__init__(T, P_0)
        self.mu: float = mu
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = HydrostaticAtmosphere.getReferenceNumberDensity(self)
        Jeans_0 = const.G * self.mu * self.planet.M / (const.k_B * self.T * self.planet.R)
        Jeans = const.G * self.mu * self.planet.M / (const.k_B * self.T * r) * np.heaviside(r - self.planet.R, 1.) 
        n = n_0 * np.exp(Jeans - Jeans_0)
        return n

class PowerLawAtmosphere(CollisionalAtmosphere):
    def __init__(self, T: float, P_0: float, q: float, planet: Any):
        super().__init__(T, P_0)
        self.q: float = q
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = PowerLawAtmosphere.getReferenceNumberDensity(self)
        n = n_0* (self.planet.R / r)**self.q * np.heaviside(r - self.planet.R, 1.)
        return n


class EvaporativeExosphere:
    def __init__(self, N: float):
        self.N: float = N
        self.hasMoon: bool = False

    def addConstituent(self, speciesName: str, sigma_v: float) -> None:
        species = const.AvailableSpecies().findSpecies(speciesName)
        constituent = AtmosphericConstituent(species, 1., sigma_v)
        self.constituents: List[AtmosphericConstituent] = [constituent]

    def addMolecularConstituent(self, speciesName: str, T: float) -> None:
        constituent = MolecularConstituent(speciesName, 1.0)
        self.constituents: List[MolecularConstituent] = [constituent]
        self.T: float = T


class PowerLawExosphere(EvaporativeExosphere):
    def __init__(self, N: float, q: float, planet: Any):
        super().__init__(N)
        self.q: float = q
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.planet.R**3) * self.N
        n = n_0 * (self.planet.R / r)**self.q * np.heaviside(r - self.planet.R, 1.)
        return n

class MoonExosphere(EvaporativeExosphere):
    def __init__(self, N: float, q: float, moon: Any):
        super().__init__(N)
        self.q: float = q
        self.moon: Any = moon
        self.hasMoon: bool = True
        self.planet: Any = moon.hostPlanet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        r = self.moon.getDistanceFromMoon(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.moon.R**3) * self.N
        n = n_0 * (self.moon.R / r)**self.q * np.heaviside(r - self.moon.R, 1.)
        return n
    
class TidallyHeatedMoon(EvaporativeExosphere):
    def __init__(self, q: float, moon: Any):
        self.q: float = q
        self.moon: Any = moon
        self.hasMoon: bool = True
        self.planet: Any = moon.hostPlanet

    def addSourceRateFunction(self, filename: str, tau_photoionization: float, mass_absorber: float) -> None:
        Mdot = np.loadtxt(filename)
        Mdot = np.concatenate((Mdot, Mdot[::-1]))
        phi_moon = np.linspace(0., 2. * np.pi, len(Mdot))
        N_function = interp1d(phi_moon, np.log10(Mdot * tau_photoionization / mass_absorber))
        self.N_function: Callable[[float], float] = N_function

    def calculateAbsorberNumber(self, orbphase: float) -> float:
        orbphase_moon = self.moon.getOrbphase(orbphase) % (2. * np.pi)
        N = 10**self.N_function(orbphase_moon)
        return N

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        N = self.calculateAbsorberNumber(orbphase)
        r = self.moon.getDistanceFromMoon(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.moon.R**3) * N
        n = n_0 * (self.moon.R / r)**self.q * np.heaviside(r - self.moon.R, 1.)
        return n

class TorusExosphere(EvaporativeExosphere):
    def __init__(self, N: float, a_torus: float, v_ej: float, planet: Any):
        super().__init__(N)
        self.a_torus: float = a_torus
        self.v_ej: float = v_ej
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        a, z = self.planet.getTorusCoords(x, phi, rho, orbphase)
        v_orbit = np.sqrt(const.G * self.planet.M / self.a_torus)
        H_torus = self.a_torus * self.v_ej / v_orbit
        n_a = np.exp(-((a - self.a_torus) / (4. * H_torus))**2)
        n_z = np.exp(-(z / H_torus)**2)
        term1 = 8. * H_torus**2 * np.exp(-self.a_torus**2 / (16. * H_torus**2))
        term2 = 2. * np.sqrt(np.pi) * self.a_torus * H_torus * (erf(self.a_torus / (4. * H_torus)) + 1.)
        n_0 = 1. / (2. * np.pi**1.5 * H_torus * (term1 + term2)) * self.N
        n = n_0 * np.multiply(n_a, n_z) 
        return n


class SerpensExosphere(EvaporativeExosphere):
    def __init__(self, filename: str, N: float, planet: Any, sigmaSmoothing: float):
        super().__init__(N)
        self.filename: str = filename
        self.planet: Any = planet
        self.sigmaSmoothing: float = sigmaSmoothing

    def addInterpolatedDensity(self, spatialGrid: Any) -> None:
        serpensOutput = np.loadtxt(self.filename) * 1e2
        particlePos = serpensOutput[:, 0:3]
        xBins = spatialGrid.constructXaxis(midpoints = False)
        yBins = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps) + 1)
        zBins = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps) + 1)
        cellVolume = (xBins[1] - xBins[0]) * (yBins[1] - yBins[0]) * (zBins[1] - zBins[0])
        n_histogram = np.histogramdd(particlePos, bins = [xBins, yBins, zBins])[0] * self.N / (np.size(particlePos, axis = 0) * cellVolume)
        if self.sigmaSmoothing > 0.:
            n_histogram = gauss(n_histogram, sigma = self.sigmaSmoothing)
        print('Sum over all particles, potentially smoothed with a Gaussian:', np.sum(n_histogram) * cellVolume)
        xPoints = spatialGrid.constructXaxis()
        yPoints = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps), endpoint = False) + 2. * spatialGrid.rho_border / (4. * spatialGrid.rho_steps)
        zPoints = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps), endpoint = False) + 2. * spatialGrid.rho_border / (4. * spatialGrid.rho_steps)
        x, y, z = np.meshgrid(xPoints, yPoints, zPoints, indexing = 'ij')
        SEL = ((y**2 + z**2) > self.planet.R**2) * ((y**2 + z**2) < self.planet.hostStar.R**2)
        print('Sum over all particles outside of the planetary disk but inside the stellar disk:', np.sum(n_histogram[SEL]) * cellVolume)
        n_function = RegularGridInterpolator((xPoints, yPoints, zPoints), n_histogram)
        self.InterpolatedDensity: Callable[[np.ndarray], np.ndarray] = n_function

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        coordArray = np.array([x, np.repeat(y, np.size(x)), np.repeat(z, np.size(x))]).T
        n = self.InterpolatedDensity(coordArray)
        return n



"""
Calculate absorption cross sections
"""

class AtmosphericConstituent:
    def __init__(self, species: Any, chi: float, sigma_v: float):
        self.isMolecule: bool = False
        self.species: Any = species
        self.chi: float = chi
        self.sigma_v: float = sigma_v
        self.wavelengthGridRefinement: float = 10.
        self.wavelengthGridExtension: float = 0.01
        self.lookupOffset: float = 1e-50

    def getLineParameters(self, wavelength: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lineList = np.loadtxt(lineListPath, dtype = str, usecols = (0, 1, 2, 3, 4), skiprows = 1)
        line_wavelength = np.array([x[1:-1] for x in lineList[:, 2]])
        line_A = np.array([x[1:-1] for x in lineList[:, 3]])
        line_f = np.array([x[1:-1] for x in lineList[:, 4]])
        SEL_COMPLETE = (line_wavelength != '') * (line_A != '') * (line_f != '') 
        SEL_SPECIES = (lineList[:, 0] == self.species.element) * (lineList[:, 1] == self.species.ionizationState)
        line_wavelength = line_wavelength[SEL_SPECIES * SEL_COMPLETE].astype(float) * 1e-8
        line_gamma = line_A[SEL_SPECIES * SEL_COMPLETE].astype(float) / (4. * np.pi)
        line_f = line_f[SEL_SPECIES * SEL_COMPLETE].astype(float)
        SEL_WAVELENGTH = (line_wavelength > min(wavelength)) * (line_wavelength < max(wavelength))
        return line_wavelength[SEL_WAVELENGTH], line_gamma[SEL_WAVELENGTH], line_f[SEL_WAVELENGTH]

    def calculateVoigtProfile(self, wavelength: np.ndarray) -> np.ndarray:
        line_wavelength, line_gamma, line_f = self.getLineParameters(wavelength)
        sigma_abs = np.zeros_like(wavelength)
        for idx in range(len(line_wavelength)):
            lineProfile = voigt_profile(const.c / wavelength - const.c / line_wavelength[idx], self.sigma_v / line_wavelength[idx], line_gamma[idx])
            sigma_abs += np.pi * (const.e)**2 / (const.m_e * const.c) * line_f[idx] * lineProfile
        return sigma_abs

    def constructLookupFunction(self, wavelengthGrid: 'WavelengthGrid') -> Callable[[np.ndarray], np.ndarray]:
        wavelengthGridRefined = deepcopy(wavelengthGrid)
        wavelengthGridRefined.resolutionHigh /= self.wavelengthGridRefinement
        wavelengthGridRefined.lower_w *= (1. - self.wavelengthGridExtension)
        wavelengthGridRefined.upper_w *= (1. + self.wavelengthGridExtension)
        wavelengthRefined = wavelengthGridRefined.constructWavelengthGridSingle(self)
        sigma_abs = self.calculateVoigtProfile(wavelengthRefined)
        lookupFunction = interp1d(wavelengthRefined, np.log10(sigma_abs + self.lookupOffset), bounds_error = False, fill_value = np.log10(self.lookupOffset))
        return lookupFunction

    def addLookupFunctionToConstituent(self, wavelengthGrid: 'WavelengthGrid') -> None:
        lookupFunction = self.constructLookupFunction(wavelengthGrid)
        self.lookupFunction: Callable[[np.ndarray], np.ndarray] = lookupFunction

    def getSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        sigma_absFlattened = 10**self.lookupFunction(wavelength.flatten()) - self.lookupOffset
        sigma_abs = sigma_absFlattened.reshape(wavelength.shape)
        return sigma_abs

class MolecularConstituent:
    def __init__(self, moleculeName: str, chi: float):
        self.isMolecule: bool = True
        self.lookupOffset: float = 1e-50
        self.moleculeName: str = moleculeName
        self.chi: float = chi

    def constructLookupFunction(self) -> Callable[[np.ndarray], np.ndarray]:
        with h5py.File(molecularLookupPath + self.moleculeName + '.h5', 'r+') as f:
            P = f['p'][:] * 10.
            T = f['t'][:]
            wavelength = 1. / f['bin_edges'][:][::-1]
            sigma_abs = f['xsecarr'][:][:, :, ::-1]
            lookupFunction = RegularGridInterpolator((P, T, wavelength), np.log10(sigma_abs + self.lookupOffset), bounds_error = False, fill_value = np.log10(self.lookupOffset))
            return lookupFunction

    def addLookupFunctionToConstituent(self) -> None:
        lookupFunction = self.constructLookupFunction()
        self.lookupFunction: Callable[[np.ndarray], np.ndarray] = lookupFunction

    def getSigmaAbs(self, P: np.ndarray, T: float, wavelength: np.ndarray) -> np.ndarray:
        wavelengthFlattened = wavelength.flatten()
        TFlattened = np.full_like(wavelengthFlattened, T)
        PFlattened = np.repeat(np.clip(P, a_min = 1e-4, a_max = None), np.size(wavelength, axis = 1))
        inputArray = np.array([PFlattened, TFlattened, wavelengthFlattened]).T
        sigma_absFlattened = 10**self.lookupFunction(inputArray) - self.lookupOffset
        sigma_abs = sigma_absFlattened.reshape(wavelength.shape)
        return sigma_abs


class Atmosphere:
    def __init__(self, densityDistributionList: List[Any], hasOrbitalDopplerShift: bool):
        self.densityDistributionList: List[Any] = densityDistributionList
        self.hasOrbitalDopplerShift: bool = hasOrbitalDopplerShift

    @staticmethod
    def getAbsorberNumberDensity(densityDistribution: Any, chi: float, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        n_total = densityDistribution.calculateNumberDensity(x, phi, rho, orbphase)
        n_abs = n_total * chi
        return n_abs

    def getAbsorberVelocityField(self, densityDistribution: Any, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        v_los = np.zeros_like(x)
        if self.hasOrbitalDopplerShift:
            if not densityDistribution.hasMoon:
                v_los += densityDistribution.planet.getLOSvelocity(orbphase)
            else:
                v_los += densityDistribution.moon.getLOSvelocity(orbphase)
        return v_los

    def getLOSopticalDepth(self, x: np.ndarray, phi: float, rho: float, orbphase: float, wavelength: np.ndarray, delta_x: float) -> np.ndarray:
        kappa = np.zeros((len(x), len(wavelength)))
        for densityDistribution in self.densityDistributionList:
            for constituent in densityDistribution.constituents:
                v_los = self.getAbsorberVelocityField(densityDistribution, x, phi, rho, orbphase)
                shift = const.calculateDopplerShift(-v_los)
                wavelengthShifted = np.tensordot(shift, wavelength, axes = 0)
                if constituent.isMolecule:
                    n_tot = densityDistribution.calculateNumberDensity(x, phi, rho, orbphase)
                    n_abs = n_tot * constituent.chi
                    T = densityDistribution.T
                    P = n_tot * const.k_B * T
                    sigma_abs = constituent.getSigmaAbs(P, T, wavelengthShifted)
                else:
                    n_abs = Atmosphere.getAbsorberNumberDensity(densityDistribution, constituent.chi, x, phi, rho, orbphase)
                    sigma_abs = constituent.getSigmaAbs(wavelengthShifted)
                kappa += np.tile(n_abs, (len(wavelength), 1)).T * sigma_abs
        LOStau = np.sum(kappa, axis = 0) * delta_x
        return LOStau

class WavelengthGrid:
    def __init__(self, lower_w: float, upper_w: float, widthHighRes: float, resolutionLow: float, resolutionHigh: float):
        self.lower_w: float = lower_w
        self.upper_w: float = upper_w
        self.widthHighRes: float = widthHighRes
        self.resolutionLow: float = resolutionLow
        self.resolutionHigh: float = resolutionHigh

    def arangeWavelengthGrid(self, linesList: List[float]) -> np.ndarray:
        peaks = np.sort(np.unique(linesList))
        diff = np.concatenate(([np.inf], np.diff(peaks), [np.inf]))
        if len(peaks) == 0:
            print('WARNING: No absorption lines from atoms/ions in the specified wavelength range!')
            return np.arange(self.lower_w, self.upper_w, self.resolutionLow)
        HighResBorders: Tuple[List[float], List[float]] = ([], [])
        for idx, peak in enumerate(peaks):
            if diff[idx] > self.widthHighRes:
                HighResBorders[0].append(peak - self.widthHighRes / 2.)
            if diff[idx + 1] > self.widthHighRes:
                HighResBorders[1].append(peak + self.widthHighRes / 2.)
        grid: List[np.ndarray] = []
        for idx in range(len(HighResBorders[0])):
            grid.append(np.arange(HighResBorders[0][idx], HighResBorders[1][idx], self.resolutionHigh))
            if idx == 0:
                if self.lower_w < HighResBorders[0][0]:
                    grid.append(np.arange(self.lower_w, HighResBorders[0][0], self.resolutionLow))
                if len(HighResBorders[0]) == 1 and self.upper_w > HighResBorders[1][-1]:
                    grid.append(np.arange(HighResBorders[1][0], self.upper_w, self.resolutionLow))
            elif idx == len(HighResBorders[0]) - 1:
                grid.append(np.arange(HighResBorders[1][idx - 1], HighResBorders[0][idx], self.resolutionLow))
                if self.upper_w > HighResBorders[1][-1]:
                    grid.append(np.arange(HighResBorders[1][-1], self.upper_w, self.resolutionLow))
            else:
                grid.append(np.arange(HighResBorders[1][idx - 1], HighResBorders[0][idx], self.resolutionLow))
        wavelengthGrid = np.sort(np.concatenate(grid))
        return wavelengthGrid

    def constructWavelengthGridSingle(self, constituent: AtmosphericConstituent) -> np.ndarray:
        linesList = constituent.getLineParameters(np.array([self.lower_w, self.upper_w]))[0]
        return self.arangeWavelengthGrid(linesList)

    def constructWavelengthGrid(self, densityDistributionList: List[Any]) -> np.ndarray:
        linesList: List[float] = []
        for densityDistribution in densityDistributionList:
            for constituent in densityDistribution.constituents:
                if constituent.isMolecule:
                    continue
                lines_w = constituent.getLineParameters(np.array([self.lower_w, self.upper_w]))[0]
                linesList.extend(lines_w)
        if len(linesList) == 0:
            return np.arange(self.lower_w, self.upper_w, self.resolutionLow)
        return self.arangeWavelengthGrid(linesList)
        

class Transit:
    def __init__(self, atmosphere: Atmosphere, wavelengthGrid: WavelengthGrid, spatialGrid: Any):
        self.atmosphere: Atmosphere = atmosphere
        self.wavelengthGrid: WavelengthGrid = wavelengthGrid
        self.spatialGrid: Any = spatialGrid
        self.planet: Any = self.atmosphere.densityDistributionList[0].planet

    def addWavelength(self) -> None:
        wavelength = self.wavelengthGrid.constructWavelengthGrid(self.atmosphere.densityDistributionList)
        self.wavelength: np.ndarray = wavelength

    def checkBlock(self, phi: float, rho: float, orbphase: float) -> bool:
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
        Fstar = self.planet.hostStar.getFstar(phi, rho, self.wavelength)
        F_out = rho * Fstar * self.wavelength / self.wavelength
        if self.checkBlock(phi, rho, orbphase):
            F_in = np.zeros_like(self.wavelength)
            return F_in, F_out
        x = self.spatialGrid.constructXaxis()
        delta_x = self.spatialGrid.getDeltaX()
        tau = self.atmosphere.getLOSopticalDepth(x, phi, rho, orbphase, self.wavelength, delta_x)
        F_in = rho * Fstar * np.exp(-tau)
        return F_in, F_out

    def sumOverChords(self) -> np.ndarray:
        chordGrid = self.spatialGrid.getChordGrid()
        F_in: List[np.ndarray] = []
        F_out: List[np.ndarray] = []
        for chord in chordGrid:
            Fsingle_in, Fsingle_out = self.evaluateChord(chord[0], chord[1], chord[2])
            F_in.append(Fsingle_in)
            F_out.append(Fsingle_out)
        F_in = np.array(F_in).reshape((self.spatialGrid.phi_steps * self.spatialGrid.rho_steps, self.spatialGrid.orbphase_steps, len(self.wavelength)))
        F_out = np.array(F_out).reshape((self.spatialGrid.phi_steps * self.spatialGrid.rho_steps, self.spatialGrid.orbphase_steps, len(self.wavelength)))
        R = np.sum(F_in, axis = 0) / np.sum(F_out, axis = 0)
        return R
