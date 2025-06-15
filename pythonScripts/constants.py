# coding=utf-8
"""
File which stores natural constants in cgs units.
Created on 2. June 2021 by Andrea Gebek.
"""

import numpy as np
from typing import List, Optional

e = 4.803e-10   # Elementary charge
m_e = 9.109e-28
c = 2.998e10
G = 6.674*10**(-8)
k_B = 1.381*10**(-16)
amu = 1.661*10**(-24)
R_J = 6.99e9        #Jupiter radius
M_J = 1.898e30      #Jupiter mass
M_E = 5.974e27      #Earth mass
R_sun = 6.96e10     # Solar radius
M_sun = 1.988e33    # Solar mass
R_Io = 1.822e8      # Io radius
euler_mascheroni = 0.57721 
AU = 1.496e13   # Conversion of one astronomical unit into cm

"""
Generally useful functions
"""
def calculateDopplerShift(v: float) -> float:
    beta = v / c
    shift = np.sqrt((1. - beta) / (1. + beta))
    return shift


"""
Available atoms/ions with their atomic masses
"""

class Species:
    def __init__(self, name: str, element: str, ionizationState: str, mass: float) -> None:
        self.name: str = name
        self.element: str = element
        self.ionizationState: str = ionizationState
        self.mass: float = mass

class SpeciesCollection:
    def __init__(self, speciesList: Optional[List[Species]] = None) -> None:
        if speciesList is None:
            self.speciesList: List[Species] = []
        else:
            self.speciesList: List[Species] = speciesList

    def findSpecies(self, nameSpecies: str) -> Optional[Species]:
        for species in self.speciesList:
            if species.name == nameSpecies:
                return species
        print('Species', nameSpecies, 'was not found.')
        return None

    def listSpeciesNames(self) -> List[str]:
        names: List[str] = []
        for species in self.speciesList:
            names.append(species.name)
        return names

    def addSpecies(self, species: Species) -> None:
        self.speciesList.append(species)


class AvailableSpecies(SpeciesCollection):
    def __init__(self) -> None:
        NaI = Species('NaI', 'Na', '1', 22.99 * amu)
        KI = Species('KI', 'K', '1', 39.0983 * amu)
        SiI = Species('SiI', 'Si', '1', 28.0855 * amu)
        SiII = Species('SiII', 'Si', '2', 28.0855 * amu)
        SiIII = Species('SiIII', 'Si', '3', 28.0855 * amu)
        SiIV = Species('SiIV', 'Si', '4', 28.0855 * amu)
        MgI = Species('MgI', 'Mg', '1', 24.305 * amu)
        MgII = Species('MgII', 'Mg', '2', 24.305 * amu)

        self.speciesList: List[Species] = [NaI, KI, SiI, SiII, SiIII, SiIV, MgI, MgII]
