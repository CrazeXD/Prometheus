"""
This file stores physical and astronomical constants in cgs units.
It also defines classes for managing atomic and ionic species used in
the simulations.

Created on 2. June 2021 by Andrea Gebek.
"""

from typing import List, Optional

import numpy as np

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
    """Calculates the relativistic Doppler shift factor.

    This factor multiplied by the rest wavelength gives the observed wavelength.

    Args:
        v (float): The line-of-sight velocity in cm/s. Positive for motion
            away from the observer (redshift).

    Returns:
        float: The dimensionless Doppler shift factor.
    """
    beta = v / c
    shift = np.sqrt((1. - beta) / (1. + beta))
    return shift


"""
Available atoms/ions with their atomic masses
"""

class Species:
    """Represents a single atomic or ionic species.

    Attributes:
        name (str): The common name of the species (e.g., 'NaI', 'SiII').
        element (str): The chemical symbol of the element (e.g., 'Na', 'Si').
        ionizationState (str): The ionization state as a string (e.g., '1' for
            neutral, '2' for singly ionized).
        mass (float): The mass of the species in grams.
    """
    def __init__(self, name: str, element: str, ionizationState: str, mass: float) -> None:
        """Initializes a Species object.

        Args:
            name (str): The common name of the species.
            element (str): The chemical symbol of the element.
            ionizationState (str): The ionization state as a string.
            mass (float): The mass of the species in grams.
        """
        self.name: str = name
        self.element: str = element
        self.ionizationState: str = ionizationState
        self.mass: float = mass

class SpeciesCollection:
    """A container for a list of Species objects.

    Provides methods for finding, listing, and adding species.

    Attributes:
        speciesList (List[Species]): A list of Species objects.
    """
    def __init__(self, speciesList: Optional[List[Species]] = None) -> None:
        """Initializes the SpeciesCollection.

        Args:
            speciesList (Optional[List[Species]]): An optional initial list of
                Species objects. Defaults to an empty list.
        """
        if speciesList is None:
            self.speciesList: List[Species] = []
        else:
            self.speciesList: List[Species] = speciesList

    def findSpecies(self, nameSpecies: str) -> Optional[Species]:
        """Finds a species in the collection by its name.

        Args:
            nameSpecies (str): The name of the species to find.

        Returns:
            Optional[Species]: The Species object if found, otherwise None.
        """
        for species in self.speciesList:
            if species.name == nameSpecies:
                return species
        print('Species', nameSpecies, 'was not found.')
        return None

    def listSpeciesNames(self) -> List[str]:
        """Returns a list of names of all species in the collection.

        Returns:
            List[str]: A list of species names.
        """
        names: List[str] = []
        for species in self.speciesList:
            names.append(species.name)
        return names

    def addSpecies(self, species: Species) -> None:
        """Adds a Species object to the collection.

        Args:
            species (Species): The species to add.
        """
        self.speciesList.append(species)


class AvailableSpecies(SpeciesCollection):
    """A pre-populated collection of common astrophysical species.

    Inherits from SpeciesCollection and initializes with a default set of
    atoms and ions relevant for exoplanet atmosphere studies.
    """
    def __init__(self) -> None:
        """Initializes and populates the list of available species."""
        NaI = Species('NaI', 'Na', '1', 22.99 * amu)
        KI = Species('KI', 'K', '1', 39.0983 * amu)
        SiI = Species('SiI', 'Si', '1', 28.0855 * amu)
        SiII = Species('SiII', 'Si', '2', 28.0855 * amu)
        SiIII = Species('SiIII', 'Si', '3', 28.0855 * amu)
        SiIV = Species('SiIV', 'Si', '4', 28.0855 * amu)
        MgI = Species('MgI', 'Mg', '1', 24.305 * amu)
        MgII = Species('MgII', 'Mg', '2', 24.305 * amu)

        self.speciesList: List[Species] = [NaI, KI, SiI, SiII, SiIII, SiIV, MgI, MgII]
