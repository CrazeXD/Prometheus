# 🔥 Prometheus

**Prometheus** is a Python-based radiative transfer forward model for simulating exoplanet transit spectroscopy. It computes wavelength-dependent transit depths across orbital phases for a variety of atmospheric and exospheric scenarios.

---

## Features

- **Multiple atmosphere/exosphere models:**
  - Barometric atmosphere
  - Hydrostatic atmosphere
  - Power-law atmosphere / exosphere
  - Exomoon exosphere
  - Torus exosphere
  - SERPENS particle simulation integration
- **Multi-species support** — atoms, ions, and molecules
- **Doppler orbital motion** correction
- **Parallelized computation** via Python `multiprocessing`
- **Configurable memory limits** to run on constrained hardware
- **High/low resolution wavelength grids** for flexible spectral sampling

---

## Requirements

- Python 3.8+
- `numpy`

---

## Installation

```bash
git clone https://github.com/CrazeXD/Prometheus.git
cd Prometheus
pip install numpy
```

---

## Usage

### 1. Create a Setup File

Run the interactive setup script to generate a JSON configuration file:

```bash
python prometheus.py setup
```

This creates a `.txt` setup file under `setupFiles/`.

### 2. Run the Forward Model

```bash
python prometheus.py <setup_name>
```

Optionally, limit RAM usage (default is 2 GB):

```bash
python prometheus.py <setup_name> --max-memory 4.0
```

### 3. Output

Results are saved to `output/<setup_name>.txt` with the format:

- **Row 1:** Orbital phases (in units of full orbit)
- **Remaining rows:** Wavelength [cm] (column 1), Transit depth R(orbital phase, wavelength) (remaining columns)

---

## Project Structure

```
Prometheus/
├── prometheus.py          # Main entry point
├── mainRetrieval.py       # Retrieval utilities
├── profile_prometheus.py  # Profiling script
├── pythonScripts/
│   ├── setup.py           # Interactive setup file generator
│   ├── gasProperties.py   # Atmosphere/exosphere models & transit computation
│   ├── celestialBodies.py # Planet & moon definitions
│   ├── geometryHandler.py # Spatial grid and chord geometry
│   ├── memoryHandler.py   # Memory-aware chunk processing
│   └── constants.py       # Physical constants & available species
├── Resources/             # Cross-section and species data
├── docs/                  # Documentation
└── setupFiles/            # Generated JSON configuration files (gitignored)
```

---

## Atmospheric Scenarios

| Scenario | Description |
|---|---|
| `barometric` | Isothermal atmosphere with exponential pressure profile |
| `hydrostatic` | Hydrostatic equilibrium atmosphere |
| `powerLaw` | Power-law density profile for tenuous exospheres |
| `exomoon` | Exosphere sourced from an orbiting moon |
| `torus` | Neutral gas torus around the planet |
| `serpens` | Interpolated density from SERPENS particle simulation output |

---

## License

This project is licensed under the GPL-3.0 License. See [LICENSE](LICENSE) for details.
