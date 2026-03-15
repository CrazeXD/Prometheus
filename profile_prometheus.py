"""
Profile Prometheus to understand where time is spent.
"""
import time
import numpy as np
import Prometheus.pythonScripts.geometryHandler as geom
import Prometheus.pythonScripts.celestialBodies as bodies
import Prometheus.pythonScripts.gasProperties as gasprop
import Prometheus.pythonScripts.constants as const

print("=" * 70)
print("Profiling Prometheus Performance")
print("=" * 70)

# Setup
wavelengthGrid = gasprop.WavelengthGrid(5888e-8, 5900e-8, 2e-8, 5e-9, 2e-10)
HD189733b = bodies.AvailablePlanets().findPlanet('HD189733b')
spatialGrid = geom.Grid(HD189733b.a, 5. * HD189733b.R, 20, HD189733b.hostStar.R,
                        35, 35, 0.25, 20)
HD189733b.hostStar.addCLVparameters(0.34, 0.28)

# Create test exosphere
moon = bodies.Moon(0.5 * 2 * np.pi, 0.5 * const.R_Io, 1.5 * HD189733b.R, HD189733b)
moon_exosphere = gasprop.TidallyHeatedMoon(3.34, moon)
moon_exosphere.addSourceRateFunction('dishoom_sodium_Rio_X_0p3_June10_notides.txt',
                                     10**2.5, 3.818e-23)
moon_exosphere.addConstituent('NaI', 1e6)
moon_exosphere.constituents[0].addLookupFunctionToConstituent(wavelengthGrid)

atmos = gasprop.Atmosphere([moon_exosphere], True)
main = gasprop.Transit(atmos, wavelengthGrid, spatialGrid)

print("\nTiming individual components...\n")

# Time wavelength grid construction
times = []
for _ in range(10):
    start = time.time()
    main.addWavelength()
    times.append(time.time() - start)
avg_wavelength = np.mean(times) * 1000
print(f"1. Wavelength grid construction: {avg_wavelength:.2f} ms")

# Time chord grid construction
times = []
for _ in range(10):
    start = time.time()
    chordGrid = spatialGrid.getChordGrid()
    times.append(time.time() - start)
avg_chordgrid = np.mean(times) * 1000
print(f"2. Chord grid generation:        {avg_chordgrid:.2f} ms")

# Time main sumOverChords
times = []
for _ in range(3):
    start = time.time()
    R = main.sumOverChords()
    times.append(time.time() - start)
avg_sumchords = np.mean(times) * 1000
print(f"3. sumOverChords (full calc):    {avg_sumchords:.2f} ms")

# Time individual chord evaluation
x = spatialGrid.constructXaxis()
delta_x = spatialGrid.getDeltaX()
phi, rho, orbphase = 0.5, HD189733b.hostStar.R * 0.8, 0.0

times = []
for _ in range(100):
    start = time.time()
    tau = atmos.getLOSopticalDepth_Batch(x, np.array([phi]), np.array([rho]),
                                         np.array([orbphase]), main.wavelength, delta_x)
    times.append(time.time() - start)
avg_single_chord = np.mean(times) * 1000
print(f"4. Single chord optical depth:   {avg_single_chord:.2f} ms")

# Estimate breakdown
n_chords = spatialGrid.rho_steps * spatialGrid.phi_steps
print(f"\nBreakdown estimate:")
print(f"  Number of chords: {n_chords}")
print(f"  Time per chord: {avg_sumchords / n_chords:.2f} ms")
print(f"  Overhead (grid, etc): {avg_wavelength + avg_chordgrid:.2f} ms")

print("\n" + "=" * 70)
print("What Can Be Precomputed?")
print("=" * 70)

precomputable = {
    "Wavelength grid": (avg_wavelength, "Fixed for all models"),
    "Chord grid geometry": (avg_chordgrid, "Fixed for all models"),
    "Absorption cross-sections": (0, "Already cached in lookup tables"),
}

parameter_dependent = {
    "Number density field": "Depends on: orbital_phase, orbital_radius, moon_radius, log_tau_Na",
    "Velocity field": "Depends on: orbital_phase (Doppler shifts)",
    "Optical depth integration": "Depends on: all parameters + wavelength",
    "Beer-Lambert attenuation": "Depends on: optical depth → all parameters"
}

print("\n✅ Can be precomputed (one-time):")
total_precompute = 0
for name, (time_ms, note) in precomputable.items():
    print(f"  {name:30s}: {time_ms:6.2f} ms - {note}")
    total_precompute += time_ms

print(f"\n  Total precomputable: {total_precompute:.2f} ms ({100*total_precompute/avg_sumchords:.1f}% of runtime)")

print("\n❌ Cannot be precomputed (parameter-dependent):")
for name, dependency in parameter_dependent.items():
    print(f"  {name:30s}: {dependency}")

remaining_time = avg_sumchords - total_precompute
print(f"\n  Remaining (must recalculate): {remaining_time:.2f} ms ({100*remaining_time/avg_sumchords:.1f}% of runtime)")

print("\n" + "=" * 70)
print("Conclusion")
print("=" * 70)
print(f"\nOnly ~{100*total_precompute/avg_sumchords:.1f}% of runtime can be precomputed!")
print(f"The remaining ~{100*remaining_time/avg_sumchords:.1f}% depends on model parameters.")
print(f"\nPrecomputation would save: {total_precompute:.0f} ms per model")
print(f"Still need to compute:     {remaining_time:.0f} ms per model")
print(f"\nSpeedup from precomputation: {avg_sumchords/remaining_time:.1f}×")
print(f"Speedup from NN emulator:    {avg_sumchords/5:.0f}×")
print(f"\n→ NN is still ~{(avg_sumchords/5)/(avg_sumchords/remaining_time):.0f}× faster than optimally precomputed Prometheus!")
