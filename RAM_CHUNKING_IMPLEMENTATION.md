# RAM-Based Chunking Implementation for PROMETHEUS

## Overview

The main loop in PROMETHEUS's `Transit.sumOverChords()` method has been vectorized with intelligent RAM-based chunking. This allows the simulation to handle large spatial grids efficiently without running out of memory.

## Key Components

### 1. New Module: `memoryHandler.py`

A utility module that provides memory-aware chunking functionality:

- **`get_available_memory(max_memory_gb)`**: Returns available memory in bytes, respecting the specified limit
  - Default: 2.0 GB
  - Set to `None` to use 80% of available system RAM
  
- **`estimate_chord_memory(num_wavelengths, orbphase_steps)`**: Estimates memory required per chord evaluation
  
- **`calculate_optimal_chunk_size(...)`**: Computes the optimal number of chords to process in one chunk based on:
  - Total number of chords
  - Number of wavelength points
  - Number of orbital phases
  - Available RAM
  
- **`chunk_array(array, chunk_size, axis)`**: Utility to split arrays into chunks
  
- **`chunk_indices(total_items, chunk_size)`**: Generates index ranges for chunking

### 2. Updated `Transit.sumOverChords()` Method

**Signature:**
```python
def sumOverChords(self, max_memory_gb: float = 2.0) -> np.ndarray
```

**Changes:**
- Now accepts optional `max_memory_gb` parameter (default: 2.0 GB)
- Automatically calculates optimal chunk size based on available RAM
- Processes all chords in manageable chunks
- Accumulates results incrementally to minimize peak memory usage
- Provides progress logging for transparency
- Clears intermediate data after each chunk to free memory

**How it works:**
1. Calculates chord grid dimensions
2. Determines optimal chunk size based on memory constraints
3. Iterates through chunks of chords
4. For each chunk:
   - Evaluates all chords in that chunk
   - Accumulates F_in and F_out contributions
   - Clears intermediate arrays
5. Returns final transit depth ratio

### 3. Updated Entry Points

#### `prometheus.py`
- Added command-line argument support: `--max-memory 4.0`
- Usage: `python prometheus.py setup_name --max-memory 4.0`
- Default: 2.0 GB if not specified

#### `mainRetrieval.py`
- Added `max_memory_gb` variable for easy configuration
- Can be modified to use different memory limits

## Usage Examples

### Default (2 GB)
```python
R = main.sumOverChords()  # Uses 2.0 GB by default
```

### Custom Memory Limit
```python
R = main.sumOverChords(max_memory_gb=4.0)  # Use 4 GB
```

### Use 50% of Available RAM
```python
R = main.sumOverChords(max_memory_gb=None)  # Uses 50% of system RAM
```

### Command Line (prometheus.py)
```bash
python prometheus.py my_setup --max-memory 8.0
```

## Performance Characteristics

The implementation is vectorized in the following ways:

1. **Memory Efficiency**: Processes only what fits in RAM at once
2. **Accumulation**: Direct summation of contributions, avoiding redundant reshaping
3. **Transparency**: Progress logging shows chunk processing in real-time
4. **Flexibility**: Easily adjustable memory constraints for different systems

## Memory Estimation

For a typical configuration:
- `num_wavelengths = 1000`
- `orbphase_steps = 25`
- Memory per chord ≈ 480 KB (with 20% overhead)

With 2 GB limit:
- Chunk size ≈ 4,200 chords
- If total chords = 100,000, this results in ~24 chunks

## Backward Compatibility

The new implementation is fully backward compatible:
- Existing code calling `sumOverChords()` without arguments works unchanged
- Optional parameter can be used for fine-tuning

## Dependencies

The `memoryHandler` module requires:
- `numpy`
- `psutil` (for system memory detection)

If `psutil` is not installed, install it with:
```bash
pip install psutil
```

## Future Enhancements

Possible improvements:
1. Parallel chunk processing with multiprocessing
2. Adaptive memory estimation based on actual measurements
3. Integration with progress bars (e.g., tqdm)
4. Caching of frequently computed values within chunks
