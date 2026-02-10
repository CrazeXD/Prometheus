"""
This module provides utilities for memory-aware chunking of large computations.
It includes functions to estimate memory usage and chunk data based on available RAM.

Created on 9. February 2026 by GitHub Copilot.
"""

import numpy as np
import psutil
from typing import Tuple, Optional


def get_available_memory(max_memory_gb: float = 2.0) -> int:
    """Gets the amount of memory available for computation.

    Args:
        max_memory_gb (float): Maximum memory to use in gigabytes. Defaults to 2.0 GB.
                             Set to None to use 50% of available system RAM.

    Returns:
        int: Available memory in bytes.
    """
    if max_memory_gb is None:
        # Use 50% of available system memory
        available_bytes = int(psutil.virtual_memory().available * 0.5)
    else:
        # Use the specified amount, but don't exceed available memory
        max_bytes = int(max_memory_gb * 1e9)
        available_bytes = min(max_bytes, psutil.virtual_memory().available)
    
    return available_bytes


def estimate_chord_memory(
    num_wavelengths: int,
    orbphase_steps: int
) -> int:
    """Estimates memory required for a single chord evaluation.

    Args:
        num_wavelengths (int): Number of wavelength points.
        orbphase_steps (int): Number of orbital phase steps.

    Returns:
        int: Estimated memory in bytes for one chord.
    """
    # Each chord produces two arrays (F_in, F_out) of shape (orbphase_steps, num_wavelengths)
    # plus some overhead for intermediate calculations
    bytes_per_array = num_wavelengths * orbphase_steps * 8  # 8 bytes for float64
    # 2 arrays (F_in, F_out) + some overhead for intermediate computations
    return 2 * bytes_per_array * 1.2  # 20% overhead for safety


def calculate_optimal_chunk_size(
    total_chords: int,
    num_wavelengths: int,
    orbphase_steps: int,
    max_memory_gb: float = 2.0
) -> int:
    """Calculates the optimal number of chords to process in a single chunk.

    Args:
        total_chords (int): Total number of chords to process.
        num_wavelengths (int): Number of wavelength points.
        orbphase_steps (int): Number of orbital phase steps.
        max_memory_gb (float): Maximum memory to use in gigabytes. Defaults to 2.0 GB.

    Returns:
        int: Number of chords to process in one chunk.
    """
    available_bytes = get_available_memory(max_memory_gb)
    memory_per_chord = estimate_chord_memory(num_wavelengths, orbphase_steps)
    
    if memory_per_chord == 0:
        # Fallback: process at least 1 chord
        chunk_size = max(1, total_chords)
    else:
        chunk_size = max(1, int(available_bytes / memory_per_chord))
    
    # Don't exceed total number of chords
    chunk_size = min(chunk_size, total_chords)
    
    return chunk_size


def chunk_array(
    array: np.ndarray,
    chunk_size: int,
    axis: int = 0
) -> list:
    """Splits an array into chunks along a specified axis.

    Args:
        array (np.ndarray): Array to chunk.
        chunk_size (int): Number of elements per chunk along the specified axis.
        axis (int): Axis along which to chunk. Defaults to 0.

    Returns:
        list: List of array chunks.
    """
    chunks = []
    array_shape = array.shape
    num_elements = array_shape[axis]
    
    for start_idx in range(0, num_elements, chunk_size):
        end_idx = min(start_idx + chunk_size, num_elements)
        chunk_slice = [slice(None)] * len(array_shape)
        chunk_slice[axis] = slice(start_idx, end_idx)
        chunks.append(array[tuple(chunk_slice)])
    
    return chunks


def chunk_indices(
    total_items: int,
    chunk_size: int
) -> list:
    """Generates index ranges for chunking an iterable.

    Args:
        total_items (int): Total number of items.
        chunk_size (int): Number of items per chunk.

    Returns:
        list: List of tuples (start_idx, end_idx) for each chunk.
    """
    chunks = []
    for start_idx in range(0, total_items, chunk_size):
        end_idx = min(start_idx + chunk_size, total_items)
        chunks.append((start_idx, end_idx))
    
    return chunks
