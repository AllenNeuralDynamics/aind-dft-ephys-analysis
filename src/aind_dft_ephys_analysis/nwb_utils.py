import os
import glob
from typing import Optional, Any
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from general_utils import extract_session_name_core

class NWBUtils:
    """
    Utility class offering static methods to locate and read ephys, behavior, and ophys NWB files.
    """

    @staticmethod
    def read_ephys_nwb(
        folder_path: str = '/root/capsule/data/',
        nwb_full_path: Optional[str] = None,
        session_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Reads an ephys NWB file and returns its data layout.

        Priority:
        1. Use `nwb_full_path` if provided.
        2. Otherwise, use `session_name` to locate a folder matching `ecephys_<session>_*sorted*` under `folder_path`,
           then find a file ending in `*experiment1_recording1.nwb` inside its `nwb/` subfolder.

        If multiple matching folders or files are found, warn and return None.

        Args:
            folder_path: Base path containing ephys sessions.
            nwb_full_path: Direct path to the NWB file (optional).
            session_name: Identifier for the ephys session (optional).

        Returns:
            NWB file data object on success, or None on failure.
        """
        # Direct path provided
        if nwb_full_path:
            if not os.path.exists(nwb_full_path):
                print(f"Warning: Provided path '{nwb_full_path}' does not exist.")
                return None
            data = NWBZarrIO(nwb_full_path, 'r').read()
            print(f"Successfully read ephys NWB from: {nwb_full_path}")
            return data

        # Need session_name to search
        if not session_name:
            print("Warning: session_name is required when nwb_full_path is not provided.")
            return None
        core = extract_session_name_core(session_name)
        pattern = os.path.join(folder_path, f"ecephys_{core}_*sorted*")
        folders = glob.glob(pattern)
        if not folders:
            print(f"Warning: No folder matching '{pattern}' found.")
            return None
        if len(folders) > 1:
            print(f"Warning: Multiple ephys folders found for session '{core}': {folders}.")
            return None
        nwb_folder = folders[0]

        exp_pattern = os.path.join(nwb_folder, 'nwb', '*experiment1_recording1.nwb')
        files = glob.glob(exp_pattern)
        if not files:
            print(f"Warning: No NWB file found with pattern '{exp_pattern}'.")
            return None
        if len(files) > 1:
            print(f"Warning: Multiple ephys NWB files found: {files}.")
            return None
        file_path = files[0]
        print(f"Found ephys NWB: {file_path}")
        try:
            io = NWBZarrIO(file_path, 'r')
            data = io.read()
            data.io = io  # Optional: attach so you can later call `data.io.close()` safely
            print(f"Successfully read ephys NWB from: {file_path}")
            return data
        except Exception as e:
            print(f"Error reading ephys NWB file '{file_path}': {e}")
            return None

    @staticmethod
    def read_behavior_nwb(
        folder_path: str = '/root/capsule/data/behavior_nwb',
        nwb_full_path: Optional[str] = None,
        session_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Reads a behavior NWB file and returns its data layout.

        Priority:
        1. Use `nwb_full_path` if provided.
        2. Otherwise, constructs the expected filename from `session_name` under
           `folder_path/behavior_nwb/`, optionally prefixed with 'behavior_'.

        If multiple matching files are found, warn and return None.

        Args:
            folder_path: Base path containing 'behavior_nwb' directory.
            nwb_full_path: Direct path to the NWB file (optional).
            session_name: Identifier for the behavior session (optional).

        Returns:
            NWB file data object on success, or None on failure.
        """
        # Direct path provided
        if nwb_full_path:
            if not os.path.exists(nwb_full_path):
                print(f"Warning: Provided path '{nwb_full_path}' does not exist.")
                return None
            path = nwb_full_path
        else:
            if not session_name:
                print("Warning: session_name is required when nwb_full_path is not provided.")
                return None
            core = extract_session_name_core(session_name)
            candidates = [os.path.join(folder_path, f"{core}.nwb"),
                          os.path.join(folder_path, f"behavior_{core}.nwb")]
            files = [p for p in candidates if os.path.exists(p)]
            if not files:
                print(f"Warning: No behavior NWB files found for session '{core}'.")
                return None
            if len(files) > 1:
                print(f"Warning: Multiple behavior NWB files found: {files}.")
                return None
            path = files[0]
        print(f"Found behavior NWB: {path}")
        # Read file (HDF5 first, then Zarr)
        try:
            io = NWBHDF5IO(path, 'r')
            data = io.read()
            data.io = io  # Optional: attach so you can later call `data.io.close()` safely
            print(f"Successfully read behavior NWB from: {path}")
            return data
        except Exception:
            try:
                io = NWBZarrIO(path, 'r')
                data = io.read()
                data.io = io  # Optional: attach so you can later call `data.io.close()` safely
                print(f"Successfully read behavior NWB from: {path}")
                return data
            except Exception as e:
                print(f"Error reading behavior NWB file '{path}': {e}")
                return None

    @staticmethod
    def read_ophys_nwb(
        folder_path: str = '/root/capsule/data/',
        nwb_full_path: Optional[str] = None,
        session_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Reads an ophys NWB file and returns its data layout.

        Priority:
        1. Use `nwb_full_path` if provided.
        2. Otherwise, searches for a folder matching '*<session_name>_*processed*'
           under `folder_path`, then any '*.nwb' file inside its 'nwb/' subfolder.

        If multiple matching folders or files are found, warn and return None.

        Args:
            folder_path: Base path containing processed ophys session folders.
            nwb_full_path: Direct path to the NWB file (optional).
            session_name: Identifier for the ophys session (optional).

        Returns:
            NWB file data object on success, or None on failure.
        """
        # Direct path provided
        if nwb_full_path:
            if not os.path.exists(nwb_full_path):
                print(f"Warning: Provided path '{nwb_full_path}' does not exist.")
                return None
            data = NWBZarrIO(nwb_full_path, 'r').read()
            print(f"Successfully read ophys NWB from: {nwb_full_path}")
            return data

        # Need session_name to search
        if not session_name:
            print("Warning: session_name is required when nwb_full_path is not provided.")
            return None
        core = extract_session_name_core(session_name)
        folder_pattern = os.path.join(folder_path, f"*{core}_*processed*")
        proc_folders = glob.glob(folder_pattern)
        if not proc_folders:
            print(f"Warning: No folder matching '{folder_pattern}' found.")
            return None
        if len(proc_folders) > 1:
            print(f"Warning: Multiple ophys folders found: {proc_folders}.")
            return None
        proc_folder = proc_folders[0]

        file_pattern = os.path.join(proc_folder, 'nwb', '*.nwb')
        files = glob.glob(file_pattern)
        if not files:
            print(f"Warning: No NWB file found with pattern '{file_pattern}'.")
            return None
        if len(files) > 1:
            print(f"Warning: Multiple ophys NWB files found: {files}.")
            return None
        file_path = files[0]
        print(f"Found ophys NWB: {file_path}")
        try:
            io = NWBZarrIO(file_path, 'r')
            data = io.read()
            data.io = io  # Optional: attach so you can later call `data.io.close()` safely
            print(f"Successfully read ophys NWB from: {file_path}")
            return data
        except Exception as e:
            print(f"Error reading ophys NWB file '{file_path}': {e}")
            return None

    @staticmethod
    def combine_nwb(
        ephys_folder_path: str = '/root/capsule/data/',
        behavior_folder_path: str = '/root/capsule/data/behavior_nwb',
        session_name: Optional[str] = None,
        ephys_nwb_full_path: Optional[str] = None,
        behavior_nwb_full_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Reads both ephys and behavior NWB for a session, appends the ephys `units` table into
        the behavior NWB file if both are present, closes file handles, and returns:

          - The combined behavior NWB if both loaded,
          - The only-loaded NWB if one failed (with a warning),
          - None if neither could load.

        If neither full path is provided, session_name (plus folder paths) is required.
        """
        # Require session_name if no explicit paths
        if not ephys_nwb_full_path and not behavior_nwb_full_path and not session_name:
            print("Warning: session_name is required when no full paths are provided.")
            return None

        # Attempt to load each
        ephys_data = NWBUtils.read_ephys_nwb(
            folder_path=ephys_folder_path,
            nwb_full_path=ephys_nwb_full_path,
            session_name=session_name
        )
        behavior_data = NWBUtils.read_behavior_nwb(
            folder_path=behavior_folder_path,
            nwb_full_path=behavior_nwb_full_path,
            session_name=session_name
        )

        # Neither loaded?
        if ephys_data is None and behavior_data is None:
            print("Warning: Could not load either ephys or behavior NWB.")
            return None

        # Only ephys
        if behavior_data is None:
            print("Warning: Behavior NWB failed to load; returning ephys NWB only.")
            return ephys_data

        # Only behavior
        if ephys_data is None:
            print("Warning: Ephys NWB failed to load; returning behavior NWB only.")
            return behavior_data

        # Both loaded → combine
        try:
            behavior_data.units = ephys_data.units
            print("Successfully appended units table to behavior NWB.")
        except Exception as e:
            print(f"Error appending units table: {e}")
        finally:
            # **Only close the ephys reader** so behavior_data stays usable
            try:
                ephys_data.io.close()
            except Exception:
                pass

        return behavior_data