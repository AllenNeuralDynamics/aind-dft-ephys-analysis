"""Example test template with NWBUtils integration."""

import unittest

from nwb_utils import NWBUtils

class NWBUtilsTest(unittest.TestCase):
    """Unit tests for NWBUtils static methods."""

    def test_assert_example(self):
        """Example of how to test the truth of a statement."""
        self.assertTrue(1 == 1)

    def test_read_ephys_nwb(self):
        """Test loading an example ephys NWB file via full path."""
        ephys_path = (
            '/root/capsule/data'
            '/ecephys_753124_2024-12-10_17-24-56_sorted_2024-12-13_09-48-25'
            '/nwb'
            '/ecephys_753124_2024-12-10_17-24-56_experiment1_recording1.nwb'
        )
        data = NWBUtils.read_ephys_nwb(nwb_full_path=ephys_path)
        self.assertIsNotNone(
            data,
            f"Failed to read ephys NWB at {ephys_path}"
        )

    def test_read_ophys_nwb(self):
        """Test loading an example ophys NWB file via full path."""
        ophys_path = (
            '/root/capsule/data'
            '/behavior_777405_2025-04-07_16-22-07_processed_2025-04-08_17-27-40'
            '/nwb'
            '/behavior_777405_2025-04-07_16-22-07.nwb'
        )
        data = NWBUtils.read_ophys_nwb(nwb_full_path=ophys_path)
        self.assertIsNotNone(
            data,
            f"Failed to read ophys NWB at {ophys_path}"
        )


if __name__ == "__main__":
    unittest.main()
