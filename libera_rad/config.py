"""Module containing paths to configuration and reference data for L1B processing."""
import os
from pathlib import Path

transfer_function_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transfer_function.nc")

l1b_ground_calibration_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'data',
                                                'l1b_ground_calibration.json'))

product_config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'data',
                                        'L1B_RAD-4CH_product_definition.yml'))
