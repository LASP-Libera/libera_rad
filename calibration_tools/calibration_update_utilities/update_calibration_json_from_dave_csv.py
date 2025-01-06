# Installed
import pandas as pd
import json
import datetime
# Local
from libera_rad.calibration.calibration_models import LiberaGroundCalibration


# These methods are not intended for regular use!

# This is a utility function that can be used to update the numerical calibration information in the calibration json
# from a csv file that Dave provides. This is useful for updating the calibration information with new numerical
# calibration information. This function will update the numerical calibration information in the associated l1b
# ground calibration json file.

def create_updated_numerical_calibration_information_object_from_csv(calibration_json, csv_file):
    """
    Update the numerical information in a python object of the calibration json from the csv file and update
    """
    # Read the csv file
    dave_data_from_csv = pd.read_csv(csv_file)
    # Get the current ground calibration information
    with open(calibration_json) as f:
        ground_calibration = json.load(f)
    ground_calibration_data = LiberaGroundCalibration(**ground_calibration)

    # Update the ground calibration information
    for channel in dave_data_from_csv.channel_name:
        csv_channel_data = dave_data_from_csv[dave_data_from_csv["channel_name"] == channel]
        channel_json_calibration = ground_calibration_data.channels[channel]
        assert channel_json_calibration.name == channel
        assert channel_json_calibration.chip_sn == csv_channel_data.chip_sn.values[0]
        assert channel_json_calibration.detector_pcb == csv_channel_data.detector_pcb.values[0]
        channel_json_calibration.collection_area = csv_channel_data.collection_area.values[0]
        channel_json_calibration.collection_area_sd = csv_channel_data.collection_area_sd.values[0]
        channel_json_calibration.solid_angle = csv_channel_data.solid_angle.values[0]
        channel_json_calibration.solid_angle_sd = csv_channel_data.solid_angle_sd.values[0]
        channel_json_calibration.radiance_coefficients.t0_per_dn = csv_channel_data.nw_per_dn_t0.values[0]
        channel_json_calibration.radiance_coefficients.constant_offset = csv_channel_data.nw_per_dn0.values[0]
        channel_json_calibration.radiance_coefficients.temp_difference_linear = csv_channel_data.nw_per_dn1.values[0]
        channel_json_calibration.radiance_coefficients.temp_difference_quadratic = csv_channel_data.nw_per_dn2.values[0]
        channel_json_calibration.radiance_coefficients.dark_offset_linear = csv_channel_data.nw_per_dn3.values[0]
        channel_json_calibration.radiance_coefficients.dark_offset_quadratic = csv_channel_data.nw_per_dn4.values[0]
        channel_json_calibration.radiance_coefficients.dark_vs_temperature_crossover = \
            csv_channel_data.nw_per_dn5.values[0]

    ground_calibration_data.channels[channel] = channel_json_calibration
    return ground_calibration_data


# Run this code in a script to update the calibration json with the new numerical calibration information

calibration_json = "../l1b_ground_calibration.json"

# Update production numerical calibration data from Dave's CSV (v2 as of 11/1/24)
# Contact Matt or Dave to get a copy of the CSV to use for updating
dave_numerical_data_csv = "/path/to/libera_science_radiometer_conversions_emfpe_v2.csv"
updated_cal_data = create_updated_numerical_calibration_information_object_from_csv(calibration_json,
                                                                                    dave_numerical_data_csv)

# Do Other Updates
# TODO add other updates as needed.

# Save the updated calibration data with a new version
current_cal_version = updated_cal_data.calibration_version

# Patch update for now
cal_patch = current_cal_version.split(".")[-1]
cal_patch = str(int(cal_patch) + 1)

new_cal_version = f"{current_cal_version[:-1]}{cal_patch}"

updated_cal_data.calibration_version = new_cal_version
updated_cal_data.calibration_notes = (updated_cal_data.calibration_notes +
                                      f"|| Updated from CSV on {datetime.datetime.now().date()}")

with open(calibration_json, 'w', encoding='utf-8') as f:
    json.dump(updated_cal_data.dict(), f, indent=4)
