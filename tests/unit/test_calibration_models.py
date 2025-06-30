# Tests for the calibration models


def test_calibration_model_from_json(calibration_data):
    """Test that we can create a calibration model from the calibration JSON file"""
    assert len(calibration_data.boards) == 4
    assert len(calibration_data.channels) == 4
    assert len(calibration_data.housekeeping_temperature_coefficients) == 4
