import depthai as dai
import json

# Load the calibration data from the JSON file
backup_filename = "oakd_factory_calibration.json"  # Replace with your actual filename
with open(backup_filename, 'r') as f:
    calibration_data = json.load(f)

# Connect to the OAK-D camera
with dai.Device() as device:
    # Convert the JSON data back to the format expected by depthai
    calibration_data_string = json.dumps(calibration_data)
    calibration_obj = dai.CalibrationHandler.parseCalibration(calibration_data_string)

    # Write the calibration data to the device
    device.flashFactoryCalibration(calibration_obj)

    print("Factory calibration data restored to the OAK-D.")