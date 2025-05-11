import depthai as dai
import json

# Connect to the OAK-D camera
with dai.Device() as device:
    # Read the factory calibration data
    calibration_data = device.readFactoryCalibration().eepromToJson()

    # Specify the filename for the backup
    backup_filename = "oakd_factory_calibration.json"

    # Save the calibration data to a JSON file
    with open(backup_filename, 'w') as f:
        json.dump(calibration_data, f, indent=4)

    print(f"Factory calibration data saved to: {backup_filename}")