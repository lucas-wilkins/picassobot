import time
import requests
import os


def upload_file_to_klipper(url, file_path):
    """
    Uploads a file to Klipper using Moonraker API.

    Args:
        url (str): The base URL of the Moonraker API.
        file_path (str): Path to the local file to be uploaded.

    Returns:
        str: The name of the uploaded file on the Klipper system.
    """
    endpoint = f"{url}/server/files/upload"
    filename = os.path.basename(file_path)

    with open(file_path, 'rb') as file:
        files = {
            'file': (filename, file)
        }
        response = requests.post(endpoint, files=files)

    response.raise_for_status()  # Ensure we raise an error if the request failed
    print(f"File '{filename}' uploaded successfully.")
    return filename


def start_print_job(url, filename):
    """
    Starts a print job using Moonraker API.

    Args:
        url (str): The base URL of the Moonraker API.
        filename (str): The name of the file to print.
    """
    endpoint = f"{url}/printer/print/start"
    params = {"filename": filename}
    response = requests.post(endpoint, json=params)
    response.raise_for_status()  # Ensure we raise an error if the request failed
    print(f"Print job for '{filename}' started successfully.")


def wait_for_print_to_finish(url):
    """
    Waits until the print job is finished by polling the print status.

    Args:
        url (str): The base URL of the Moonraker API.
    """
    endpoint = f"{url}/printer/objects/query?print_stats"
    while True:
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()

        state = data["result"]["status"]["print_stats"]["state"]
        print(f"Current print state: {state}")

        if state in ["complete", "error", "cancelled"]:
            print(f"Print job finished with state: {state}")
            break

        time.sleep(5)

def print_file(moonraker_url):

    # Replace with your Moonraker base URL and file path
    # moonraker_url = "http://169.254.171.193:7125"
    file_to_upload = "latest_image.gcode"

    try:
        uploaded_filename = upload_file_to_klipper(moonraker_url, file_to_upload)
        start_print_job(moonraker_url, uploaded_filename)
        wait_for_print_to_finish(moonraker_url)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
