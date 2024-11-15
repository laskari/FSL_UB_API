import requests
import time
import os
# Define the URL of the API endpoint
url = "http://localhost:5002/ub_extraction"

# Define the payload data as a dictionary
data = {
    "FilePath": r"/Data/FSl_ML/Test_Data/UB_Test_data/BSCPRO/GRPNG2X/TIFF/PHP1245WC030_001.tiff"
}

try:
    # Measure the start time
    start_time = time.time()
    for img_name in os.listdir("/Data/FSl_ML/Test_Data/UB_Test_data/BSCPRO/GRPNG2X/TIFF")[:50]:
    # Make the POST request with the JSON payloads
        if img_name.endswith(".tiff"):
            print(img_name)
            data = {
                "FilePath": f"/Data/FSl_ML/Test_Data/UB_Test_data/BSCPRO/GRPNG2X/TIFF/{img_name}"
            }

            response = requests.post(url, json=data)

    # Measure the total time taken for execution
    total_time_taken = time.time() - start_time
    print(start_time)
    print(time.time())
    print(f"Total time taken for execution: {total_time_taken}")

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        # print("Response:", response.json())
        print("Request was successful.")
    else:
        # If request was not successful, print error message
        # print("Error:", response.content)
        print(f"Request failed with status code {response.status_code}")
except Exception as e:
    print("Error occurred during API call:", e)


# import requests
# import time
# import io

# # Define the URL of the API endpoint
# url = "http://localhost:8001/hcfa_extraction"

# # Read the image file as bytes
# # with open(r"D:\project\FSL\FSL_codebase\api\ADA\images\2027C43DD013_001.jpg", "rb") as file:
# #     image_content = file.read()
# files = {'file': (f'2027C43DD013_001.jpg', open(r'D:\project\FSL\FSL_codebase\api\ADA\images\2027C43DD013_001.jpg', 'rb'))}


# try:
#     # Measure the start time
#     start_time = time.time()

#     # Make the POST request with the file content
#     response = requests.post(url, files=files)

#     # Measure the total time taken for execution
#     total_time_taken = time.time() - start_time
#     print(f"Total time taken for execution: {total_time_taken}")

#     # Check if the request was successful (status code 200)
#     if response.status_code == 200:
#         # Print the response content
#         print("Response:", response.json())
#         print("Request was successful.")
#     else:
#         # If request was not successful, print error message
#         print("Error:", response.content)
#         print(f"Request failed with status code {response.status_code}")
# except Exception as e:
#     print("Error occurred during API call:", e)


# import requests

# # Define the URL where the endpoint is hosted
# url = "http://localhost:5001/hcfa_extraction"

# # Path to the file you want to upload
# file_path = r"D:\project\FSL\FSL_codebase\api\HCFA\images\117914B4AZ009_001_0.jpg"

# # Open the file in binary mode
# with open(file_path, "rb") as file:
#     # Prepare the file to be uploaded
#     files = {"file": file}

#     # Send the POST request with the file
#     response = requests.post(url, files=files)

# # Print the response
# print(response.json())