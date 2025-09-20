import json
import boto3
import base64

# ==============================
# Lambda 1: Serialize Image Data
# ==============================

s3 = boto3.client('s3')


def serializeImageData(event, context):
    """A function to serialize target data from S3"""
    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

# ==============================
# Lambda 2: Predict Image Class
# ==============================


ENDPOINT = "image-classification-2025-09-20-08-00-11-224"

runtime = boto3.client("sagemaker-runtime")


def lambda_handler(event, context):

    # Parse body
    if "body" in event:
        event = event["body"]
    # Decode image
    image = base64.b64decode(event["image_data"])

    # call the sagemaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="image/png",
        Body=image
    )
    # read the JSON response
    inferences = json.loads(response["Body"].read().decode("utf-8"))

    # We return the data back to the Step Function
    event["inferences"] = inferences
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
# ==============================
# Lambda 3: Filter Low Confidence
# ==============================


THRESHOLD = .93


def filterLowConfidence(event, context):
    # unwrap body if it’s still a JSON string
    if "body" in event and isinstance(event["body"], str):
        event = json.loads(event["body"])
    elif "body" in event and isinstance(event["body"], dict):
        event = event["body"]

    # at this point, event["inferences"] should be a list of floats
    inferences = event["inferences"]
    # confirm validate type
    inferences = [float(x) for x in inferences]
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) >= THRESHOLD

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    try:
        if not meets_threshold:
            raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
        return {
            'statusCode': 200,
            'body': json.dumps(event)
        }
    except Exception as e:
        return {"error": str(e)}