import json, os, base64, boto3, csv, io, logging
from typing import Any, Dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "")
RAW_BUCKET = os.environ.get("RAW_BUCKET", "")
OUTPUTS_BUCKET = os.environ.get("OUTPUTS_BUCKET", "")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
LLM_IMPUTER_ENABLED = os.environ.get("LLM_IMPUTER_ENABLED","true").lower()=="true"

s3 = boto3.client('s3')
smr = boto3.client('sagemaker-runtime')
bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)

def respond(status: int, body: Dict[str, Any]):
    return {"statusCode": status, "headers": {"Content-Type":"application/json"}, "body": json.dumps(body)}

def call_bedrock_reasoning(payload: Dict[str,Any]) -> Dict[str,Any]:
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    try:
      resp = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"messages":[{"role":"user","content":[{"type":"text","text":json.dumps(payload)}]}]})
      )
      txt = json.loads(resp['body'].read().decode('utf-8'))
      return {"ok": True, "raw": txt}
    except Exception as e:
      logger.exception("Bedrock error")
      return {"ok": False, "error": str(e)}

def call_sagemaker_imputer(records: list) -> list:
    csv_bytes = io.StringIO()
    writer = csv.writer(csv_bytes)
    for r in records:
        writer.writerow(r)
    body = csv_bytes.getvalue().encode("utf-8")
    resp = smr.invoke_endpoint(EndpointName=SAGEMAKER_ENDPOINT, ContentType="text/csv", Body=body)
    out = resp["Body"].read().decode("utf-8").strip().splitlines()
    return [line.split(",") for line in out]

def handle_s3_event(record) -> Dict[str, Any]:
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]
    logger.info(f"New object s3://{bucket}/{key}")

    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read().decode("utf-8")
    rows = list(csv.reader(io.StringIO(content)))
    header, data = rows[0], rows[1:]

    if LLM_IMPUTER_ENABLED:
        reasoning = call_bedrock_reasoning({"action":"plan-imputation","columns":header[:10]})
        logger.info(f"Bedrock reasoning ok={reasoning.get('ok')}")

    imputed = call_sagemaker_imputer(data)
    out_key = key.rsplit(".",1)[0] + ".imputed.csv"
    out_csv = io.StringIO()
    w = csv.writer(out_csv); w.writerow(header); [w.writerow(r) for r in imputed]
    s3.put_object(Bucket=OUTPUTS_BUCKET, Key=out_key, Body=out_csv.getvalue().encode("utf-8"))
    return {"ok": True, "output_key": out_key}

def handler(event, context):
    if "Records" in event and event["Records"] and event["Records"][0].get("eventSource") == "aws:s3":
        return handle_s3_event(event["Records"][0])

    try:
        body = event.get("body")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        payload = json.loads(body) if isinstance(body, str) and body.startswith("{") else {}
        if "s3_key" in payload:
            s3_key = payload["s3_key"]
            return handle_s3_event({"s3":{"bucket":{"name":RAW_BUCKET},"object":{"key":s3_key}}})
        if "csv" in payload:
            content = payload["csv"]
            rows = list(csv.reader(io.StringIO(content)))
            header, data = rows[0], rows[1:]
            if LLM_IMPUTER_ENABLED:
                _ = call_bedrock_reasoning({"action":"plan-imputation","columns":header[:10]})
            imputed = call_sagemaker_imputer(data)
            out_csv = io.StringIO(); w = csv.writer(out_csv)
            w.writerow(header); [w.writerow(r) for r in imputed]
            s3_key = "api-upload/imputed.csv"
            s3.put_object(Bucket=OUTPUTS_BUCKET, Key=s3_key, Body=out_csv.getvalue().encode("utf-8"))
            return respond(200, {"ok": True, "output_key": s3_key})
        return respond(400, {"ok": False, "error":"Provide {s3_key} or {csv}"})
    except Exception as e:
        logger.exception("Lambda error")
        return respond(500, {"ok": False, "error": str(e)})
