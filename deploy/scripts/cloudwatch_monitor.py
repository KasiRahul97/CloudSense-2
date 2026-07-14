"""
CloudSense -- AWS CloudWatch autoscaling integration (optional, real AWS).
=========================================================================
On a schedule (cron / EventBridge -> Lambda), this:
  1. reads recent CPU history for an instance/ASG from CloudWatch,
  2. asks the CloudSense API for the horizon-ahead forecast,
  3. writes the forecast back as a custom CloudWatch metric, and
  4. optionally adjusts an Auto Scaling Group based on the recommendation.

Requires `boto3` and AWS credentials (env / instance role). Configure via env:
  CLOUDSENSE_API            e.g. http://127.0.0.1:8000   (the inference API)
  CLOUDSENSE_AWS_REGION     e.g. ap-south-1
  CLOUDSENSE_INSTANCE_ID    the EC2 instance to read CPU for
  CLOUDSENSE_ASG_NAME       (optional) ASG to scale; if unset, only logs
"""
import os
import logging
from datetime import datetime, timezone, timedelta

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cloudsense.monitor")

API = os.getenv("CLOUDSENSE_API", "http://127.0.0.1:8000").rstrip("/")
REGION = os.getenv("CLOUDSENSE_AWS_REGION", "ap-south-1")
INSTANCE_ID = os.getenv("CLOUDSENSE_INSTANCE_ID", "")
ASG_NAME = os.getenv("CLOUDSENSE_ASG_NAME", "")


def _api_config() -> dict:
    """Discover look_back/horizon from the API so this stays in sync with the model."""
    r = requests.get(f"{API}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_cpu_history(cw, instance_id: str, n_points: int) -> list:
    """Pull the last n_points 5-min average CPU readings from CloudWatch."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=5 * (n_points + 2))
    resp = cw.get_metric_statistics(
        Namespace="AWS/EC2", MetricName="CPUUtilization",
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        StartTime=start, EndTime=end, Period=300, Statistics=["Average"])
    pts = sorted(resp["Datapoints"], key=lambda x: x["Timestamp"])
    return [float(p["Average"]) for p in pts]


def forecast(cpu_history: list) -> dict:
    """Call the CloudSense API for the horizon-ahead forecast."""
    r = requests.post(f"{API}/predict", json={"cpu_percent": cpu_history}, timeout=30)
    r.raise_for_status()
    return r.json()


def push_forecast(cw, instance_id: str, predicted_cpu: float):
    cw.put_metric_data(Namespace="CloudSense/Predictions", MetricData=[{
        "MetricName": "PredictedCPUUtilization",
        "Dimensions": [{"Name": "InstanceId", "Value": instance_id}],
        "Timestamp": datetime.now(timezone.utc),
        "Value": predicted_cpu, "Unit": "Percent"}])
    log.info("pushed predicted CPU=%.1f%% to CloudWatch", predicted_cpu)


def maybe_scale(asg, result: dict):
    rec = result.get("recommendation", "hold")
    pred = result.get("predicted_cpu_percent", 0.0)
    if not ASG_NAME:
        log.info("recommendation=%s (predicted %.1f%%); no ASG configured -> log only", rec, pred)
        return
    if rec == "scale_up":
        log.warning("SCALE UP: predicted %.1f%% in %s", pred, result.get("horizon_label"))
        # desc = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])
        # cap = desc["AutoScalingGroups"][0]["DesiredCapacity"]
        # asg.set_desired_capacity(AutoScalingGroupName=ASG_NAME, DesiredCapacity=cap + 1)
    elif rec == "scale_down":
        log.info("SCALE DOWN candidate: predicted %.1f%%", pred)
    else:
        log.info("HOLD: predicted %.1f%%", pred)


def main():
    import boto3  # imported here so the module is importable without boto3 installed
    if not INSTANCE_ID:
        raise SystemExit("set CLOUDSENSE_INSTANCE_ID")
    cfg = _api_config()
    look_back = cfg.get("look_back", 48)
    cw = boto3.client("cloudwatch", region_name=REGION)
    asg = boto3.client("autoscaling", region_name=REGION)

    history = fetch_cpu_history(cw, INSTANCE_ID, n_points=look_back + 8)
    if len(history) < look_back:
        log.error("not enough CloudWatch data: have %d, need %d", len(history), look_back)
        return
    result = forecast(history)
    log.info("forecast: %s", result)
    push_forecast(cw, INSTANCE_ID, result["predicted_cpu_percent"])
    maybe_scale(asg, result)


if __name__ == "__main__":
    main()
