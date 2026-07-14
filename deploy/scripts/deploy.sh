

set -e


KEY_FILE="${CLOUDSENSE_KEY_FILE:-$HOME/Downloads/cloudsense-key.pem}"
REGION="${CLOUDSENSE_AWS_REGION:-ap-south-1}"
# Repo root (two levels up from deploy/scripts): we upload src/ + model_export/.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# By default restrict SSH to your current public IP. Override: SSH_CIDR=0.0.0.0/0
SSH_CIDR="${SSH_CIDR:-$(curl -s ifconfig.me)/32}"

echo "=================================================="
echo " CloudSense — AWS EC2 Free Tier Deployment"
echo " Region: $REGION"
echo "=================================================="

# ── Step 1: Configure AWS CLI ─────────────────────────────────────────────────
echo ""
echo "📋 Step 1: Checking AWS CLI..."
if ! command -v aws &>/dev/null; then
    echo "  AWS CLI not found. Install it first:"
    echo "    https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    exit 1
fi
aws sts get-caller-identity > /dev/null || { echo "  Run 'aws configure' first."; exit 1; }
echo "  AWS CLI configured"

# ── Step 2: Create Security Group ────────────────────────────────────────────
echo ""
echo "📋 Step 2: Creating Security Group..."
SG_ID=$(aws ec2 create-security-group \
    --group-name cloudsense-sg \
    --description "CloudSense inference API" \
    --region "$REGION" \
    --query 'GroupId' --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
        --filters Name=group-name,Values=cloudsense-sg \
        --region "$REGION" \
        --query 'SecurityGroups[0].GroupId' --output text)

# Allow SSH (22, restricted to your IP) and API (8000)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 22 \
    --cidr "$SSH_CIDR" --region "$REGION" 2>/dev/null || true
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 8000 \
    --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 80 \
    --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true
echo "  Security Group: $SG_ID"

# ── Step 3: Create Key Pair (if not exists) ───────────────────────────────────
echo ""
echo "📋 Step 3: Key Pair..."
KEY_NAME="cloudsense-key"
if [ ! -f "$KEY_FILE" ]; then
    echo "   Creating new key pair → $KEY_FILE"
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "✅  Key saved: $KEY_FILE"
else
    echo "✅  Using existing key: $KEY_FILE"
fi

# ── Step 4: Launch EC2 t2.micro (FREE TIER) ───────────────────────────────────
echo ""
echo "📋 Step 4: Launching EC2 t2.micro (Free Tier)..."
# Ubuntu 22.04 LTS AMI for ap-south-1
AMI_ID="ami-0f58b397bc5c1f2e8"   # Ubuntu 22.04 LTS, Mumbai

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type t2.micro \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$REGION" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cloudsense}]' \
    --query 'Instances[0].InstanceId' --output text)

echo "   Instance ID: $INSTANCE_ID"
echo "   Waiting for instance to start (≈60 seconds)..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo " Instance running at: $PUBLIC_IP"

# Save IP for future use
echo "$PUBLIC_IP" > .ec2_ip
echo "$INSTANCE_ID" > .ec2_instance_id

# ── Step 5: Wait for SSH to be ready ─────────────────────────────────────────
echo ""
echo "📋 Step 5: Waiting for SSH to be ready..."
sleep 30
for i in {1..10}; do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" "echo ok" 2>/dev/null && break
    echo "   Attempt $i/10... waiting"
    sleep 10
done
echo "  SSH ready"

# ── Step 6: Upload inference files ───────────────────────────────────────────
echo ""
echo "📋 Step 6: Uploading files to EC2..."
# The rewritten API imports the project's src/ modules and loads model_export/,
# so we upload BOTH (no Colab zip anymore). model_export/ must exist locally
# (run `python src/main.py` first to train + export it).
if [ ! -d "$REPO_ROOT/model_export" ]; then
    echo "  ERROR: $REPO_ROOT/model_export not found. Run 'python src/main.py' first."
    exit 1
fi
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    "$REPO_ROOT/deploy/inference/app.py" \
    "$REPO_ROOT/deploy/inference/requirements.txt" \
    "$REPO_ROOT/deploy/scripts/setup_ec2.sh" \
    ubuntu@"$PUBLIC_IP":/home/ubuntu/
scp -r -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    "$REPO_ROOT/src" "$REPO_ROOT/model_export" \
    ubuntu@"$PUBLIC_IP":/home/ubuntu/
echo "  app.py + src/ + model_export/ uploaded"

# Lay out /app = { app.py, requirements.txt, setup_ec2.sh, src/, model_export/ }
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" \
    "sudo mkdir -p /app && sudo cp -r /home/ubuntu/{app.py,requirements.txt,setup_ec2.sh,src,model_export} /app/ && sudo chown -R ubuntu:ubuntu /app"

# ── Step 7: Run setup script on EC2 ──────────────────────────────────────────
echo ""
echo "📋 Step 7: Running setup on EC2 (≈4 minutes)..."
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" \
    "cd /app && chmod +x setup_ec2.sh && ./setup_ec2.sh"

# ── Step 8: Health check ──────────────────────────────────────────────────────
echo ""
echo "📋 Step 8: Health check..."
sleep 5
HEALTH=$(curl -s "http://$PUBLIC_IP:8000/health" 2>/dev/null || echo "not ready")
echo "   Response: $HEALTH"

echo ""
echo "=================================================="
echo " 🎉  DEPLOYMENT COMPLETE!"
echo "=================================================="
echo ""
echo " API Base URL:  http://$PUBLIC_IP:8000"
echo " Health:        http://$PUBLIC_IP:8000/health"
echo " Docs (Swagger):http://$PUBLIC_IP:8000/docs"
echo " Metrics:       http://$PUBLIC_IP:8000/metrics"
echo " Predict:       POST http://$PUBLIC_IP:8000/predict"
echo ""
echo " SSH:  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo " Cost: \$0/month (AWS Free Tier for 12 months)"
echo ""
echo " To stop: aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION"
echo " To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo "=================================================="
