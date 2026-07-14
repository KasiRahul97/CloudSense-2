
set -e   # exit on error

echo "=================================================="
echo " CloudSense EC2 Setup"
echo "=================================================="

# 1. Update system ─
echo "[1/7] Updating system packages..."
sudo apt-get update -q
sudo apt-get upgrade -y -q

#  2. Install Python 3.11 + pip ─
echo "[2/7] Installing Python 3.11..."
sudo apt-get install -y -q python3.11 python3.11-venv python3-pip git unzip

#  3. Create app directory 
echo "[3/7] Creating /app directory..."
sudo mkdir -p /app/model_export
sudo chown -R ubuntu:ubuntu /app

#  4. Copy app files (already SCP-d by deploy.sh) ─
cd /app

#  5. Create virtual environment 
echo "[4/7] Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# 6. Install dependencies 
echo "[5/7] Installing Python packages (this takes ~3 min on t2.micro)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo " Packages installed"

# 7. Create systemd service (auto-start on reboot) 
echo "[6/7] Creating systemd service..."
sudo tee /etc/systemd/system/cloudsense.service > /dev/null <<'SERVICE'
[Unit]
Description=CloudSense Inference API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/app
Environment="MODEL_DIR=/app/model_export"
Environment="CLOUDSENSE_SRC=/app/src"
ExecStart=/app/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable cloudsense
sudo systemctl start cloudsense

#  8. Open port 8000 in local firewall (UFW) 
echo "[7/7] Opening firewall port 8000..."
sudo ufw allow 8000/tcp 2>/dev/null || true

echo ""
echo "=================================================="
echo "   Setup Complete!"
echo " API is running at http://$(curl -s ifconfig.me):8000"
echo " Health check: curl http://$(curl -s ifconfig.me):8000/health"
echo "=================================================="
