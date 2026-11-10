#!/bin/bash
set -e

# 1. Unzip the dataset
7z x data/primevul.7z -odata/ -y

# 2. Set repository details (Override via ENV or edit defaults below)
REPO_OWNER_REPO="gae-m/codesage_ft"
TARGET_DIR="./codesage_ft"

# Wait for DNS and network to become available (up to 30 seconds)
echo "Testing internet connectivity..."
max_retries=10
count=0
until ping -c 1 github.com > /dev/null 2>&1 || [ $count -eq $max_retries ]; do
  echo "Waiting for network/DNS resolution..."
  sleep 2
  count=$((count+1))
done

if [ $count -eq $max_retries ]; then
  echo "ERROR: Network or DNS resolution failed." >&2
  exit 1
fi

echo "Network connected. Proceeding with git clone..."

# 3. Clone or pull the repository
git clone "https://github.com/${REPO_OWNER_REPO}.git" "$TARGET_DIR"
# if [ ! -d "$TARGET_DIR" ]; then
#   echo "Cloning private repository: ${REPO_OWNER_REPO}..."
#   git clone "https://${GH_TOKEN}@github.com/${REPO_OWNER_REPO}.git" "$TARGET_DIR"
# else
#   echo "Directory $TARGET_DIR already exists. Updating..."
#   cd "$TARGET_DIR"
#   git pull
# fi

# 4. Install Python dependencies
pip install --no-cache-dir -r $TARGET_DIR/requirements.txt

# 5. Hand off execution to RunPod's standard entrypoint
echo "Initialization complete. Launching RunPod services..."
exec /start.sh "$@"