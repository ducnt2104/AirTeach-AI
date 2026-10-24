#!/usr/bin/env bash
# ==============================================================================
# AirTeach AI - Automatic Environment Installer & macOS App Bundle Creator
# ==============================================================================
set -e

cd "$(dirname "$0")"
ROOT="$(pwd)"

echo "==============================================================================="
echo "       AIRTEACH AI - TOUCHLESS GESTURE TEACHING WORKSTATION"
echo "       Trình cài đặt tự động môi trường & Tạo Ứng Dụng macOS (.app)"
echo "==============================================================================="
echo ""

mkdir -p logs data data/workspaces models assets/icons
INSTALL_LOG="logs/installer_macos.log"
echo "[$(date)] Bat dau cai dat AirTeach AI tren macOS..." > "$INSTALL_LOG"

# Step 1: Check Python on macOS
echo "[1/5] Kiểm tra môi trường Python 3.10+ trên macOS..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "  → Không tìm thấy python3."
    if command -v brew >/dev/null 2>&1; then
        echo "  → Đang cài đặt Python qua Homebrew..."
        brew install python@3.11 >> "$INSTALL_LOG" 2>&1
    else
        echo "[LỖI] Vui lòng cài đặt Python 3.10+ từ https://www.python.org hoặc Homebrew."
        exit 1
    fi
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Phát hiện Python $PY_VERSION"

# Step 2: Create Python Virtual Environment
echo "[2/5] Thiết lập môi trường ảo cô lập (.venv)..."
if [ ! -d ".venv" ] || [ ! -f ".venv/bin/activate" ]; then
    echo "  → Đang khởi tạo .venv..."
    python3 -m venv .venv >> "$INSTALL_LOG" 2>&1
fi
source .venv/bin/activate
echo "  ✓ Môi trường ảo .venv đã được kích hoạt."

# Step 3: Install dependencies
echo "[3/5] Cài đặt dependencies từ requirements.txt..."
pip install --upgrade pip setuptools wheel >> "$INSTALL_LOG" 2>&1
pip install -r requirements.txt >> "$INSTALL_LOG" 2>&1
echo "  ✓ Toàn bộ thư viện đã được cài đặt thành công!"

# Step 4: Verify models and system
echo "[4/5] Kiểm tra và tải mô hình AI mã nguồn mở..."
python3 main.py --check >> "$INSTALL_LOG" 2>&1 || true
echo "  ✓ Kiểm tra hệ thống hoàn tất."

chmod +x run_macos.sh

# Step 5: Create native macOS App Bundle (AirTeach AI.app)
echo "[5/5] Đang tạo ứng dụng native macOS 'AirTeach AI.app'..."
APP_BUNDLE="$ROOT/AirTeach AI.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"

mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Copy ICNS icon
if [ -f "$ROOT/assets/icons/airteach.icns" ]; then
    cp "$ROOT/assets/icons/airteach.icns" "$RESOURCES_DIR/airteach.icns"
fi

# Write Info.plist
cat << 'EOF' > "$CONTENTS/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>AirTeachAI</string>
    <key>CFBundleIconFile</key>
    <string>airteach.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.airteach.ai</string>
    <key>CFBundleName</key>
    <string>AirTeach AI</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>AirTeach AI cần quyền camera để nhận diện cử chỉ tay không chạm khi giảng dạy.</string>
</dict>
</plist>
EOF

# Write executable launcher script inside App Bundle
cat << EOF > "$MACOS_DIR/AirTeachAI"
#!/usr/bin/env bash
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
exec python3 "$ROOT/main.py" "\$@"
EOF

chmod +x "$MACOS_DIR/AirTeachAI"

# Create shortcut on ~/Desktop
if [ -d "$HOME/Desktop" ]; then
    rm -rf "$HOME/Desktop/AirTeach AI.app" 2>/dev/null || true
    cp -R "$APP_BUNDLE" "$HOME/Desktop/AirTeach AI.app"
    echo "  ✓ Đã sao chép 'AirTeach AI.app' ra Desktop: $HOME/Desktop/AirTeach AI.app"
fi

echo ""
echo "==============================================================================="
echo "  CHÚC MỪNG! CÀI ĐẶT AIRTEACH AI TRÊN MACOS THÀNH CÔNG 100%!"
echo "  Ứng dụng 'AirTeach AI.app' với biểu tượng chính thức đã có sẵn trên Desktop."
echo "  Bạn có thể nhấp đúp vào icon Desktop hoặc kéo vào Dock/Applications để dùng!"
echo "==============================================================================="
echo ""
echo "Đang khởi chạy AirTeach AI..."
open "$APP_BUNDLE" || ./run_macos.sh
exit 0
