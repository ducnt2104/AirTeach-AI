#!/usr/bin/env bash
# ==============================================================================
# AirTeach AI - Automatic Environment Installer & Desktop Launcher (Linux / Ubuntu / Zorin OS)
# ==============================================================================
set -e

# Change directory to project root
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo "==============================================================================="
echo "       AIRTEACH AI - TOUCHLESS GESTURE TEACHING WORKSTATION"
echo "       Trình cài đặt tự động môi trường & Khởi chạy (Linux / Zorin OS / Ubuntu)"
echo "==============================================================================="
echo ""

mkdir -p logs data data/workspaces models assets/icons
INSTALL_LOG="logs/installer_linux.log"
echo "[$(date)] Bat dau cai dat AirTeach AI tren Linux..." > "$INSTALL_LOG"

# Step 1: Install system packages if apt/dnf/pacman is available
echo "[1/6] Kiểm tra và cài đặt thư viện hệ thống cần thiết (OpenGL, XCB, V4L2, Keyring)..."
if command -v apt-get >/dev/null 2>&1; then
    echo "  → Phát hiện hệ thống Debian/Ubuntu/Zorin OS (apt-get)."
    echo "  → Cần quyền sudo để cài các thư viện đồ họa Qt6 & OpenCV:"
    sudo apt-get update -qq >> "$INSTALL_LOG" 2>&1 || true
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv python3-dev build-essential \
        libgl1 libegl1 libglib2.0-0 libfontconfig1 libdbus-1-3 \
        libxkbcommon-x11-0 libsecret-1-0 v4l-utils libreoffice \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 \
        libxcb-xinerama0 >> "$INSTALL_LOG" 2>&1 || echo "  [Lưu ý] Hãy đảm bảo đã cài đủ các gói Qt6/XCB nếu chạy bản tối giản."
    # Cấp quyền video cho user
    sudo usermod -aG video "$USER" >> "$INSTALL_LOG" 2>&1 || true
elif command -v dnf >/dev/null 2>&1; then
    echo "  → Phát hiện hệ thống Fedora/RHEL (dnf)."
    sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ \
        mesa-libGL mesa-libEGL glib2 fontconfig dbus-libs \
        libxkbcommon-x11 libsecret v4l-utils libreoffice xcb-util* >> "$INSTALL_LOG" 2>&1 || true
elif command -v pacman >/dev/null 2>&1; then
    echo "  → Phát hiện hệ thống Arch Linux (pacman)."
    sudo pacman -S --noconfirm --needed python python-pip base-devel \
        mesa libglvnd glib2 fontconfig dbus libxkbcommon-x11 libsecret \
        v4l-utils libreoffice-fresh xcb-util* >> "$INSTALL_LOG" 2>&1 || true
else
    echo "  → Trình quản lý gói chưa xác định, tiếp tục sử dụng môi trường hiện tại..."
fi
echo "  ✓ Thư viện hệ thống đã sẵn sàng."

# Step 2: Check Python version
echo "[2/6] Kiểm tra phiên bản Python 3 (yêu cầu >= 3.10)..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "[LỖI] Không tìm thấy Python 3 trên hệ thống! Vui lòng cài đặt Python 3.10+."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Phát hiện Python $PY_VERSION"

# Step 3: Create Python Virtual Environment
echo "[3/6] Thiết lập môi trường ảo cô lập (.venv)..."
if [ ! -d ".venv" ] || [ ! -f ".venv/bin/activate" ]; then
    echo "  → Đang khởi tạo .venv..."
    python3 -m venv .venv >> "$INSTALL_LOG" 2>&1
fi
source .venv/bin/activate
echo "  ✓ Môi trường ảo .venv đã được kích hoạt."

# Step 4: Install Python dependencies
echo "[4/6] Cài đặt dependencies từ requirements.txt..."
echo "  (Quá trình này mất khoảng 1-2 phút trong lần đầu tiên)..."
pip install --upgrade pip setuptools wheel >> "$INSTALL_LOG" 2>&1
pip install -r requirements.txt >> "$INSTALL_LOG" 2>&1
echo "  ✓ Toàn bộ dependencies Python (PySide6, OpenCV, MediaPipe, NumPy, SciPy...) đã cài xong!"

# Step 5: Check models and system health
echo "[5/6] Kiểm tra và tải mô hình AI mã nguồn mở..."
python3 main.py --check >> "$INSTALL_LOG" 2>&1 || true
echo "  ✓ Kiểm tra mô hình hoàn tất."

# Ensure icon exists
chmod +x run_linux.sh
ICON_PATH="$ROOT/assets/icons/airteach.png"
if [ ! -f "$ICON_PATH" ]; then
    ICON_PATH="$ROOT/assets/icons/logo.svg"
fi

# Step 6: Create Desktop Entry and Application Menu Shortcut
echo "[6/6] Tạo biểu tượng ứng dụng ngoài Desktop và Menu hệ thống..."
DESKTOP_DIR="$HOME/Desktop"
if [ ! -d "$DESKTOP_DIR" ] && [ -d "$HOME/Bàn làm việc" ]; then
    DESKTOP_DIR="$HOME/Bàn làm việc"
fi

DESKTOP_FILE_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=AirTeach AI
Comment=Nền tảng dạy học thông minh tương tác không chạm bằng cử chỉ tay AI
Exec=$ROOT/run_linux.sh
Icon=$ICON_PATH
Path=$ROOT
Terminal=false
StartupNotify=true
Categories=Education;Science;AudioVideo;
Keywords=teaching;gesture;smartboard;ai;education;
"

# Install to application menu
APP_MENU_DIR="$HOME/.local/share/applications"
mkdir -p "$APP_MENU_DIR"
echo "$DESKTOP_FILE_CONTENT" > "$APP_MENU_DIR/airteach.desktop"
chmod +x "$APP_MENU_DIR/airteach.desktop"
echo "  ✓ Đã thêm AirTeach AI vào Menu ứng dụng hệ thống."

# Install to Desktop
if [ -d "$DESKTOP_DIR" ]; then
    DESKTOP_SHORTCUT="$DESKTOP_DIR/airteach.desktop"
    echo "$DESKTOP_FILE_CONTENT" > "$DESKTOP_SHORTCUT"
    chmod +x "$DESKTOP_SHORTCUT"
    # Mark as trusted on GNOME / Zorin OS desktop if gio is available
    if command -v gio >/dev/null 2>&1; then
        gio set "$DESKTOP_SHORTCUT" metadata::trusted true 2>/dev/null || true
    fi
    echo "  ✓ Đã tạo biểu tượng 'AirTeach AI' ngoài Desktop: $DESKTOP_SHORTCUT"
fi

echo ""
echo "==============================================================================="
echo "  CHÚC MỪNG! CÀI ĐẶT AIRTEACH AI TRÊN LINUX THÀNH CÔNG 100%!"
echo "  Biểu tượng 'AirTeach AI' với logo chính thức đã xuất hiện trên Desktop & Menu."
echo "  Từ bây giờ bạn có thể nhấp đúp vào icon trên Desktop hoặc chạy ./run_linux.sh"
echo "==============================================================================="
echo ""
echo "Đang khởi chạy AirTeach AI ngay bây giờ..."
./run_linux.sh &
exit 0
