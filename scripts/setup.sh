#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Mnemosyne — setup.sh
#  Compiles the project and downloads required embedding models.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD=$'\033[1m';  RESET=$'\033[0m'
RED=$'\033[31m';  GREEN=$'\033[32m';  YELLOW=$'\033[33m';  CYAN=$'\033[36m'

info()    { echo "${CYAN}▶  $*${RESET}"; }
success() { echo "${GREEN}✓  $*${RESET}"; }
warn()    { echo "${YELLOW}⚠  $*${RESET}"; }
error()   { echo "${RED}✗  $*${RESET}"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cat <<'BANNER'
 __  __
|  \/  |_ __   ___ _ __ ___   ___  ___ _   _ _ __   ___
| |\/| | '_ \ / _ \ '_ ` _ \ / _ \/ __| | | | '_ \ / _ \
| |  | | | | |  __/ | | | | | (_) \__ \ |_| | | | |  __/
|_|  |_|_| |_|\___|_| |_| |_|\___/|___/\__, |_| |_|\___|
                                         |___/
 Intelligent Local File Search  ·  setup.sh
BANNER
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────
info "Checking prerequisites..."
command -v cargo &>/dev/null || error "Rust/Cargo not found. Install via https://rustup.rs"
command -v rustc &>/dev/null
RUST_VER=$(rustc --version | awk '{print $2}')
success "Rust $RUST_VER"

# ── Parse flags ───────────────────────────────────────────────────────────────
CANDLE=false
SKIP_MODEL=false
SKIP_GUI=false
BUILD_TYPE="release"

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --candle          Build with real BERT inference (larger, slower build)"
  echo "  --skip-model      Skip model download"
  echo "  --skip-gui        Skip Tauri GUI build"
  echo "  --dev             Build in debug mode (faster compile)"
  echo "  -h, --help        Show this help"
  exit 0
}

for arg in "$@"; do
  case "$arg" in
    --candle)      CANDLE=true ;;
    --skip-model)  SKIP_MODEL=true ;;
    --skip-gui)    SKIP_GUI=true ;;
    --dev)         BUILD_TYPE="debug" ;;
    -h|--help)     usage ;;
    *) warn "Unknown option: $arg" ;;
  esac
done

# ── Build CLI + API server ────────────────────────────────────────────────────
info "Building CLI and API server (profile: $BUILD_TYPE)..."

CARGO_OPTS=""
FEATURE_FLAGS=""
[[ "$BUILD_TYPE" == "release" ]] && CARGO_OPTS="--release"

if [[ "$CANDLE" == "true" ]]; then
  warn "candle-backend enabled: this requires downloading ~300MB of LLVM/build tools."
  warn "Build may take 10–20 minutes on first run."
  FEATURE_FLAGS="--features mnemosyne-model/candle-backend"
fi

cargo build $CARGO_OPTS -p mnemosyne-cli -p mnemosyne-api $FEATURE_FLAGS

if [[ "$BUILD_TYPE" == "release" ]]; then
  BIN_DIR="target/release"
else
  BIN_DIR="target/debug"
fi

success "Binaries built:"
echo "   CLI:  $BIN_DIR/mnemosyne"
echo "   API:  $BIN_DIR/mnemosyne-server"

# ── Install to ~/.local/bin (optional) ────────────────────────────────────────
INSTALL_DIR="$HOME/.local/bin"
if [[ -d "$INSTALL_DIR" ]]; then
  read -r -p "Install binaries to $INSTALL_DIR? [y/N] " REPLY
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    cp "$BIN_DIR/mnemosyne"        "$INSTALL_DIR/mnemosyne"
    cp "$BIN_DIR/mnemosyne-server" "$INSTALL_DIR/mnemosyne-server"
    success "Installed to $INSTALL_DIR"
  fi
fi

# ── Download models ───────────────────────────────────────────────────────────
if [[ "$SKIP_MODEL" == "false" ]]; then
  echo ""
  info "Downloading default text embedding model..."
  echo "   Model : sentence-transformers/all-MiniLM-L6-v2"
  echo "   Size  : ~90 MB"
  echo "   Cache : ~/.mnemosyne/models/"
  echo ""

  if ! "$BIN_DIR/mnemosyne" model-download sentence-transformers/all-MiniLM-L6-v2; then
    warn "Model download failed. You can retry later with:"
    warn "  mnemosyne model-download sentence-transformers/all-MiniLM-L6-v2"
    warn "Or set HF_ENDPOINT to a mirror (e.g., https://hf-mirror.com)"
  else
    success "Text embedding model ready"
  fi

  # Download CLIP if clip-backend was enabled
  if [[ "$CANDLE" == "true" ]]; then
    info "Downloading CLIP image embedding model..."
    if ! "$BIN_DIR/mnemosyne" model-download openai/clip-vit-base-patch32; then
      warn "CLIP model download failed (optional — image search will use text fallback)"
    else
      success "CLIP model ready"
    fi
  fi
fi

# ── Build GUI (optional) ──────────────────────────────────────────────────────
if [[ "$SKIP_GUI" == "false" ]]; then
  if command -v node &>/dev/null && command -v npm &>/dev/null; then
    echo ""
    NODE_VER=$(node --version)
    info "Node.js $NODE_VER found"
    read -r -p "Build Tauri desktop GUI? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
      info "Installing npm dependencies..."
      npm install --prefer-offline
      info "Building Tauri app..."
      cargo tauri build

      # ── Post-build: inject TCC usage descriptions + re-sign ──────────────
      APP="target/release/bundle/macos/Mnemosyne.app"
      ENT="src-tauri/entitlements.plist"
      if [[ -d "$APP" ]]; then
        info "Injecting TCC permission descriptions into Info.plist..."
        for kv in \
          "NSDownloadsFolderUsageDescription:Mnemosyne 需要访问下载文件夹以索引您的文件。" \
          "NSDocumentsFolderUsageDescription:Mnemosyne 需要访问文稿文件夹以索引您的文件。" \
          "NSDesktopFolderUsageDescription:Mnemosyne 需要访问桌面以索引您的文件。"
        do
          k="${kv%%:*}"; v="${kv#*:}"
          /usr/libexec/PlistBuddy -c "Delete :$k" "$APP/Contents/Info.plist" 2>/dev/null || true
          /usr/libexec/PlistBuddy -c "Add :$k string '$v'" "$APP/Contents/Info.plist"
        done
        codesign --force --deep --sign "-" --entitlements "$ENT" "$APP"
        success "GUI bundle ready in target/release/bundle/"
        
        # Reset TCC so macOS shows fresh permission dialogs on first launch
        tccutil reset All com.mnemosyne.app 2>/dev/null || true
        info "TCC permissions reset — app will request access on first launch"
      fi
    fi
  else
    warn "Node.js not found — skipping GUI build."
    warn "Install Node.js ≥ 18 from https://nodejs.org to build the desktop app."
  fi
fi

# ── Data directory ────────────────────────────────────────────────────────────
mkdir -p "$HOME/.mnemosyne/lib"

# ── sqlite-vector (optional) ──────────────────────────────────────────────────
echo ""
info "Optional: sqlite-vec extension for KNN vector search (asg017/sqlite-vec)"
echo "   Run the following to install (macOS arm64):"
echo ""
echo "   mkdir -p ~/.mnemosyne/lib"
echo "   curl -fsSL https://github.com/asg017/sqlite-vec/releases/latest/download/sqlite-vec-0.1.9-loadable-macos-aarch64.tar.gz \\"
echo "     | tar -xzO vec0.dylib > ~/.mnemosyne/lib/vec0.dylib"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────"
success "Setup complete!"
echo ""
echo "${BOLD}Quick start:${RESET}"
echo ""
echo "  # Index your documents"
echo "  ./$BIN_DIR/mnemosyne index ~/Documents"
echo ""
echo "  # Search"
echo "  ./$BIN_DIR/mnemosyne search \"machine learning papers\""
echo ""
echo "  # Start REST API server"
echo "  ./$BIN_DIR/mnemosyne serve"
echo ""
echo "  # Watch a directory for changes"
echo "  ./$BIN_DIR/mnemosyne watch ~/Documents"
echo ""
echo "${BOLD}Mirrors (if HuggingFace is blocked):${RESET}"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
echo "  ./$BIN_DIR/mnemosyne model-download sentence-transformers/all-MiniLM-L6-v2"
echo ""
