#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Mnemosyne — build.sh
#
#  Builds CLI, API server, and optionally the Tauri GUI.
#  sqlite-vector is loaded at *runtime* from ~/.mnemosyne/lib/ — no compile
#  flag required.  See --sqlite-vector to auto-download the extension.
#
#  Usage:
#    ./scripts/build.sh [OPTIONS]
#
#  Options:
#    --release           Release build (default)
#    --dev               Debug build (fast compile, no optimisations)
#    --candle            Enable real BERT text embeddings (candle-backend)
#    --clip              Enable CLIP image embeddings  (clip-backend)
#    --whisper           Enable Whisper audio transcription (whisper-backend)
#    --full              Enable candle + clip + whisper
#    --gui               Also build the Tauri desktop GUI
#    --sqlite-vector     Download sqlite_vector.dylib/.so to ~/.mnemosyne/lib/
#    -h, --help          Show this help
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOLD=$'\033[1m';  RESET=$'\033[0m'
RED=$'\033[31m';  GREEN=$'\033[32m';  YELLOW=$'\033[33m';  CYAN=$'\033[36m'

info()    { echo "${CYAN}▶  $*${RESET}"; }
success() { echo "${GREEN}✓  $*${RESET}"; }
warn()    { echo "${YELLOW}⚠  $*${RESET}"; }
error()   { echo "${RED}✗  $*${RESET}" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
BUILD_PROFILE="release"
FEAT_CANDLE=false
FEAT_CLIP=false
FEAT_WHISPER=false
BUILD_GUI=false
INSTALL_SQLITE_VECTOR=false

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  sed -n '/^#  Usage:/,/^# ──/p' "$0" | head -n -1 | sed 's/^# \{0,3\}//'
  exit 0
}

for arg in "$@"; do
  case "$arg" in
    --release)          BUILD_PROFILE="release" ;;
    --dev)              BUILD_PROFILE="debug"   ;;
    --candle)           FEAT_CANDLE=true ;;
    --clip)             FEAT_CLIP=true; FEAT_CANDLE=true ;;
    --whisper)          FEAT_WHISPER=true; FEAT_CANDLE=true ;;
    --full)             FEAT_CANDLE=true; FEAT_CLIP=true; FEAT_WHISPER=true ;;
    --gui)              BUILD_GUI=true ;;
    --sqlite-vector)    INSTALL_SQLITE_VECTOR=true ;;
    -h|--help)          usage ;;
    *) warn "Unknown option: $arg (ignored)" ;;
  esac
done

# ── Prerequisites ─────────────────────────────────────────────────────────────
info "Checking prerequisites..."
command -v cargo &>/dev/null || error "Rust/Cargo not found — install from https://rustup.rs"
RUST_VER=$(rustc --version | awk '{print $2}')
success "Rust $RUST_VER"

# ── Feature flags ─────────────────────────────────────────────────────────────
# Features are forwarded via mnemosyne-cli's [features] section.
# mnemosyne-api hard-codes candle/clip/whisper in its Cargo.toml, so it needs
# no extra flags.
CLI_FEATURES=""
if [[ "$FEAT_CANDLE" == "true" ]]; then
  CLI_FEATURES="candle-backend"
fi
if [[ "$FEAT_CLIP" == "true" ]]; then
  CLI_FEATURES="${CLI_FEATURES:+$CLI_FEATURES,}clip-backend"
fi
if [[ "$FEAT_WHISPER" == "true" ]]; then
  CLI_FEATURES="${CLI_FEATURES:+$CLI_FEATURES,}whisper-backend"
fi

PROFILE_FLAG=""
[[ "$BUILD_PROFILE" == "release" ]] && PROFILE_FLAG="--release"

CLI_FLAGS="$PROFILE_FLAG"
[[ -n "$CLI_FEATURES" ]] && CLI_FLAGS="$CLI_FLAGS --features $CLI_FEATURES"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}Build configuration${RESET}"
echo "  Profile  : $BUILD_PROFILE"
echo "  CLI features : ${CLI_FEATURES:-none (stub embedder)}"
echo "  API features : candle-backend, clip-backend, whisper-backend (always on)"
echo "  GUI      : $BUILD_GUI"
echo "  sqlite-vector download : $INSTALL_SQLITE_VECTOR"
echo ""

# ── Note on sqlite-vector ─────────────────────────────────────────────────────
echo "${CYAN}ℹ  sqlite-vector is a runtime extension — no compile flag needed.${RESET}"
echo "   Place sqlite_vector.dylib (macOS) / sqlite_vector.so (Linux) in"
echo "   ~/.mnemosyne/lib/ and Mnemosyne will load it automatically on startup."
echo ""

# ── Build CLI + API server ────────────────────────────────────────────────────
info "Building mnemosyne-cli..."
# shellcheck disable=SC2086
cargo build $CLI_FLAGS -p mnemosyne-cli

info "Building mnemosyne-api (all ML backends always enabled)..."
cargo build $PROFILE_FLAG -p mnemosyne-api

BIN_DIR="target/$BUILD_PROFILE"
success "Binaries ready:"
echo "   CLI    : $BIN_DIR/mnemosyne"
echo "   Server : $BIN_DIR/mnemosyne-server"

# ── Optional: download sqlite-vector extension ────────────────────────────────
if [[ "$INSTALL_SQLITE_VECTOR" == "true" ]]; then
  echo ""
  info "Downloading sqlite-vector extension..."

  LIB_DIR="$HOME/.mnemosyne/lib"
  mkdir -p "$LIB_DIR"

  # sqlite-vec (asg017/sqlite-vec) — provides the vec0 virtual table used for
  # HNSW KNN search.  Releases ship as .tar.gz archives containing vec0.dylib.
  BASE_URL="https://github.com/asg017/sqlite-vec/releases/latest/download"

  case "$(uname -s)" in
    Darwin)
      EXT_FILE="vec0.dylib"
      ARCH=$(uname -m)
      if [[ "$ARCH" == "arm64" ]]; then
        DOWNLOAD_CANDIDATES=(
          "$BASE_URL/sqlite-vec-loadable-macos-aarch64.tar.gz"
          "$BASE_URL/sqlite-vec-0.1.9-loadable-macos-aarch64.tar.gz"
        )
      else
        DOWNLOAD_CANDIDATES=(
          "$BASE_URL/sqlite-vec-loadable-macos-x86_64.tar.gz"
          "$BASE_URL/sqlite-vec-0.1.9-loadable-macos-x86_64.tar.gz"
        )
      fi
      ;;
    Linux)
      EXT_FILE="vec0.so"
      DOWNLOAD_CANDIDATES=(
        "$BASE_URL/sqlite-vec-loadable-linux-x86_64.tar.gz"
        "$BASE_URL/sqlite-vec-0.1.9-loadable-linux-x86_64.tar.gz"
      )
      ;;
    *)
      warn "Auto-download not supported on $(uname -s). Download manually from:"
      warn "  https://github.com/asg017/sqlite-vec/releases"
      EXT_FILE=""
      DOWNLOAD_CANDIDATES=()
      ;;
  esac

  if [[ -n "${EXT_FILE:-}" ]]; then
    TARGET="$LIB_DIR/$EXT_FILE"

    # ── Proxy support ────────────────────────────────────────────────────────
    # Honour any proxy already set in the environment; fall back to common
    # local proxy ports if nothing is set.
    if [[ -z "${https_proxy:-}${HTTPS_PROXY:-}" ]]; then
      for port in 7890 1087 8080; do
        if curl -s --max-time 2 --proxy "http://127.0.0.1:$port" \
            "https://www.google.com" -o /dev/null 2>/dev/null; then
          export https_proxy="http://127.0.0.1:$port"
          export http_proxy="http://127.0.0.1:$port"
          export all_proxy="socks5://127.0.0.1:$port"
          info "Detected local proxy on port $port"
          break
        fi
      done
    fi
    [[ -n "${https_proxy:-}" ]] && info "Using proxy: $https_proxy"

    # ── Try each candidate URL until one succeeds ────────────────────────────
    DOWNLOADED=false
    for url in "${DOWNLOAD_CANDIDATES[@]}"; do
      info "Trying: $(basename "$url") ..."
      # Archives contain vec0.dylib / vec0.so — extract just that file.
      if curl -fsSL --retry 2 --max-time 60 "$url" \
          | tar -xzO "$EXT_FILE" > "$TARGET" 2>/dev/null; then
        if [[ -s "$TARGET" ]]; then
          success "sqlite-vec ($EXT_FILE) saved to $TARGET"
          echo "   URL: $url"
          echo "   KNN vector search will be active on next run."
          DOWNLOADED=true
          break
        fi
      fi
      rm -f "$TARGET"   # clean up empty/partial file
    done

    if [[ "$DOWNLOADED" == "false" ]]; then
      warn "All download candidates failed. Install manually:"
      warn "  mkdir -p ~/.mnemosyne/lib"
      warn "  # Download from: https://github.com/asg017/sqlite-vec/releases"
      warn "  # Extract $EXT_FILE and place it at: $TARGET"
    fi
  fi
fi

# ── Optional: Tauri GUI ───────────────────────────────────────────────────────
if [[ "$BUILD_GUI" == "true" ]]; then
  echo ""
  command -v node &>/dev/null || error "Node.js not found — required for GUI build (https://nodejs.org)"
  info "Installing npm dependencies..."
  npm install --prefer-offline
  info "Building Tauri desktop app..."
  cargo tauri build

  APP="target/release/bundle/macos/Mnemosyne.app"
  ENT="src-tauri/entitlements.plist"
  if [[ -d "$APP" ]]; then
    info "Injecting TCC permission descriptions..."
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
    success "GUI bundle ready: target/release/bundle/"
    tccutil reset All com.mnemosyne.app 2>/dev/null || true
  fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────"
success "Build complete!"
echo ""
echo "${BOLD}Quick start:${RESET}"
echo "  ./$BIN_DIR/mnemosyne index ~/Documents"
echo "  ./$BIN_DIR/mnemosyne search \"machine learning papers\""
echo "  ./$BIN_DIR/mnemosyne-server"
echo ""
