#!/usr/bin/env bash
# Install zeitenwende-sgns and extraceval for use without internet access.
# Run as root on the target machine while it still has internet access.
set -euo pipefail

usage() {
    echo "Usage: sudo $0 <group> [prefix]" >&2
    echo "  group   group whose members may use and change the installation" >&2
    echo "  prefix  installation directory (default: /opt)" >&2
    exit 1
}

die() {
    echo "Error: $*" >&2
    exit 1
}

GROUP=${1:-}
PREFIX=${2:-/opt}
[[ -n $GROUP ]] || usage

PROFILE=/etc/profile.d/zeitenwende.sh
REPOS=(
    "zeitenwende-sgns https://github.com/pdl-lex/zeitenwende-sgns.git"
    "extraceval https://github.com/fsimonjetz/extraceval.git"
)

[[ $EUID -eq 0 ]] || die "must run as root"
getent group "$GROUP" >/dev/null || die "group '$GROUP' does not exist"
for cmd in git curl; do
    command -v "$cmd" >/dev/null || die "$cmd is not installed"
done

export UV_PYTHON_INSTALL_DIR=$PREFIX/uv/python
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy
export HF_HOME=$PREFIX/hf-cache
unset UV_OFFLINE UV_NO_SYNC HF_HUB_OFFLINE

echo "==> Installing uv"
if [[ ! -x /usr/local/bin/uv ]]; then
    curl -LsSf https://astral.sh/uv/install.sh |
        env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi
UV=/usr/local/bin/uv

for entry in "${REPOS[@]}"; do
    read -r name url <<<"$entry"
    dir=$PREFIX/$name
    echo "==> Setting up $dir"
    if [[ -d $dir/.git ]]; then
        git -C "$dir" pull --ff-only
    else
        git clone "$url" "$dir"
    fi
    git config --system --get-all safe.directory | grep -qxF "$dir" ||
        git config --system --add safe.directory "$dir"
    git -C "$dir" config core.sharedRepository group
    (cd "$dir" && "$UV" sync --locked)
done

echo "==> Checking the spaCy model"
(cd "$PREFIX/zeitenwende-sgns" &&
    "$UV" run --no-sync python -c 'import spacy; spacy.load("de_core_news_sm")')

echo "==> Downloading extraction models"
mkdir -p "$HF_HOME"
(cd "$PREFIX/extraceval" && "$UV" run --no-sync extraceval init)

echo "==> Setting permissions"
shared=("$PREFIX/zeitenwende-sgns" "$PREFIX/extraceval" "$HF_HOME")
chgrp -R "$GROUP" "${shared[@]}"
chmod -R g+rwX "${shared[@]}"
find "${shared[@]}" -type d -exec chmod g+s {} +
chmod -R a+rX "$PREFIX/uv"

echo "==> Writing $PROFILE"
cat >"$PROFILE" <<EOF
export UV_PYTHON_INSTALL_DIR=$PREFIX/uv/python
export HF_HOME=$PREFIX/hf-cache
export UV_OFFLINE=1 UV_NO_SYNC=1 HF_HUB_OFFLINE=1
EOF

cat <<EOF

Done. Remaining steps:
  - Log in again (or 'source $PROFILE') to pick up the environment.
  - Provide a notebook interface (JupyterLab, or VS Code with the Python and Jupyter extensions).
  - Copy the raw data to $PREFIX/zeitenwende-sgns/data/raw/<source>/.
EOF
