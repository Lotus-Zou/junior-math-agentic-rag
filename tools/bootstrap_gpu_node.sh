#!/usr/bin/env bash
# Example: sudo env USE_CHINA_MIRRORS=1 bash tools/bootstrap_gpu_node.sh
# Install a real, isolated k3s node. Run on the GPU host as root.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$EUID" -ne 0 ]]; then
  echo 'Root is required to install the runtime, enable forwarding and start kubelet.' >&2
  exit 1
fi
if systemctl is-active --quiet k3s || systemctl is-active --quiet k3s-agent; then
  echo 'An existing k3s service is active; use its administrator-managed GPU runtime instead.' >&2
  exit 1
fi
if [[ "$(uname -m)" != x86_64 ]]; then
  echo 'This audited binary targets linux/amd64.' >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT
nvidia_repo_url='https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH)'
k3s_release_base='https://github.com/k3s-io/k3s/releases/download'
if [[ "${USE_CHINA_MIRRORS:-0}" == 1 ]]; then
  . /etc/os-release
  if [[ "${ID:-}" != ubuntu || -z "${VERSION_CODENAME:-}" ]]; then
    echo 'China APT mirror mode currently requires Ubuntu with VERSION_CODENAME.' >&2
    exit 1
  fi
  mirror_backup_root=/var/backups/math-agent-mirrors
  backup_config() {
    local source="$1"
    local backup="$mirror_backup_root$source"
    if [[ -e "$source" && ! -e "$backup" ]]; then
      install -d -m 755 "$(dirname "$backup")"
      cp -a -- "$source" "$backup"
    fi
  }
  for source in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
    [[ -f "$source" ]] || continue
    if grep -Eq 'archive\.ubuntu\.com/ubuntu|security\.ubuntu\.com/ubuntu' "$source"; then
      backup_config "$source"
      sed -i -E \
        -e 's#https?://([[:alnum:]-]+\.)?archive\.ubuntu\.com/ubuntu/?#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
        -e 's#https?://security\.ubuntu\.com/ubuntu/?#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
        "$source"
    fi
  done
  nvidia_repo_url='https://gh-proxy.com/https://raw.githubusercontent.com/NVIDIA/libnvidia-container/gh-pages/stable/deb/$(ARCH)'
  k3s_release_base='https://gh-proxy.com/https://github.com/k3s-io/k3s/releases/download'

  backup_config /etc/pip.conf
  cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF
  chmod 644 /etc/pip.conf

  backup_config /etc/conda/.condarc
  install -d -m 755 /etc/conda
  cat > /etc/conda/.condarc <<'EOF'
channels:
  - conda-forge
  - defaults
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
show_channel_urls: true
EOF
  chmod 644 /etc/conda/.condarc

  backup_config /etc/profile.d/90-math-agent-mirrors.sh
  cat > /etc/profile.d/90-math-agent-mirrors.sh <<'EOF'
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export npm_config_registry=https://registry.npmmirror.com
export GOPROXY=https://goproxy.cn
export HF_ENDPOINT=https://hf-mirror.com
EOF
  chmod 644 /etc/profile.d/90-math-agent-mirrors.sh
fi
apt-get update
apt-get install -y ca-certificates curl gnupg iptables conntrack socat

curl -fSL --connect-timeout 15 --max-time 60 --retry 2 \
  https://nvidia.github.io/libnvidia-container/gpgkey -o "$work_dir/nvidia.gpg"
gpg --batch --yes --dearmor -o "$work_dir/nvidia-keyring.gpg" "$work_dir/nvidia.gpg"
install -m 644 "$work_dir/nvidia-keyring.gpg" /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
printf 'deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] %s /\n' "$nvidia_repo_url" \
  > "$work_dir/nvidia-container-toolkit.list"
if [[ "${USE_CHINA_MIRRORS:-0}" == 1 ]]; then
  backup_config /etc/apt/sources.list.d/nvidia-container-toolkit.list
fi
install -m 644 "$work_dir/nvidia-container-toolkit.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
nvidia-container-cli --version
nvidia-ctk --version

# Digest is from the official k3s-io/k3s GitHub release asset metadata.
k3s_digest=92b94582eb7b34a8cf532e6006842b9ed215a6783cf56a3dd6271fd35243b2c0
if [[ -n "${K3S_BINARY:-}" ]]; then
  cp -- "$K3S_BINARY" "$work_dir/k3s"
else
  curl -fSL --connect-timeout 15 --max-time 300 --retry 2 \
    "$k3s_release_base/v1.37.1%2Bk3s1/k3s" \
    -o "$work_dir/k3s"
fi
echo "$k3s_digest  $work_dir/k3s" | sha256sum --check --status
install -m 755 "$work_dir/k3s" /usr/local/bin/k3s
install -d -m 700 /etc/rancher/k3s
install -m 600 "$repo_dir/deploy/node/k3s-gpu-config.yaml" /etc/rancher/k3s/math-agent-gpu-config.yaml
if [[ "${USE_CHINA_MIRRORS:-0}" == 1 ]]; then
  if [[ -f /etc/rancher/k3s/registries.yaml ]]; then
    echo 'Existing registries.yaml is preserved; merge the supplied mirror config manually.' >&2
  else
    install -m 600 "$repo_dir/deploy/node/registries-china.yaml" /etc/rancher/k3s/registries.yaml
  fi
fi
sysctl -w net.ipv4.ip_forward=1
echo 'net.ipv4.ip_forward=1' > /etc/sysctl.d/90-math-agent-k3s.conf
install -m 644 "$repo_dir/deploy/node/math-agent-k3s.service" /etc/systemd/system/math-agent-k3s.service
systemctl daemon-reload
systemctl enable --now math-agent-k3s
export KUBECONFIG=/etc/rancher/k3s/math-agent-gpu.kubeconfig
for attempt in $(seq 1 60); do
  if /usr/local/bin/k3s kubectl get --raw=/readyz >/dev/null 2>&1; then break; fi
  sleep 2
done
/usr/bin/nvidia-ctk runtime configure --runtime=containerd \
  --config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml
systemctl restart math-agent-k3s
sleep 5
/usr/local/bin/k3s kubectl wait --for=condition=Ready node/math-agent-gpu --timeout=180s
if ! rg -q 'nvidia-container-runtime' /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null && \
   ! grep -q 'nvidia-container-runtime' /var/lib/rancher/k3s/agent/etc/containerd/config.toml; then
  echo 'k3s did not detect NVIDIA runtime; inspect the generated containerd configuration.' >&2
  exit 1
fi
/usr/local/bin/k3s kubectl get nodes -o wide
echo 'Real kubelet/containerd installed. GPU allocatable requires the device plugin next.'
echo 'Kubeconfig: /etc/rancher/k3s/math-agent-gpu.kubeconfig (admin credential; keep private).'
