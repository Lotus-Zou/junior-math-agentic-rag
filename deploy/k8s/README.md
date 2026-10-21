# Production Deployment

The manifests describe the service topology, but production data services must be
provided by HA MySQL and Redis deployments or managed services. The checked-in
single-replica StatefulSets are a development fallback, not an HA guarantee.

Before a release:

1. Build the application and gateway images with `tools/build_application_images.sh`; it emits a digest-pinned Kustomize overlay. Scan those images and the locked base/GPU images before rollout. The checked-in artifact evidence is under `artifacts/`; each `pinned` value is the immutable deployment reference.
2. Create `math-agent-secrets` from `secret.example.yaml`; never apply the example file as-is.
3. Use HA MySQL with backups and point `MYSQL_URL` at its writer endpoint. Use Redis Sentinel/Cluster or a managed Redis endpoint.
4. Provide an RWX trace volume or replace JSONL traces with object storage/OTel. Do not share a writable Chroma database between replicas unless the storage layer supports the database's locking semantics.
5. Apply `namespace.yaml`, the secrets, dependencies, model inference pool, app, gRPC, worker, gateway and ingress manifests.
   The Kustomization also includes `index-job.yaml`. Wait for
   `kubectl wait -n math-agent --for=condition=complete job/math-agent-index-v1 --timeout=30m`
   before sending traffic. Stop application readers before rebuilding a local
   Chroma index; version the new Job and `KNOWLEDGE_INDEX_VERSION` together.
   Switching PVC access modes between base and single-GPU profiles requires a
   storage migration or fresh namespace, not an in-place `kubectl apply`.
6. Run the GPU image scan and model artifact checks before rollout:

```bash
tools/build_application_images.sh
./tools/scan_gpu_images.sh
sha256sum /usr/local/bin/k3s
```

`scan_gpu_images.sh` exits nonzero when Trivy is missing, when a registry is
unreachable, or when a HIGH/CRITICAL finding is present. A digest lookup is
integrity evidence; it is not a vulnerability or signature approval.

7. Run the load test against the actual ingress and archive the report:

```bash
python tools/load_test.py --url https://math-agent.example.com/ask \
  --requests 5000 --concurrency 300 --secret "$AUTH_SHARED_SECRET" \
  --output evaluation/reports/load-$(date +%Y%m%d%H%M%S).json
```

Do not describe the system as having stable peak capacity until the report records
the target request mix, QPS, p95/p99, error rate, 429 rate, pod CPU/memory, model
provider saturation, Redis latency, MySQL pool wait and recovery behavior during
pod termination.

## Real GPU node

The repository includes `tools/bootstrap_gpu_node.sh`. It is intentionally a
root-only installer because a real node needs system containerd/kubelet,
`net.ipv4.ip_forward=1`, NVIDIA Container Toolkit and access to
`/var/lib/kubelet/device-plugins`. It refuses to run beside an existing k3s
service and verifies the pinned k3s binary before enabling the service:

```bash
sudo env USE_CHINA_MIRRORS=1 bash tools/bootstrap_gpu_node.sh
export KUBECONFIG=/etc/rancher/k3s/math-agent-gpu.kubeconfig
GPU_PROFILE=single-gpu MIN_FREE_GIB=50 tools/deploy_gpu_stack.sh
```

`USE_CHINA_MIRRORS=1` persistently configures Tsinghua Ubuntu, pip, and Conda
mirrors; npm, Go, and Hugging Face mirror environment variables for login
shells; a GitHub proxy for the NVIDIA package repository and k3s release binary;
and Daocloud container registry mirrors. Existing Ubuntu APT sources are backed
up under `/var/backups/math-agent-mirrors` before official Ubuntu URLs are
rewritten. The NVIDIA signing key remains fetched from its official host, and
the k3s binary is still checked against the pinned SHA-256 digest. User-level
package-manager settings may override system defaults. Set `K3S_BINARY` to use
a previously reviewed local binary instead of downloading it. There is no
single mirror that transparently covers arbitrary download URLs.

The checked-in base manifests use `math-agent-ai:local` and
`math-agent-gateway:local` only so Kustomize can render without a private
registry. They are not production image identities. The build script must
produce `deploy/overlays/local-build`; `deploy_gpu_stack.sh` refuses to deploy
without that digest-pinned overlay. Capacity jobs use
`deploy/overlays/capacity-test` so the load-generator image resolves to the
same pinned AI image. The single-GPU overlay runs one Qwen2.5-7B-Instruct replica. The model is
downloaded from a fixed ModelScope commit and every file is checked against
`deploy/k8s/model-artifacts/manifest.json` before vLLM starts. The multi-GPU
base manifests use one replica per GPU and require a topology with at least
two schedulable GPU nodes for the configured two replicas.

The stack contains NVIDIA Device Plugin, Metrics Server, Prometheus,
Prometheus Adapter, queue metrics (`vllm_num_requests_waiting`) and an HPA
that scales the model pool from queue depth. `tools/verify_gpu_stack.py`
executes a real request inside each ready model Pod, checks `nvidia-smi`,
checks vLLM metrics, and queries both resource and custom metrics APIs.

## Local control-plane validation without Docker

On a host without Docker/Podman, Kubernetes SIG's KWOK binary runtime can bring
up a user-space apiserver, etcd, scheduler, controller-manager and simulated
nodes. It validates manifests, scheduling, controllers and health objects, but
does not execute application images or provide real Service networking. The
repository's local sequence is:

```bash
kwokctl --name math-agent create cluster --runtime=binary --wait=180s \
  --kube-apiserver-port=16443 --etcd-port=12379 \
  --controller-port=10257 --kube-scheduler-port=10259
kwokctl --name math-agent scale node --replicas 1
kubectl config use-context kwok-math-agent
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/secret.example.yaml
kubectl apply -f deploy/k8s/kwok-storage.yaml
kubectl apply -f deploy/k8s/dependencies.yaml
kubectl apply -f deploy/k8s/app.yaml
kubectl apply -f deploy/k8s/grpc.yaml
kubectl apply -f deploy/k8s/worker.yaml
kubectl apply -f deploy/k8s/gateway.yaml
kubectl apply -f deploy/k8s/ingress.yaml
```

The static hostPath volumes in `kwok-storage.yaml` are only for this simulated
cluster. A production deployment must replace them with CSI-backed storage.
`tools/gpu_cluster_preflight.py` rejects KWOK before a GPU capacity test.

## Independent model pool

`model-inference.yaml` is the production model-pool contract: two GPU-backed
vLLM replicas, continuous batching, prefix caching, a ClusterIP Service, PDB
and queue-based HPA. It requires a real GPU node pool, NVIDIA device plugin, a
reviewed digest-pinned vLLM image and Prometheus Adapter metrics. The KWOK
cluster must not be used to claim that this pool is running.

The in-cluster capacity Job and collection wrapper are:

```bash
kubectl apply -f deploy/k8s/capacity-test-job.yaml
./tools/run_k8s_capacity_test.sh
```
