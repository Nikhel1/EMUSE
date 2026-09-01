# Running EMUSE on EASI with Kubernetes

## What this document is about

EMUSE (https://doi.org/10.1017/pasa.2025.10064) is an interactive app for exploring EMU radio-survey image cutouts. This document explains how to package EMUSE into a container image, what Kubernetes is and how it relates to that container, and how to actually get the app running as a live web service on EASI's shared Kubernetes cluster — including how it reads its image cutouts from the S3 bucket set up in the companion document `EASI_storage_S3bucket.md`, and how deployments are kept in sync with a git repository using Flux.

The rough shape of the pipeline is:

1. Code lives in a GitHub repository.
2. It's packaged into a **container image** using Docker, and that image is pushed to a registry (Docker Hub).
3. A set of **Kubernetes manifests** describe how that image should be run, exposed, and reached on EASI's cluster.
4. **Flux** watches a separate admin git repository and automatically applies those manifests to the cluster whenever they change, so redeploying is usually a matter of updating that repository rather than manually re-running commands.

---

## 1. Get the code

The EMUSE source is on GitHub, on the `docker-bucket-data` branch:

```bash
git clone --branch docker-bucket-data https://github.com/Nikhel1/EMUSE.git
cd EMUSE
```

This branch includes the app code and the `Dockerfile` (the recipe Docker uses to build a container image, described below) needed for the next step.

---

## 2. Containerise the app with Docker

A **container** is a lightweight, self-contained package that bundles an application together with everything it needs to run — the code, the Python environment, system libraries, and configuration — so it behaves the same way regardless of what machine it's run on. Docker is the tool used to build and run these containers. This matters for EASI because the cluster has no idea what "EMUSE" is; it only knows how to run container images, so the app has to be packaged into one before it can be deployed there.

### Build and push a basic image

From inside the cloned repository (which contains a `Dockerfile`):

```bash
docker build -t emuse .
docker tag emuse nikhel/emuse:latest
docker push nikhel/emuse:latest
```

- `docker build -t emuse .` reads the `Dockerfile` in the current directory (`.`) and builds a container image from it, naming (tagging) it `emuse` locally.
- `docker tag emuse nikhel/emuse:latest` gives that same image an additional tag in the form `<dockerhub-username>/<repository>:<version>`, which is the naming format Docker Hub (a public registry for container images) expects. `latest` here is just a version label, not a special keyword — it's a convention meaning "the most recent build."
- `docker push nikhel/emuse:latest` uploads the tagged image to Docker Hub, from where the EASI cluster will later pull it.

### Pull and run it locally

Anyone (including EASI's cluster) can now fetch and run that same image:

```bash
docker pull nikhel/emuse:latest
docker run -p 8501:8501 nikhel/emuse:latest
```

`docker pull` downloads the image from Docker Hub. `docker run -p 8501:8501 ...` starts a container from it, mapping port 8501 inside the container to port 8501 on the local machine — 8501 is the default port for Streamlit apps (which is what EMUSE's interface is built with), so this makes the running app reachable at `http://localhost:8501`.

### Building for a specific CPU architecture (Ubuntu/AMD64)

Container images are built for a specific CPU architecture (commonly `amd64`/`x86_64`, used by most cloud servers and Ubuntu machines, or `arm64`, used by Apple Silicon Macs). An image built on an Apple Silicon laptop defaults to `arm64` and won't necessarily run on an `amd64` cloud server like EASI's — hence explicitly targeting the platform:

```bash
docker build --platform=linux/amd64 -t emuse-amd .
docker tag emuse-amd nikhel/emuse-amd:latest
docker push nikhel/emuse-amd:latest
```

```bash
docker pull nikhel/emuse-amd:latest
```

`--platform=linux/amd64` tells Docker to build an image for that specific architecture, regardless of what architecture the machine doing the building actually has (Docker handles the cross-compilation via emulation).

### Building one image that works on both architectures

Rather than maintaining two separate image tags, Docker's `buildx` tool can build and publish a single **multi-platform image** — one tag that automatically serves the right architecture-specific version depending on who's pulling it:

```bash
# 1. Create a dedicated multi-platform buildx builder
docker buildx create --name emuse-builder --driver docker-container --bootstrap --use

# 2. Build for linux/amd64 + linux/arm64 and push directly to Docker Hub
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag nikhel/emuse:latest \
  --push \
  .
```

- `docker buildx create --name emuse-builder --driver docker-container --bootstrap --use` creates a new "builder" instance (`emuse-builder`) capable of multi-architecture builds, starts it up (`--bootstrap`), and switches to using it (`--use`) for subsequent build commands.
- `docker buildx build --platform linux/amd64,linux/arm64 ...` builds the image for both architectures in one go and, because of `--push`, uploads the result straight to Docker Hub under a single `nikhel/emuse:latest` tag — anyone pulling that tag automatically gets the version matching their own machine's architecture.

---

## 3. What is Kubernetes, and how does it relate to Docker?

Docker builds and runs a *single* container on a *single* machine. In practice, running a real application means running many containers (the app itself, dependent services, replicas for reliability) across many machines, keeping them healthy, restarting ones that crash, and routing network traffic to them correctly. **Kubernetes** (often shortened to "k8s") is the system that does all of that automatically across a cluster of machines.

A few core Kubernetes concepts, in the order they build on each other:

- **Pod** — the smallest unit Kubernetes runs; typically one running instance of a container (e.g. one running copy of the EMUSE image).
- **Deployment** — describes how many copies ("replicas") of a pod should be running, which container image to use, and how to roll out updates. Kubernetes continuously works to keep the real state matching what a Deployment describes — if a pod crashes, Kubernetes starts a new one to replace it.
- **Service** — a stable internal network address that routes traffic to whichever pods are currently running for a given app, even as individual pods come and go.
- **Ingress** — configures how traffic from *outside* the cluster (e.g. a public URL) gets routed in to a Service.

Rather than clicking through a UI to configure any of this, Kubernetes is normally controlled by writing **manifest files** — YAML documents that *declare* the desired end state ("I want 1 replica of this image, exposed on this port, reachable at this hostname") rather than a sequence of commands. Kubernetes reads the manifest and continuously makes reality match it. This is the same principle as the Docker image itself: define what should exist, and let the system take care of making it so, and keeping it that way.

---

## 4. The EMUSE Kubernetes manifest, explained

EMUSE is deployed on EASI's cluster using a single manifest file containing three resources, separated by `---`: a **Deployment**, a **Service**, and an **Ingress**.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: emuse
  namespace: csa-parkes-team
  labels:
    app: emuse
    toolkit.fluxcd.io/tenant: csa-parkes-team
spec:
  replicas: 1
  selector:
    matchLabels:
      app: emuse
  template:
    metadata:
      labels:
        app: emuse
    spec:
      containers:
        - name: emuse
          image: docker.io/nikhel/emuse:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8501
              protocol: TCP
          resources:
            requests:
              cpu: 500m
              memory: 4Gi
            limits:
              cpu: "2"
              memory: 8Gi
          livenessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 10
            periodSeconds: 5
```

**Deployment, explained field by field:**

- `namespace: csa-parkes-team` — Kubernetes clusters are subdivided into **namespaces**, which partition resources (and access to them) between different teams or projects sharing the same cluster. This deployment lives in the `csa-parkes-team` namespace.
- `labels` — arbitrary key/value tags attached to a resource; `app: emuse` is used elsewhere (in the Service and Ingress) to find and target this specific set of pods. `toolkit.fluxcd.io/tenant: csa-parkes-team` marks this resource as belonging to that team for Flux's own bookkeeping (Flux is covered below).
- `replicas: 1` — run exactly one copy of the app. Increasing this would run multiple identical copies for load-balancing/redundancy.
- `selector.matchLabels` / `template.metadata.labels` — the Deployment manages any pod matching `app: emuse`, and the pods it creates from `template` carry that same label, tying the two together.
- `image: docker.io/nikhel/emuse:latest` — the exact container image to run, pulled from Docker Hub — this is the image built and pushed in Step 2.
- `imagePullPolicy: Always` — always fetch the image fresh from the registry rather than reusing a locally cached copy, so pushing a new `:latest` build and restarting the deployment picks up the change.
- `ports.containerPort: 8501` — the port the container listens on internally (matching the Streamlit default used in the `docker run` example earlier).
- `resources.requests` / `resources.limits` — `requests` is the amount of CPU/memory guaranteed to the pod (used by Kubernetes to decide which machine to schedule it on); `limits` is the maximum it's allowed to use before being throttled (CPU) or restarted (memory). `500m` means 0.5 of a CPU core; `4Gi`/`8Gi` are gibibytes of memory.
- `livenessProbe` / `readinessProbe` — Kubernetes periodically checks a health-check URL (`/_stcore/health`, Streamlit's built-in health endpoint) on the container. The **liveness** probe determines whether the container should be restarted (if it stops responding, something is wrong internally); the **readiness** probe determines whether the pod should currently receive traffic (useful while the app is still starting up). `initialDelaySeconds` waits before the first check; `periodSeconds` is how often it's repeated afterwards.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: emuse
  namespace: csa-parkes-team
  labels:
    app: emuse
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8501
      protocol: TCP
      name: http
  selector:
    app: emuse
```

**Service, explained:** this creates a stable internal address for the app. `selector: app: emuse` means it automatically routes traffic to whichever pod(s) currently carry that label — even if the Deployment restarts or replaces them. `type: ClusterIP` means the address is only reachable from *inside* the cluster (not directly from the internet) — external access is what the Ingress below is for. `port: 80` is the port other things inside the cluster use to reach this Service; `targetPort: 8501` is where that traffic actually gets sent on the pod (matching the container's port).

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: emuse
  namespace: csa-parkes-team
  labels:
    app: emuse
  annotations:
    alb.ingress.kubernetes.io/healthcheck-path: /_stcore/health
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:ap-southeast-2:444488357543:certificate/1aa65e02-d62f-4a65-a672-1bf084329aba
    external-dns.alpha.kubernetes.io/hostname: emuse.csiro.easi-eo.solutions
spec:
  ingressClassName: easi-group
  rules:
    - host: emuse.csiro.easi-eo.solutions
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: emuse
                port:
                  number: 80
```

**Ingress, explained:** this is what makes the app reachable from outside the cluster at a real URL. The `annotations` configure AWS's Application Load Balancer (ALB) integration specifically: which path to use for health checks, which port to listen on for HTTPS (443), that traffic should be routed directly to pod IPs, and which AWS Certificate Manager (ACM) certificate to use for HTTPS. `external-dns.alpha.kubernetes.io/hostname` tells an external-DNS controller to automatically create a DNS record pointing `emuse.csiro.easi-eo.solutions` at this load balancer. The `rules` section says: for requests arriving for that hostname, on any path (`/`, `Prefix` match), send them to the `emuse` Service on port 80 — the same Service defined above.

---

## 5. Useful `kubectl` commands

`kubectl` is the command-line tool for interacting with a Kubernetes cluster — reading its current state and applying manifests to it. These all run from an EASI JupyterHub terminal (or anywhere else with `kubectl` configured against the EASI cluster), and generally need `-n <namespace>` to target the right namespace (`csa-parkes-team` for this deployment):

```bash
# Apply (create or update) resources from a manifest file
kubectl apply -f emuse-manifest.yaml -n csa-parkes-team

# List pods, and check their status
kubectl get pods -n csa-parkes-team

# List the Deployment, Service, and Ingress
kubectl get deployment,service,ingress -n csa-parkes-team

# Get details about the Ingress, including the load balancer address it's bound to
kubectl get ingress emuse -n csa-parkes-team

# Full details and recent events for a specific resource (useful for diagnosing problems)
kubectl describe deployment emuse -n csa-parkes-team
kubectl describe pod <pod-name> -n csa-parkes-team

# View logs from the running app (add -f to follow/stream them live)
kubectl logs -l app=emuse -n csa-parkes-team -f

# Delete resources — either by file, or by matching label
kubectl delete -f emuse-manifest.yaml -n csa-parkes-team
kubectl delete deployment,service,ingress -l app=emuse -n csa-parkes-team
```

`-l app=emuse` in the last examples selects resources by label rather than by name — handy since the Deployment, Service, and Ingress all share that same `app: emuse` label.

---

## 6. Reading data from S3: an extra permission step under Kubernetes

EMUSE fetches its EMU image cutouts directly from the S3 bucket set up in `EASI_storage_S3bucket.md`, using `boto3` (AWS's Python SDK) to talk to S3 from within the app's code.

When EMUSE runs as a notebook inside EASI's JupyterHub, this "just works," because the bucket policy already grants access to the `easihub-csiro-csiro-easihub-client` role that JupyterHub sessions run under (as described in that document). Running the same app as a Kubernetes Deployment is a different situation: the pod doesn't automatically inherit that JupyterHub identity, so EASI administrators had to set up a separate path for it — creating a dedicated namespace, `csa-parkes-apps`, and adding a **Kubernetes ServiceAccount** to the manifest, which is what lets the pod assume an AWS role with S3 access in its own right (rather than borrowing the JupyterHub session's identity). The configuration for this lives in a separate admin repository: https://github.com/csiro-internal/csa-easi-astro-flux-admin.git, under the `emuse` folder.

---

## 7. Deploying via Flux (GitOps)

Rather than running `kubectl apply` by hand every time something changes, EASI uses **Flux**, a tool that continuously watches a designated git repository and automatically applies whatever manifests it finds there to the cluster — a pattern generally called GitOps. In practice, this means the source of truth for what's actually running is the git repository, not a manual `kubectl` command run once and then forgotten; changing the deployment means changing the repository, and Flux takes care of syncing the cluster to match.

The admin repository for EMUSE's Kubernetes configuration is `csa-easi-astro-flux-admin`. The typical workflow for getting a new or updated deployment live looks like this, run from an EASI JupyterHub terminal:

```bash
git clone https://github.com/csiro-internal/csa-easi-astro-flux-admin.git
cd csa-easi-astro-flux-admin

git fetch origin
git checkout Nikhel1-patch-1   # a personal branch with in-progress changes
git checkout main              # once changes have gone through as a pull request and merged

flux reconcile kustomization apps -n csa-parkes-apps

kubectl delete deployment,service,ingress -l app=emuse -n csa-parkes-apps
```

- `git fetch origin` pulls the latest state of all branches from the remote repository without changing the currently checked-out branch.
- `git checkout Nikhel1-patch-1` switches to a personal feature branch to inspect or test in-progress changes before they're merged.
- `git checkout main` switches back to the `main` branch, which reflects changes that have already been reviewed and merged via a pull request — this is the branch Flux is actually configured to watch and deploy from.
- `flux reconcile kustomization apps -n csa-parkes-apps` manually triggers Flux to immediately re-check the repository and apply any pending changes to the `apps` kustomization in the `csa-parkes-apps` namespace, rather than waiting for its normal automatic sync interval.
- The final `kubectl delete` removes the currently running Deployment, Service, and Ingress for EMUSE — since Flux is watching and will recreate them from the repository's manifests on its next reconcile, this effectively forces a clean redeploy rather than leaving old, possibly stale, resources in place.

---

## Quick troubleshooting checklist

- **Pod stuck in `Pending`** → check `kubectl describe pod <pod-name> -n <namespace>` for scheduling errors (often insufficient CPU/memory available to meet the `resources.requests` values).
- **Pod running but Ingress not reachable** → confirm the Ingress has a load balancer address with `kubectl get ingress emuse -n <namespace>`, and check the ACM certificate ARN and hostname annotations match an actual, valid certificate and DNS entry.
- **App can't read from S3** → check the pod is running with the ServiceAccount configured for S3 access (relevant only to the `csa-parkes-apps`/Kubernetes deployment, not the JupyterHub notebook case), and cross-check the bucket policy from `EASI_storage_S3bucket.md`.
- **Old version still running after a Docker push** → confirm `imagePullPolicy: Always` is set, and that the pod has actually restarted since the new image was pushed (`kubectl rollout restart deployment emuse -n <namespace>` forces this).
- **Changes in the admin repo aren't taking effect** → confirm the change was merged into `main` (the branch Flux watches) and run `flux reconcile kustomization apps -n <namespace>` to force an immediate sync rather than waiting for the next automatic interval.
