# Gh Agent Backend on NST K3s

## 1) SSH + clone/pull repo

```bash
ssh nst-n1
cd ~/git-agent
git pull origin main
```

## 2) Build image

```bash
docker build -f Backend/Dockerfile -t gh-agent:v1 .
```

## 3) Tag + push to local registry

```bash
docker tag gh-agent:v1 localhost:30500/gh-agent:v1
docker push localhost:30500/gh-agent:v1
```

## 4) Create namespace (first time only)

```bash
kubectl create namespace gh-agent
```

## 5) Create secrets with API keys

```bash
kubectl -n gh-agent create secret generic gh-agent-secrets \
  --from-literal=GITHUB_TOKEN="YOUR_GITHUB_TOKEN" \
  --from-literal=llm_api_key="YOUR_LLM_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 6) Apply Kubernetes manifests

```bash
kubectl apply -f Backend/deploy/k8s/backend/namespace.yaml
kubectl apply -f Backend/deploy/k8s/backend/deployment.yaml
kubectl apply -f Backend/deploy/k8s/backend/service.yaml
kubectl apply -f Backend/deploy/k8s/backend/ingress.yaml
```

## 7) Verify

```bash
kubectl get pods -n gh-agent
kubectl get ingress -n gh-agent
```

## 8) Test API

```bash
curl -i -H "Host: gh-agent.nstsdc.org" http://127.0.0.1
curl -i -H "Host: gh-agent.nstsdc.org" http://127.0.0.1/health
```

## Update to new image version

For `v2`, `v3`, etc:

```bash
docker build -f Backend/Dockerfile -t gh-agent:v2 .
docker tag gh-agent:v2 localhost:30500/gh-agent:v2
docker push localhost:30500/gh-agent:v2
kubectl -n gh-agent set image deploy/gh-agent gh-agent=localhost:30500/gh-agent:v2
kubectl -n gh-agent rollout status deploy/gh-agent
```
