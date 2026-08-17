# Rancher でアプリを公開する

> 这是最后的山场:在 Rancher 网页上把 JPeaks-Forest 前台部署起来。前提:镜像已推到 Docker Hub([06](06-レジストリに預ける.md)),K8s 清单已备好([07](07-K8sマニフェスト.md))。
>
> 本章的 `<namespace>`、`<intermediate-ip>`、`<YOUR_NAME>` 等占位符含义见 [00 全体像](00-はじめに-全体像.md) 顶部说明,替换成你环境的实际值。

我们真实的部署对象:

| 资源 | 名字 | 说明 |
|---|---|---|
| Namespace | `<namespace>` | 我们组的区画(已存在,和其他服务共用) |
| Deployment | `ffformer-infer-base` | 网页服务本体(FastAPI,`deploy/server.py`) |
| Service | `ffformer-infer-svc` | 集群内入口(ClusterIP:8000) |
| TLS 代理 | `ffformer-tls` | 独立 nginx,终结 HTTPS |
| 对外 Service | `ffformer-public` | LoadBalancer,MetalLB 绑公网 |
| PVC | `ffformer-pvc` | 持久卷,挂 `/workspace/data`(模型权重等) |
| Secret | `ffformer-hpc-ssh` | 连超算用的 SSH 私钥 |

## 8.1 登录 Rancher,确认 Namespace

1. 浏览器打开 Rancher URL(向组内基础设施负责人要),登录。
2. 选集群,左菜单 **Cluster → Projects/Namespaces**。
3. 确认 **`<namespace>`** 已存在;若要全新区画才点 **Create Namespace**。

> **Namespace** = 同一栋楼(集群)里分出来的"房间",分开管理不打架。全程统一用 `<namespace>`。

## 8.2 登记镜像仓库的钥匙(Registry Secret)

让 Rancher 能从 Docker Hub 拉我们的镜像。若镜像是**公开**的可跳过;**私有**则需要:

1. 左菜单 **Storage → Secrets → Create → Registry**。
2. Namespace 选 `<namespace>`。
3. 仓库地址 `docker.io`,填 Docker Hub 用户名 `<YOUR_NAME>` + **Access Token**(不是登录密码,在 Docker Hub → Account Settings → Security 里生成)。
4. 命名如 `dockerhub-secret` 保存,并在 Deployment 的 `imagePullSecrets` 里引用。

> Token/密码 = 密码,别贴进代码、别外传。

## 8.3 登记连超算的 SSH 密钥(Secret)

前台要 SSH 到超算投作业,需把私钥作为 Secret 挂进 pod:

```bash
kubectl -n <namespace> create secret generic ffformer-hpc-ssh \
  --from-file=key=<你的HPC私钥路径>
```

Deployment 里通过 `HPC_KEY_PATH=/secrets/hpc/key` 引用(见 [07](07-K8sマニフェスト.md))。超算侧 `authorized_keys` 已限制来源 `from="<pod-egress-subnet>",no-pty`(pod 的 SNAT 网段)。

## 8.4 贴清单部署(Import YAML)

1. 右上角 **Import YAML**(`>_` 图标)。
2. Namespace 选 `<namespace>`。
3. 把 [07](07-K8sマニフェスト.md) 里的 `k8s-deployment.yaml` + `k8s-hpc-patch.yaml`(HPC 后端相关 env/挂载)整段贴进去(多文件用 `---` 分隔),点 **Import**。

> 报红了:把报错原文复制给 Claude Code,"Rancher 出这个错怎么修"。常见坑见下方和 [10](10-更新とトラブル対処.md)。

**⚠️ 本环境两个真实坑:**
- **PodSecurity `restricted`**:本 namespace 强制 restricted。pod 必须设 `securityContext`(`runAsNonRoot: true`、`allowPrivilegeEscalation: false`、`capabilities.drop: [ALL]`、`seccompProfile: RuntimeDefault`),否则 pod 起不来。
- 命令行操作前 `export KUBECONFIG=<你的 config>`;**token 会过期**(报 `system:unauthenticated`),过期就回 Rancher 重新下载 kubeconfig。

## 8.5 开公网入口(MetalLB + TLS)

对外访问链路:
```
用户 → 域名(HTTPS 443) →(中心 NAT)→ 中间IP <intermediate-ip>
     → MetalLB LoadBalancer(ffformer-public)→ ffformer-tls(nginx 终结 HTTPS)→ ffformer-infer-base:8000
```

`ffformer-public` Service 的关键:用注解 **`metallb.universe.tf/loadBalancerIPs: <intermediate-ip>`** 让 MetalLB 把中间 IP 绑上;`selector` 指向 `app=ffformer-tls`;端口 443→8443。

> 公网 IP / 中间 IP 是向中心「サービス公開申請」申请来的(放行来源:校园网段 `<campus-cidr>`)。若新 Service 外部 IP 一直 `<pending>`、MetalLB 报 `not allowed in config`,说明该中间 IP 还没进 MetalLB 池,需要中心配。

## 8.6 看是否跑起来

1. 左菜单 **Workloads**,找 `ffformer-infer-base`,状态变**绿(Active/Running)**即成功。
2. 不行就点进去,看 **Events** 标签和 **View Logs**(排错见 [10](10-更新とトラブル対処.md))。

命令行等价:
```bash
kubectl -n <namespace> get deploy ffformer-infer-base
kubectl -n <namespace> get pods -l app=ffformer-infer-base -o wide
kubectl -n <namespace> logs deploy/ffformer-infer-base | tail
```

pod 日志出现 `[entrypoint] Code updated to <commit>` 和 `Uvicorn running on http://0.0.0.0:8000` 就对了。

## ✔ 确认点

- ☐ **Workloads** 里 `ffformer-infer-base` 是绿的
- ☐ pod 日志有 `Uvicorn running on ...:8000`,且 `Code updated to <最新commit>`
- ☐ `ffformer-public` 有外部 IP(不是 `<pending>`)
- ☐ 浏览器开 `https://j-peaks-forestformer3d.hucc.hokudai.ac.jp/` 能看到登录页

> 下一章 [09](09-アクセスする.md):怎么访问、以及还没有入口时如何用 Port Forward 临时确认。
