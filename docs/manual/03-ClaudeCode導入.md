# Claude Code を導入する

> Claude Code 是在**终端里**跑的 AI 助手,用自然语言就能让它读改本项目的代码。装法有两种,初学者推荐**原生安装器**(不用 Node.js、自动更新)。
>
> 前提:已备好 Anthropic 付费计划或 API Key(见 [02](02-準備するもの.md))。

## 3.1 选安装方式

| 方式 | 特点 | 适合 |
|---|---|---|
| **原生安装器(推荐)** | 额外准备最少、自动更新到最新版 | 想最省事开始的人 |
| npm 方式 | 需要 Node.js,可自己管版本 | 已经在用 Node.js 的人 |

## 3.2 方式 A:原生安装器(推荐)

打开终端,按你的系统执行:

**macOS / Linux:**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows(PowerShell):**
```powershell
irm https://claude.ai/install.ps1 | iex
```

> **想用图形界面**:不习惯终端的话,也可以装 Claude 的桌面应用(macOS / Windows / Linux),无需终端即可使用 Claude Code。

## 3.3 方式 B:npm

先装 Node.js(22 以上),`node --version` 显示 v22+ 即可。然后:

```bash
npm install -g @anthropic-ai/claude-code
```

> **注意**:报错也**别加 `sudo`**。多数权限错误应通过换 Node.js 的装法(如用 nvm)解决,而不是 sudo。

## 3.4 启动与首次登录

本项目的代码在 `github.com/<YOUR_NAME>/ffformer`。**在项目目录里启动 Claude Code**,它就能直接读改本项目:

```bash
# 1. 克隆本项目(第一次)
git clone https://github.com/<YOUR_NAME>/ffformer.git
cd ffformer

# 2. 启动 Claude Code
claude
```

1. 首次会按屏幕提示登录(认证):浏览器打开,用 Anthropic 账号授权。
2. 出现提示符(等待输入的状态)就准备好了。
3. 之后直接用中文/日文/英文跟它说"帮我改 XX""这段报错什么意思",它就会读本项目的文件来干活。

> **全新项目**:如果是要从零建一个新应用(而不是改本项目),就 `mkdir myapp && cd myapp && claude`,在空目录里让它生成——参考 [04](04-アプリを作る.md)。本项目属于"已存在、拿来改",直接在 `ffformer` 目录里跑即可。

## ✔ 确认点

- ☐ 终端里打 `claude` 能进入对话界面
- ☐ 用中文跟它说话有回应
- ☐ 在 `ffformer` 目录里启动的 Claude Code,能列出/读到本项目的文件(比如让它"看一下 deploy/server.py")

> 下一章 [04](04-アプリを作る.md):怎么用 Claude Code 改代码 / 建新应用。
