"""一键启动前后端"""
import argparse
import subprocess
import sys
import os

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
PORT = int(os.environ.get("PORT", "8000"))

# 解析命令行参数
_parser = argparse.ArgumentParser(description="启动 BigDogBarkProject")
_parser.add_argument(
    "--adapter",
    default="agent_adapter",
    choices=["agent_adapter", "agent_adapter_local_LLM"],
    help="选择 Agent 适配器：agent_adapter（DeepSeek）或 agent_adapter_local_LLM（本地 vLLM）",
)
_args = _parser.parse_args()


def kill_port(port: int):
    """杀掉占用指定端口的进程（跨平台）"""
    if sys.platform == "win32":
        # Windows: netstat 找 PID → taskkill
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                pid = parts[-1]
                try:
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True)
                    print(f"🔫 已杀掉端口 {port} 上的旧进程 (PID={pid})")
                except Exception:
                    pass
    else:
        # Linux / WSL: lsof 找 PID → kill
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split():
                    subprocess.run(["kill", "-9", pid], capture_output=True)
                    print(f"🔫 已杀掉端口 {port} 上的旧进程 (PID={pid})")
        except FileNotFoundError:
            pass  # lsof 没装就算了


def ensure_frontend_deps(npm: str):
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if os.path.isdir(node_modules):
        return
    print("未检测到 frontend/node_modules，正在安装前端依赖...")
    subprocess.check_call([npm, "install"], cwd=FRONTEND_DIR)


def main():
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    ensure_frontend_deps(npm)

    # 先杀掉旧进程，再启动后端（传入 AGENT_ADAPTER 环境变量）
    kill_port(PORT)
    env = os.environ.copy()
    env["AGENT_ADAPTER"] = _args.adapter
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app",
         "--host", "0.0.0.0", "--port", str(PORT), "--reload"],
        cwd=ROOT,
        env=env,
    )
    print(f"✅ 后端启动 (PID={backend.pid}) → http://localhost:8000")

    # 启动前端
    frontend = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=FRONTEND_DIR,
    )
    print(f"✅ 前端启动 (PID={frontend.pid}) → http://localhost:3000")

    print("\n按 Ctrl+C 停止所有服务")
    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("已停止")

if __name__ == "__main__":
    main()
