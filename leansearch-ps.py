#!/usr/bin/env python3
"""
LeanSearch-PS Setup Script

This script automatically sets up the LeanSearch-PS premise selection server.
It is idempotent and can be safely run multiple times.

Usage:
    python3 leansearch-ps.py [--install-dir PATH] [--port PORT] [--skip-tests]

Requirements:
    - Python 3.8+
    - git
    - ~20GB disk space
    - ~16GB RAM for running the server
"""

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @staticmethod
    def disable():
        """Disable colors for non-terminal output"""
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''
        Colors.END = ''


def print_step(msg: str):
    """Print a step message"""
    print(f"{Colors.BOLD}{Colors.BLUE}==>{Colors.END} {Colors.BOLD}{msg}{Colors.END}")


def print_success(msg: str):
    """Print a success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")


def print_warning(msg: str):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")


def print_error(msg: str):
    """Print an error message"""
    print(f"{Colors.RED}✗{Colors.END} {msg}", file=sys.stderr)


def run_command(cmd: list, cwd: Optional[Path] = None, check: bool = True,
                capture_output: bool = False, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Run a shell command with error handling"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.CalledProcessError as e:
        if capture_output:
            print_error(f"Command failed: {' '.join(cmd)}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
        raise
    except subprocess.TimeoutExpired:
        print_error(f"Command timed out: {' '.join(cmd)}")
        raise


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def find_server_processes(repo_dir: Path) -> List[int]:
    """Find PIDs of running server.py processes"""
    server_script = repo_dir / "LeanSearch-PS-inference" / "server.py"
    pids = []

    try:
        if platform.system() == "Windows":
            # Windows: use wmic or tasklist
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                capture_output=True,
                text=True,
                check=False
            )
            # Parse output to find processes running server.py
            # This is simplified - full implementation would parse CSV
        else:
            # Unix-like: use ps and grep
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                check=False
            )
            for line in result.stdout.split('\n'):
                if 'server.py' in line and str(server_script) in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pids.append(int(parts[1]))
                        except ValueError:
                            pass
    except Exception:
        pass

    return pids


def kill_existing_servers(repo_dir: Path, port: int) -> bool:
    """Kill existing server processes"""
    pids = find_server_processes(repo_dir)

    if not pids and not is_port_in_use(port):
        return True

    if pids:
        print(f"{Colors.YELLOW}Found {len(pids)} running server process(es){Colors.END}")
        for pid in pids:
            try:
                print(f"  Killing process {pid}...")
                if platform.system() == "Windows":
                    subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=False)
                else:
                    os.kill(pid, signal.SIGTERM)
                    # Wait a bit for graceful shutdown
                    time.sleep(1)
                    # Force kill if still running
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Already dead
            except Exception as e:
                print_error(f"Failed to kill process {pid}: {e}")

        # Wait for port to be released
        for _ in range(10):
            if not is_port_in_use(port):
                print_success("Existing server processes stopped")
                return True
            time.sleep(0.5)

    if is_port_in_use(port):
        print_error(f"Port {port} is still in use. Please stop the process manually or use a different port.")
        return False

    return True


def check_dependencies() -> Tuple[bool, list]:
    """Check if required system dependencies are installed"""
    print_step("Checking system dependencies")

    missing = []
    deps = {
        'git': ['git', '--version'],
        'python3': [sys.executable, '--version'],
    }

    for name, cmd in deps.items():
        try:
            run_command(cmd, capture_output=True)
            print_success(f"{name} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_error(f"{name} is not installed")
            missing.append(name)

    return len(missing) == 0, missing


def ensure_pip_in_venv(venv_python: Path):
    """Ensure pip is available in the virtual environment"""
    # First, check if pip is already available
    try:
        run_command([str(venv_python), "-m", "pip", "--version"], capture_output=True)
        return  # pip is available, nothing to do
    except subprocess.CalledProcessError:
        pass  # pip not available, need to install it

    print_step("Installing pip in virtual environment")

    # Try ensurepip first (built-in method)
    try:
        run_command([str(venv_python), "-m", "ensurepip", "--upgrade"], capture_output=True)
        print_success("pip installed via ensurepip")
        return
    except subprocess.CalledProcessError:
        pass  # ensurepip failed, try get-pip.py

    # Fallback: download and run get-pip.py
    print("ensurepip not available, downloading get-pip.py...")
    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as tmp:
            tmp_path = tmp.name
            urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', tmp_path)

        run_command([str(venv_python), tmp_path], capture_output=True)
        os.unlink(tmp_path)
        print_success("pip installed via get-pip.py")
    except Exception as e:
        print_error(f"Failed to install pip: {e}")
        raise


def setup_virtual_env(repo_dir: Path) -> Path:
    """Create and setup Python virtual environment"""
    venv_dir = repo_dir / "venv"

    if venv_dir.exists():
        print_success(f"Virtual environment already exists at {venv_dir}")
        # Even if venv exists, ensure pip is available
        venv_python = get_venv_python(venv_dir)
        ensure_pip_in_venv(venv_python)
        return venv_dir

    print_step(f"Creating virtual environment at {venv_dir}")
    run_command([sys.executable, "-m", "venv", str(venv_dir)])

    # Ensure pip is available in the venv (some systems don't include it by default)
    venv_python = get_venv_python(venv_dir)
    ensure_pip_in_venv(venv_python)

    print_success("Virtual environment created")

    return venv_dir


def get_venv_python(venv_dir: Path) -> Path:
    """Get the path to the Python executable in the virtual environment"""
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"


def install_python_deps(venv_python: Path):
    """Install required Python packages"""
    print_step("Installing Python dependencies")

    packages = [
        "transformers==4.52.4",
        "faiss-cpu",
        "flask",
        "requests",
        "torch",
        "torchvision",
        "peft",
        "accelerate",
        "huggingface_hub",
    ]

    # Check if packages are already installed
    try:
        result = run_command(
            [str(venv_python), "-m", "pip", "list", "--format=json"],
            capture_output=True
        )
        installed = {pkg['name'].lower() for pkg in json.loads(result.stdout)}

        to_install = [pkg for pkg in packages if pkg.split('==')[0].split('[')[0].lower() not in installed]

        if not to_install:
            print_success("All Python packages already installed")
            return
    except:
        to_install = packages

    print(f"Installing {len(to_install)} packages: {', '.join(to_install)}")

    run_command(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True
    )

    run_command(
        [str(venv_python), "-m", "pip", "install"] + to_install,
        timeout=1800  # 30 minutes for large packages
    )

    print_success("Python dependencies installed")


def clone_repository(install_dir: Path) -> Path:
    """Clone the REAL-Prover repository"""
    repo_dir = install_dir / "REAL-Prover"

    if repo_dir.exists() and (repo_dir / ".git").exists():
        print_success(f"Repository already cloned at {repo_dir}")
        return repo_dir

    print_step("Cloning REAL-Prover repository")

    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    run_command(
        ["git", "clone", "https://github.com/frenzymath/REAL-Prover.git", str(repo_dir)],
        timeout=300
    )

    print_success("Repository cloned")
    return repo_dir


def verify_file_size(file_path: Path, min_size_mb: int) -> bool:
    """Verify a file exists and is at least min_size_mb megabytes"""
    if not file_path.exists():
        return False
    size_mb = file_path.stat().st_size / (1024 * 1024)
    return size_mb >= min_size_mb


def download_with_hf_hub(venv_python: Path, repo_id: str, local_dir: Path):
    """Download a HuggingFace repository using the venv Python"""
    # Create a Python script to run with venv Python
    download_script = f"""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{repo_id}",
    local_dir="{local_dir}",
    local_dir_use_symlinks=False
)
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write(download_script)
        tmp_path = tmp.name

    try:
        run_command([str(venv_python), tmp_path], timeout=3600)
    finally:
        os.unlink(tmp_path)


def download_models(repo_dir: Path, venv_python: Path):
    """Download pre-trained models and FAISS index"""
    models_dir = repo_dir / "LeanSearch-PS-inference" / "models"
    models_dir.mkdir(exist_ok=True)

    # Check if models are already downloaded
    model_dir = models_dir / "LeanSearch-PS"
    faiss_dir = models_dir / "LeanSearch-PS-faiss"
    faiss_index = faiss_dir / "LeanSearch-PS-faiss.index"

    # Download LeanSearch-PS model
    if model_dir.exists() and (model_dir / "adapter_config.json").exists():
        print_success("LeanSearch-PS model already downloaded")
    else:
        print_step("Downloading LeanSearch-PS model (~42MB)")
        if model_dir.exists():
            shutil.rmtree(model_dir)

        download_with_hf_hub(venv_python, "FrenzyMath/LeanSearch-PS", model_dir)
        print_success("LeanSearch-PS model downloaded")

    # Download FAISS index using HuggingFace Hub (automatic progress bar)
    needs_download = True
    if faiss_dir.exists() and verify_file_size(faiss_index, 3000):  # At least 3GB
        print_success("FAISS index already downloaded")
        needs_download = False

    if needs_download:
        print_step("Downloading FAISS index (~4GB with progress bar)")
        if faiss_dir.exists():
            shutil.rmtree(faiss_dir)

        # Use HuggingFace Hub for reliable download with progress
        download_with_hf_hub(venv_python, "FrenzyMath/LeanSearch-PS-faiss", faiss_dir)

        # Verify the download
        if not verify_file_size(faiss_index, 3000):
            raise RuntimeError(f"FAISS index file is too small ({faiss_index.stat().st_size / (1024*1024):.1f} MB). Download may have failed.")

        print_success("FAISS index downloaded")


def configure_server(repo_dir: Path):
    """Configure server paths and settings"""
    print_step("Configuring server")

    config_file = repo_dir / "LeanSearch-PS-inference" / "conf" / "config.py"
    models_dir = repo_dir / "LeanSearch-PS-inference" / "models"

    config_content = f'''"""
This module provide configurations.
"""
INDEX_PATH = "{models_dir / 'LeanSearch-PS-faiss' / 'LeanSearch-PS-faiss.index'}"
TOKENIZER_PATH = "intfloat/e5-mistral-7b-instruct"
MODEL_PATH = "{models_dir / 'LeanSearch-PS'}"
ANSWER_PATH = "{models_dir / 'LeanSearch-PS-faiss' / 'answers.json'}"


EXPIRE_TIME = 5 * 60

TEST_QUERY = \'\'\'G : Type u_1
      inst✝ : Group G
      a b c : G
      ⊢ (fun x => a * x * b = c) (a⁻¹ * c * b⁻¹) ∧ ∀ (y : G), (fun x => a * x * b = c) y → y = a⁻¹ * c * b⁻¹\'\'\'


'''

    config_file.write_text(config_content)
    print_success("Server configured")


def patch_for_compatibility(repo_dir: Path, venv_python: Path):
    """Patch code for cross-platform compatibility"""
    print_step("Applying compatibility patches")

    premise_selector = repo_dir / "LeanSearch-PS-inference" / "worker" / "premise_selector.py"
    content = premise_selector.read_text()

    # Check if already patched
    if "torch.backends.mps.is_available()" in content:
        print_success("Compatibility patches already applied")
        return

    # Patch device selection
    old_device = "    def __init__(self):\n        self.device = 'cuda'"
    new_device = """    def __init__(self):
        # Use MPS if available (macOS), otherwise CPU
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f'Using device: {self.device}')"""

    content = content.replace(old_device, new_device)

    # Patch PEFT model loading
    old_import = "from transformers import AutoTokenizer, AutoModel\nimport faiss"
    new_import = "from transformers import AutoTokenizer, AutoModel\nfrom peft import PeftModel\nimport faiss"
    content = content.replace(old_import, new_import)

    old_model_load = """    def init_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(conf.config.TOKENIZER_PATH)
            self.model = AutoModel.from_pretrained(conf.config.MODEL_PATH).half()
            self.model.to(self.device)"""

    new_model_load = """    def init_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(conf.config.TOKENIZER_PATH)
            # Load base model first
            base_model = AutoModel.from_pretrained(conf.config.TOKENIZER_PATH)
            # Load PEFT adapter on top of base model
            self.model = PeftModel.from_pretrained(base_model, conf.config.MODEL_PATH)
            # Only use half precision on CUDA
            if self.device == 'cuda':
                self.model = self.model.half()
            self.model.to(self.device)"""

    content = content.replace(old_model_load, new_model_load)

    # Patch GPU cache clearing
    old_cache = """            # Release cuda memory
            del batch_dict
            del outputs

        # Release cuda memory
        torch.cuda.empty_cache()

        return embeddings"""

    new_cache = """            # Release memory
            del batch_dict
            del outputs

        # Release GPU memory cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()

        return embeddings"""

    content = content.replace(old_cache, new_cache)

    premise_selector.write_text(content)
    print_success("Compatibility patches applied")


def create_startup_script(repo_dir: Path, port: int) -> Path:
    """Create a startup script for the server"""
    startup_script = repo_dir / "LeanSearch-PS-inference" / "start_server.sh"

    if platform.system() == "Windows":
        startup_script = startup_script.with_suffix(".bat")
        content = f"""@echo off
cd /d "%~dp0"
call venv\\Scripts\\activate
set KMP_DUPLICATE_LIB_OK=TRUE
python server.py
"""
    else:
        content = f"""#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
python server.py
"""

    startup_script.write_text(content)

    if platform.system() != "Windows":
        startup_script.chmod(0o755)

    print_success(f"Startup script created at {startup_script}")
    return startup_script


def run_tests(venv_python: Path, repo_dir: Path, port: int) -> bool:
    """Run self-tests to verify installation"""
    print_step("Running self-tests")

    # Create temporary test file to avoid Unicode issues with -c flag
    import tempfile

    test_script = f'''# -*- coding: utf-8 -*-
import requests
import time
import sys

url = 'http://localhost:{port}/retrieve_premises'

# Test cases
tests = [
    {{
        'name': 'Simple Nat arithmetic',
        'query': 'n : Nat\\n⊢ n + 0 = n',
        'expected_theorem': 'Nat.add_zero'
    }},
    {{
        'name': 'Topological monoid action',
        'query': """M : Type u_1
α : Type u_2
inst✝³ : TopologicalSpace α
inst✝² : Monoid M
inst✝¹ : MulAction M α
inst✝ : ContinuousConstSMul M α
s : Set M
t : Set α
⊢ ⋃ a ∈ s, a • closure t ⊆ closure (⋃ a ∈ s, a • t)""",
        'expected_theorem': 'smul_closure_subset'
    }}
]

print("Waiting for server to be ready...")
max_retries = 30
for i in range(max_retries):
    try:
        response = requests.post(url, json={{'query': 'test', 'num': 1}}, timeout=5)
        if response.status_code in [200, 400]:
            print("✓ Server is responding")
            break
    except:
        pass
    time.sleep(2)
else:
    print("✗ Server did not start in time")
    sys.exit(1)

# Run tests
all_passed = True
for test in tests:
    print(f"\\nTesting: {{test['name']}}")
    start = time.time()
    try:
        response = requests.post(url, json={{'query': test['query'], 'num': 10}}, timeout=60)
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            results = data['data'][0]
            found = any(r['Formal name'] == test['expected_theorem'] for r in results)

            if found:
                print(f"  ✓ Found {{test['expected_theorem']}} in {{elapsed:.2f}}s")
            else:
                print(f"  ✗ Did not find {{test['expected_theorem']}}")
                print(f"    Got: {{[r['Formal name'] for r in results[:3]]}}")
                all_passed = False
        else:
            print(f"  ✗ Request failed with status {{response.status_code}}")
            all_passed = False
    except Exception as e:
        print(f"  ✗ Test failed: {{e}}")
        all_passed = False

if all_passed:
    print("\\n✓ All tests passed!")
    sys.exit(0)
else:
    print("\\n✗ Some tests failed")
    sys.exit(1)
'''

    # Kill any existing servers first
    if not kill_existing_servers(repo_dir, port):
        return False

    # Start server in background
    print("Starting server for tests...")
    server_script = repo_dir / "LeanSearch-PS-inference" / "server.py"

    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    server_process = subprocess.Popen(
        [str(venv_python), str(server_script)],
        cwd=repo_dir / "LeanSearch-PS-inference",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Write test script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(test_script)
            test_file = f.name

        try:
            # Run tests from file
            result = run_command(
                [str(venv_python), test_file],
                check=False,
                timeout=300
            )

            success = result.returncode == 0

            if success:
                print_success("All self-tests passed!")
            else:
                print_error("Some self-tests failed")

            return success
        finally:
            # Clean up test file
            Path(test_file).unlink(missing_ok=True)

    finally:
        # Stop server
        print("Stopping test server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()


def start_server(venv_python: Path, repo_dir: Path, port: int):
    """Start the LeanSearch-PS server"""
    print_step(f"Starting server on port {port}")

    # Kill any existing servers first
    if not kill_existing_servers(repo_dir, port):
        print_error("Failed to stop existing servers. Exiting.")
        sys.exit(1)

    server_script = repo_dir / "LeanSearch-PS-inference" / "server.py"
    server_cwd = repo_dir / "LeanSearch-PS-inference"

    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print(f"\n{Colors.BOLD}Server starting...{Colors.END}")
    print(f"  URL: http://localhost:{port}/retrieve_premises")
    print(f"  Press {Colors.BOLD}Ctrl+C{Colors.END} to stop\n")

    try:
        # Start server in foreground (not background)
        subprocess.run(
            [str(venv_python), str(server_script)],
            cwd=server_cwd,
            env=env,
            check=True
        )
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Server stopped{Colors.END}")
    except Exception as e:
        print_error(f"Server error: {e}")
        raise


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup LeanSearch-PS premise selection server")
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=Path.cwd(),
        help="Installation directory (default: current directory)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running self-tests"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start the server after installation (default: start server)"
    )

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("=" * 70)
    print("  LeanSearch-PS Setup")
    print("  Semantic premise selection for Lean theorem proving")
    print("=" * 70)
    print(f"{Colors.END}\n")

    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print_error(f"Missing dependencies: {', '.join(missing)}")
        print("\nPlease install missing dependencies and try again.")
        return 1

    # Create installation directory
    install_dir = args.install_dir.expanduser().absolute()
    install_dir.mkdir(parents=True, exist_ok=True)
    print(f"Installation directory: {install_dir}\n")

    try:
        # Setup steps
        repo_dir = clone_repository(install_dir)

        venv_dir = setup_virtual_env(repo_dir)
        venv_python = get_venv_python(venv_dir)

        install_python_deps(venv_python)

        download_models(repo_dir, venv_python)

        configure_server(repo_dir)

        patch_for_compatibility(repo_dir, venv_python)

        startup_script = create_startup_script(repo_dir, args.port)

        print(f"\n{Colors.BOLD}{Colors.GREEN}Installation complete!{Colors.END}\n")

        # Run tests
        if not args.skip_tests:
            test_passed = run_tests(venv_python, repo_dir, args.port)
            if not test_passed:
                print_warning("Tests failed, but installation is complete")

        # Print usage instructions
        print(f"\n{Colors.BOLD}Usage:{Colors.END}")
        print(f"  Start server: {startup_script}")
        print(f"  Or manually:  cd {repo_dir / 'LeanSearch-PS-inference'}")
        print(f"                source venv/bin/activate")
        print(f"                KMP_DUPLICATE_LIB_OK=TRUE python server.py")
        print(f"\n  Server URL: http://localhost:{args.port}/retrieve_premises")
        print(f"\n{Colors.BOLD}Example query:{Colors.END}")
        print(f'''  curl -X POST http://localhost:{args.port}/retrieve_premises \\
    -H "Content-Type: application/json" \\
    -d '{{"query": "n : Nat\\n⊢ n + 0 = n", "num": 5}}'
''')

        # Start server if requested
        if not args.no_start_server:
            print(f"\n{Colors.BOLD}{Colors.GREEN}Starting server...{Colors.END}")
            start_server(venv_python, repo_dir, args.port)

        return 0

    except KeyboardInterrupt:
        print_error("\n\nSetup interrupted by user")
        return 130
    except Exception as e:
        print_error(f"\n\nSetup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
