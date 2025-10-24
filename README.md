# LeanSearch-PS Setup Script

Automated setup script for the [REAL-Prover](https://github.com/frenzymath/REAL-Prover) LeanSearch-PS premise selection server.

## Quick Start

**One-line installation:**

```bash
curl -sSL https://raw.githubusercontent.com/kim-em/run_leansearch_ps/master/leansearch-ps.py | python3 -
```

This downloads and runs the setup script, which installs everything to the current directory and starts the server.

**Or download and run manually:**

```bash
# Download the script
curl -O https://raw.githubusercontent.com/kim-em/run_leansearch_ps/master/leansearch-ps.py

# Run with default settings (installs to current directory and starts server)
python3 leansearch-ps.py

# Or specify custom install directory
python3 leansearch-ps.py --install-dir /path/to/install

# Skip tests (faster)
python3 leansearch-ps.py --skip-tests

# Just install, don't start server
python3 leansearch-ps.py --no-start-server
```

**Note:** By default, the script will start the server after installation. Press Ctrl+C to stop it.

## What It Does

The script automatically:

1. ✓ **Checks dependencies** (Python 3.8+, git)
2. ✓ **Creates virtual environment** (isolated Python packages)
3. ✓ **Installs Python packages** (transformers, PyTorch, FAISS, etc.)
4. ✓ **Clones REAL-Prover repository** from GitHub
5. ✓ **Downloads models** (~15GB total):
   - e5-mistral-7b-instruct base model
   - LoRA fine-tuning adapter (~42MB)
   - FAISS index (~4GB)
6. ✓ **Configures server** for your system (macOS/Linux/Windows)
7. ✓ **Applies compatibility patches** (MPS/CUDA/CPU support)
8. ✓ **Creates startup script** for easy server launching
9. ✓ **Runs self-tests** to verify installation
10. ✓ **Starts the server** automatically (unless --no-start-server is used)

## Requirements

- **Python:** 3.8 or newer
- **Disk space:** ~20GB free
- **RAM:** ~16GB recommended for running the server
- **Internet:** For downloading models
- **OS:** macOS, Linux, or Windows

## Installation Time

- **First run:** 30-60 minutes (downloads ~15GB)
- **Subsequent runs:** < 1 minute (idempotent, skips existing files)

## Usage

### Command Line Options

```bash
python3 leansearch-ps.py [OPTIONS]

Options:
  --install-dir PATH   Installation directory (default: .)
  --port PORT          Server port (default: 8080)
  --skip-tests         Skip running self-tests
  --no-start-server    Don't start server after installation
  --no-color           Disable colored output
  -h, --help           Show help message
```

### After Installation

Start the server:

```bash
# Use the generated startup script
./REAL-Prover/LeanSearch-PS-inference/start_server.sh

# Or manually:
cd ./REAL-Prover/LeanSearch-PS-inference
source ../../venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
python server.py
```

Query the server:

```bash
curl -X POST http://localhost:8080/retrieve_premises \
  -H "Content-Type: application/json" \
  -d '{"query": "n : Nat\n⊢ n + 0 = n", "num": 5}'
```

## Features

### Idempotent

Run the script multiple times safely - it skips already-completed steps:
- ✓ Won't re-download existing models
- ✓ Won't reinstall existing packages
- ✓ Won't re-clone existing repository

### Cross-Platform

Automatically detects and configures for:
- **macOS:** Uses MPS (Metal Performance Shaders) acceleration
- **Linux with NVIDIA GPU:** Uses CUDA acceleration
- **Linux/Windows without GPU:** Uses CPU (slower but works)

### Self-Testing

Includes automated tests:
1. **Simple query:** `n : Nat ⊢ n + 0 = n`
   - Expects: `Nat.add_zero`

2. **Complex query:** Topological monoid action with closures
   - Expects: `smul_closure_subset`

Both tests verify:
- Server starts successfully
- Models load correctly
- Queries return relevant results
- Response time is reasonable

## Performance

After the first query (cold start ~11s), typical performance:

- **Average:** 200-300ms per query
- **Best case:** 200ms
- **Worst case:** <1s
- **Throughput:** 3-5 queries/second

## Troubleshooting

### Out of memory during model download

The base model is ~15GB. If download fails:
```bash
# Clear partial downloads and retry
rm -rf ~/.cache/huggingface/hub/models--intfloat--e5-mistral-7b-instruct
python3 leansearch-ps.py
```

### Server fails to start

Check available RAM:
```bash
# macOS
vm_stat

# Linux
free -h
```

The 7B parameter model needs ~14GB RAM when loaded.

### OpenMP library conflict (macOS)

The script automatically sets `KMP_DUPLICATE_LIB_OK=TRUE` to handle this.
If you see warnings, they're safe to ignore.

### Tests fail but installation succeeds

This is usually fine - the server works but tests may timeout on slower machines.
You can verify manually by starting the server and querying it.

## File Locations

Default installation structure:

```
./
├── REAL-Prover/
│   ├── venv/                      # Python virtual environment
│   └── LeanSearch-PS-inference/
│       ├── models/
│       │   ├── LeanSearch-PS/     # LoRA adapter (42MB)
│       │   └── LeanSearch-PS-faiss/
│       │       ├── LeanSearch-PS-faiss.index  (4GB)
│       │       └── answers.json   (232MB)
│       ├── server.py
│       ├── start_server.sh        # Generated startup script
│       └── conf/config.py         # Auto-configured
└── ~/.cache/huggingface/hub/
    └── models--intfloat--e5-mistral-7b-instruct/  (15GB)
```

## Examples

### Example 1: Simple Arithmetic

Query:
```json
{
  "query": "n : Nat\n⊢ n + 0 = n",
  "num": 5
}
```

Response includes:
- `Nat.add_zero`: ∀ (n : Nat), n + 0 = n
- `Nat.zero_add`: ∀ (n : Nat), 0 + n = n
- `Nat.add_comm`: ∀ (n m : Nat), n + m = m + n

### Example 2: Topology

Query:
```json
{
  "query": "M : Type u_1\nα : Type u_2\ninst✝³ : TopologicalSpace α\ninst✝² : Monoid M\ninst✝¹ : MulAction M α\ninst✝ : ContinuousConstSMul M α\ns : Set M\nt : Set α\n⊢ ⋃ a ∈ s, a • closure t ⊆ closure (⋃ a ∈ s, a • t)",
  "num": 10
}
```

Response includes:
- `smul_closure_subset`: Key lemma for scalar multiplication and closure
- `closure_mono`: Monotonicity of closure
- `Set.biUnion_mono`: Monotonicity of bounded unions
- And 7 more relevant theorems

## Citation

If you use LeanSearch-PS in research, please cite:

```bibtex
@misc{realprover2025,
  title={REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning},
  author={Ziju Shen and Naohao Huang and Fanyi Yang and Yutong Wang and Guoxiong Gao and Tianyi Xu and Jiedong Jiang and Wanyi He and Pu Yang and Mengzhou Sun and Haocheng Ju and Peihao Wu and Bryan Dai and Bin Dong},
  year={2025},
  eprint={2505.20613},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.20613}
}
```

## License

This setup script is provided as-is. The REAL-Prover project and models have their own licenses - see the [original repository](https://github.com/frenzymath/REAL-Prover) for details.
