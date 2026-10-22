# Parallel-BvN: Decomposição de Birkhoff-von Neumann via Coloração Euleriana Paralela

**Disciplina:** Programação e Arquitetura de Computadores Paralelos (DCC/UFMG)  
**Professor:** George Luiz Medeiros Teodoro  
**Aluno:** João Vitor Soares (Matrícula: 2023002138)  

Proposta completa disponível em: [`docs/PROPOSTA.md`](docs/PROPOSTA.md).

---

## Organização do Repositório

```text
├── CMakeLists.txt
├── docs/
│   └── PROPOSTA.md               # Proposta do Trabalho Prático (Foco em Arquitetura NUMA/Cache)
├── include/bvn/
│   ├── types.hpp                 # Estruturas base (AlignedCursor alignas(64), FlatEdge, Config)
│   ├── traffic.hpp               # Gerador de matrizes de tráfego (uniform, skewed, bursty)
│   ├── regularizer.hpp           # Regularização / Padding O(N^2) SIMD + O(N) two-pointer
│   ├── euler_trail.hpp           # Costura determinística lock-free de caminhadas abertas
│   ├── decomposer.hpp            # Interfaces de decomposição BvN (seq, omp_naive, omp_arch)
│   └── postprocessor.hpp         # Filtragem de tráfego fantasma e verificador matemático exato
├── src/
│   ├── main.cpp                  # CLI de benchmark e relatório de métricas
│   ├── traffic.cpp               # Implementação do gerador de matrizes de demanda
│   ├── regularizer.cpp           # Fase 1: Regularização de grau D = 2^k
│   ├── euler_trail.cpp           # Reconciliação de caminhadas abertas no super-grafo Euleriano
│   ├── decomposer_seq.cpp        # Fases 2 e 3: Baseline Sequencial em Flat CSR
│   ├── decomposer_naive.cpp      # Fases 2 e 3: OpenMP Ingênuo (expondo false sharing e task overhead)
│   ├── decomposer_arch.cpp       # Fases 2 e 3: OpenMP Orientado à Arquitetura (alignas(64), cut-off, NUMA)
│   └── postprocessor.cpp         # Fase 4: Limpeza lock-free de tráfego fantasma e validação
└── scripts/
    └── run_cloudlab_bench.sh     # Suite automatizada de benchmarks e perf stat para nós CloudLab
```

---

## Compilação e Execução Local

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Executar modo orientado à arquitetura (omp_arch)
./build/bvn_bench --mode omp_arch --n 512 --d 512 --threads 6 --cutoff 8

# Comparar com modo ingênuo (omp_naive) e sequencial (seq)
./build/bvn_bench --mode omp_naive --n 512 --d 512 --threads 6
./build/bvn_bench --mode seq --n 512 --d 512
```

## Execução Automatizada no CloudLab (Multi-Socket NUMA + `perf`)

Em um nó bare-metal do CloudLab (ex.: Dual-Socket AMD EPYC `c6525-100g`, `r6525` ou Intel Xeon `c6420`):

```bash
./scripts/run_cloudlab_bench.sh
```
