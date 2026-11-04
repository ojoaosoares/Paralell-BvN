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
├── profile.py                    # Perfil do CloudLab (geni-lib): nó bare-metal c6420 + setup automático
└── scripts/
    ├── cloudlab/setup.sh         # Setup no boot do nó: dependências, perf, governor e build Release
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

### Perfil do CloudLab a partir deste repositório

O arquivo [`profile.py`](profile.py) define a topologia: um nó bare-metal `c6420`
(Clemson, Dual-Socket Intel Xeon Gold 6142, 384 GB de RAM) com Ubuntu 22.04.
No boot, o CloudLab clona o repositório em `/local/repository` e executa
[`scripts/cloudlab/setup.sh`](scripts/cloudlab/setup.sh), que instala as dependências
(`cmake`, `numactl`, `hwloc`, `perf`), libera o `perf` para usuários comuns,
fixa o governor da CPU em `performance` e compila o `bvn_bench` em Release.

1. No portal do CloudLab: **Experiments › Create Experiment Profile**.
2. Em **Source**, escolha **Git Repository** e informe
   `https://github.com/ojoaosoares/Paralell-BvN.git`.
3. Instancie o perfil. Parâmetros disponíveis: tipo de hardware, imagem de disco,
   rodar a suíte de benchmarks ao final do setup e governor `performance`.
4. Acompanhe o setup no nó com `tail -f /local/logs/bvn-setup.log`.

O binário fica em `/local/repository/build/bvn_bench`. A topologia detectada
(`lscpu`, `numactl --hardware`) é salva em `/local/logs/`: nós `c6420` podem vir
com 2 ou 4 nós NUMA, o que muda os resultados.

### Rodando a suíte manualmente

```bash
cd /local/repository
./scripts/run_cloudlab_bench.sh
```
